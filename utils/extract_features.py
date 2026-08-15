"""
패치 jpg/png → frozen tile encoder(backbone) feature 사전 추출 스크립트

--backbone resnet50 (기본): models/cnn_encoder.py::CNNEncoder(ResNet50 Lunit SwAV, 2048-dim)
--backbone uni        : models/uni_encoder.py::UNIEncoder(UNI ViT-L/16, 1024-dim, 224 리사이즈)

출력:
    <patches_root>/<slide_id>/{features.pt|features_uni.pt}   (N_patches, feature_dim) float32
    행 순서 = data.patch_utils.list_patch_paths()와 동일한 정렬 순서

사용법:
    python -m utils.extract_features                            # 기본: cptac, resnet50
    python -m utils.extract_features --dataset tcga --backbone uni
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent  # 프로젝트 루트
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DataConfig
from data.dataset import COHORT_PATHS
from data.patch_utils import (
    FEATURES_FILENAME, FEATURES_UNI_FILENAME, FEATURES_UNI2_FILENAME,
    PATCH_TRANSFORM, UNI_PATCH_TRANSFORM, UNI2_NATIVE_PATCH_TRANSFORM, list_patch_paths,
)
from models.cnn_encoder import CNNEncoder
from models.uni_encoder import UNIEncoder
from models.uni2_encoder import UNI2hEncoder
from utils import load_env, send_slack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# backbone별 배치 크기
BACKBONE_REGISTRY = {
    "resnet50": {
        "encoder_cls":  CNNEncoder,
        "transform":    PATCH_TRANSFORM,
        "out_filename": FEATURES_FILENAME,
        "batch_size":   8,
    },
    "uni": {
        "encoder_cls":  UNIEncoder,
        "transform":    UNI_PATCH_TRANSFORM,
        "out_filename": FEATURES_UNI_FILENAME,
        "batch_size":   32,
    },
    "uni2": {
        # UNI2-h(ViT-H/14, models/uni2_encoder.py) — 파이프라인이 뽑은 1024px@1.0MPP
        # 원본을 UNI와 같은 UNI_PATCH_TRANSFORM(512 리사이즈)으로 넣는다.
        "encoder_cls":  UNI2hEncoder,
        "transform":    UNI_PATCH_TRANSFORM,
        "out_filename": FEATURES_UNI2_FILENAME,
        "batch_size":   8,
    },
    "uni2native": {
        # UNI2-h 공식 스펙(256px@20x, ~0.5MPP)대로 재타일링한 타일에서 feature를 추출할 때 쓴다
        # 산출물은 scripts/reconcile_uni2native_features.py가 기존 patches 트리로 복사해온다.
        "encoder_cls":  UNI2hEncoder,
        "transform":    UNI2_NATIVE_PATCH_TRANSFORM,
        "out_filename": FEATURES_UNI2_FILENAME,
        "batch_size":   32,
    },
}


@torch.no_grad()
def _extract_node(encoder, patch_paths: list[Path], transform, batch_size: int) -> torch.Tensor:
    chunks = []
    for i in range(0, len(patch_paths), batch_size):
        batch = torch.stack([
            transform(Image.open(p).convert("RGB")) for p in patch_paths[i : i + batch_size]
        ]).to(DEVICE, non_blocking=True)
        # ResNet50(CNNEncoder)은 feature map을 반환해 별도 pool이 필요하고,
        # UNI(UNIEncoder)는 ViT라 backbone 출력이 이미 pooled (B, feature_dim)이다.
        raw = encoder.backbone(batch)
        pooled = encoder.pool(raw) if hasattr(encoder, "pool") else raw
        chunks.append(pooled.cpu())
    return torch.cat(chunks)


def extract_features_for_root(
    patches_root: Path, backbone: str = "resnet50", notify: bool = True,
    task_id: int = 0, num_tasks: int = 1,
) -> int:
    """
    patches_root 바로 아래의 각 디렉터리(슬라이드/노드 1개당 1폴더)에 feature 파일을 생성한다.
    이미 산출물이 있는 디렉터리는 skip한다.

    다른 전처리 파이프라인(예: data/wsi_preprocess.py)이 타일링 직후 같은 프로세스 안에서
    바로 이어 호출할 수 있도록 만든 진입점.

    task_id/num_tasks: data/wsi_preprocess.py --task-id/--num-tasks와 동일한 관례로 슬라이드
    목록을 num_tasks개로 나눠 그중 task_id번째 몫만 처리한다(HPC array job 샤딩용). 이미
    처리된 디렉터리는 그대로 skip하므로 샤딩과 재실행 skip 로직이 서로 안전하게 공존한다.

    Returns: 새로 추출한 디렉터리 수
    """
    start_time = datetime.now()

    spec = BACKBONE_REGISTRY[backbone]
    out_filename = spec["out_filename"]
    transform    = spec["transform"]
    batch_size   = spec["batch_size"]

    # encoder 생성 및 eval 모드
    encoder = BACKBONE_REGISTRY[backbone]["encoder_cls"](embed_dim=1, with_backbone=True).to(DEVICE)
    encoder.eval()
    encoder.requires_grad_(False)

    # 슬라이드/노드 디렉터리 목록을 task_id/num_tasks로 샤딩
    node_dirs = sorted(d for d in patches_root.iterdir() if d.is_dir())
    if num_tasks > 1:
        node_dirs = node_dirs[task_id::num_tasks]

    desc = f"Extracting {backbone} features"
    if num_tasks > 1:
        desc += f" (task {task_id}/{num_tasks})"
    node_dirs = tqdm(node_dirs, desc=desc, unit="node")

    done = 0
    for node_dir in node_dirs:
        out_path = node_dir / out_filename
        if out_path.exists():
            continue

        patch_paths = list_patch_paths(node_dir)
        if not patch_paths:
            continue

        features = _extract_node(encoder, patch_paths, transform, batch_size)
        # skip 판단이 out_path.exists()뿐이라, torch.save 도중 job이 죽으면 잘린 파일이
        # "이미 완료"로 오판될 수 있다 — 임시 파일에 쓴 뒤 원자적 rename으로 교체해
        # out_path가 존재하면 항상 완전히 쓰인 파일임을 보장한다.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(features, tmp_path)
        tmp_path.replace(out_path)
        done += 1

    elapsed = datetime.now() - start_time
    h, rem  = divmod(int(elapsed.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    print(f"완료: {done}개 노드 → {patches_root}/<slide_id>/{out_filename}")
    if notify:
        send_slack(
            f":white_check_mark: *Feature 추출 완료* ({backbone})\n"
            f"> 저장 위치: `{patches_root}/<slide_id>/{out_filename}`\n"
            f"> 처리 노드: *{done}개*\n"
            f"> 소요 시간: {h}h {m}m {s}s"
        )
    return done


def main():
    parser = argparse.ArgumentParser(description="패치 jpg/png → frozen tile encoder feature 사전 추출")
    parser.add_argument("--dataset", type=str, default="cptac", choices=["tcga", "cptac"])
    parser.add_argument("--backbone", type=str, default="resnet50", choices=list(BACKBONE_REGISTRY))
    parser.add_argument("--patches-root", type=str, default=None)
    parser.add_argument("--task-id",   type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    args = parser.parse_args()

    cfg = DataConfig()
    patches_root = Path(args.patches_root) if args.patches_root else Path(getattr(cfg, COHORT_PATHS[args.dataset].patches_root_attr))
    extract_features_for_root(patches_root / "tiles", backbone=args.backbone,
                               task_id=args.task_id, num_tasks=args.num_tasks)


if __name__ == "__main__":
    load_env()
    try:
        main()
    except Exception as e:
        send_slack(f":x: *Feature 추출 에러*\n```{type(e).__name__}: {e}```")
        raise

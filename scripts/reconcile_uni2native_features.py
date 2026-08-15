"""
preprocess_uni2native_retile_array_hpc.sh + extract_features_uni2native_array_hpc.sh 산출물
(data/patches_{tcga,cptac}_uni2native/tiles/<slide>/features_uni2.pt, 우리 자체 파이프라인으로
256px@0.5MPP 재타일링한 UNI2-h feature)을, 기존 patches 트리(data/patches_{tcga,cptac}/tiles/<slide>/)
아래 features_uni2native.pt + coords_uni2native.pt로 복사해온다.

uni2official(MahmoodLab 공식 feature)과 똑같이 patch grid가 기존 1024px 트리와 달라(개수/좌표
불일치) coords도 새 트리 자체 파일명(r####_c####)에서 파싱해 짝으로 저장한다 — 우리 자체 좌표
컨벤션(작은 grid index)을 그대로 쓰므로 uni2official의 4000배 스케일 문제는 없다.

사용법:
    python scripts/reconcile_uni2native_features.py
"""
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.patch_utils import (
    FEATURES_UNI2_FILENAME, FEATURES_UNI2NATIVE_FILENAME, COORDS_UNI2NATIVE_FILENAME,
    list_patch_paths, _parse_coord,
)

SRC_ROOTS = {
    "tcga":  _ROOT / "data" / "patches_tcga_uni2native" / "tiles",
    "cptac": _ROOT / "data" / "patches_cptac_uni2native" / "tiles",
}
DST_ROOTS = {
    "tcga":  _ROOT / "data" / "patches_tcga" / "tiles",
    "cptac": _ROOT / "data" / "patches_cptac" / "tiles",
}


def main():
    for tag, src_root in SRC_ROOTS.items():
        if not src_root.exists():
            print(f"{tag}: {src_root} 없음 — 스킵")
            continue
        slide_dirs = sorted(d for d in src_root.iterdir() if d.is_dir())
        dst_root = DST_ROOTS[tag]
        n_ok, n_skip_no_feat, n_skip_no_dst = 0, 0, 0
        for slide_dir in tqdm(slide_dirs, desc=tag):
            feat_path = slide_dir / FEATURES_UNI2_FILENAME
            if not feat_path.exists():
                n_skip_no_feat += 1
                continue
            dst_dir = dst_root / slide_dir.name
            if not dst_dir.exists():
                n_skip_no_dst += 1
                continue

            features = torch.load(feat_path, weights_only=True)
            patch_paths = list_patch_paths(slide_dir)
            coords = torch.tensor([_parse_coord(p.name) for p in patch_paths], dtype=torch.long)
            if len(coords) != len(features):
                raise RuntimeError(
                    f"{slide_dir}: feature 행 수({len(features)})가 patch 수({len(coords)})와 다릅니다."
                )
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()

            torch.save(features, dst_dir / FEATURES_UNI2NATIVE_FILENAME)
            torch.save(coords, dst_dir / COORDS_UNI2NATIVE_FILENAME)
            n_ok += 1
        print(f"{tag}: 변환 {n_ok}개, feature 없음(추출 미완료) {n_skip_no_feat}개, "
              f"기존 patches 트리에 대응 디렉토리 없음 {n_skip_no_dst}개")


if __name__ == "__main__":
    main()

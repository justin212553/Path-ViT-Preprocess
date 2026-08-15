"""
WSI 패치 공용 유틸리티 — 패치 파일명 좌표 파싱, 정렬된 패치 목록, 표준 patch transform.

data/dataset.py(WSISurvivalDataset)와 data/extract_features.py, data/fit_clusters.py 등
패치 단위로 동작하는 모듈들이 공통으로 재사용한다.
"""
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from torchvision import transforms
from tqdm import tqdm

PATCH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# UNI(ViT-L/16, models/uni_encoder.py)용 transform
UNI_PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# UNI2-h native용 transform
UNI2_NATIVE_PATCH_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 실시간 augmentation용 캐시 — 원본 타일을 학습 시작 시 1회만 512로 디코딩+리사이즈해 RAM에
# 캐싱해두고(build_tile_cache), 매 epoch은 그 위에서 flip/ColorJitter/GaussianBlur/정규화 같은
# 값싼 연산만 다시 적용한다. eval(val/test/external)도 train과 같은 512 해상도로 맞춰야 유효
# 배율(magnification)이 일치한다 — RAM 캐싱 없이 그때그때 디코딩+리사이즈한다(PATCH_TRANSFORM_512).
TILE_CACHE_SIZE = 512

PATCH_TRANSFORM_AUGMENTED_CACHED = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# eval(val/test/external, train_c_index 리포팅)용 — augmentation 없이 train과 동일한 512
# 해상도로 맞춘다. tile_cache 없이(RAM 캐싱 안 함) 그때그때 원본을 열어 리사이즈한다.
PATCH_TRANSFORM_512 = transforms.Compose([
    transforms.Resize((TILE_CACHE_SIZE, TILE_CACHE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# 512로 리사이즈한 JPEG 디스크 캐시 루트 — 슬라이드별로 미러링해 저장한다. 한 번 캐싱되면
# 이후 다른 seed/실행이 같은 슬라이드를 쓸 때 원본 디코딩+리사이즈를 건너뛰고 재사용한다.
TILE_DISK_CACHE_DIR = Path("data/tile_cache_512")


FEATURES_FILENAME      = "features.pt"       # data/extract_features.py 산출물 파일명(ResNet50/Lunit SwAV)
FEATURES_UNI_FILENAME  = "features_uni.pt"   # UNI 산출물
FEATURES_UNI2_FILENAME = "features_uni2.pt"  # UNI2-h(models/uni2_encoder.py) 산출물, 우리 파이프라인으로 추출
# MahmoodLab 공식 UNI2-h feature(256px@20x, 공식 스펙 그대로) — patch grid가 우리 자체 추출본과
# 달라 coords도 별도 파일로 저장한다(scripts/convert_uni2h_official_features.py).
FEATURES_UNI2OFFICIAL_FILENAME = "features_uni2official.pt"
COORDS_UNI2OFFICIAL_FILENAME   = "coords_uni2official.pt"
# 우리 raw WSI를 우리 파이프라인(data/wsi_preprocess.py --target-mpp 0.5 --tile-size 256)으로
# UNI2-h 공식 스펙에 맞춰 재타일링한 결과 — scripts/reconcile_uni2native_features.py 산출물.
FEATURES_UNI2NATIVE_FILENAME = "features_uni2native.pt"
COORDS_UNI2NATIVE_FILENAME   = "coords_uni2native.pt"
FEATURES_NORM_FILENAME = "features_norm.pt"  # Macenko stain-normalized + ResNet50 산출물 (utils/extract_features_stain_norm.py)

_COORD_RE = re.compile(r"r(\d+)_c(\d+)")


def _disk_cache_path(patch_path: Path, cache_dir: Path) -> Path:
    return cache_dir / patch_path.parent.name / (patch_path.stem + ".jpg")


def _decode_with_disk_cache(p: Path, cached_path: Path | None, size: int | None) -> Image.Image:
    """
    디스크 JPEG 캐시가 있으면 읽고, 없거나 깨져 있으면 원본을 디코딩해 캐시에 원자적으로 쓴다
    (임시 파일에 먼저 쓰고 os.replace로 교체 — 여러 프로세스가 동시에 같은 캐시 파일을 쓰거나
    읽어도 반쪽짜리 파일이 노출되지 않는다).
    """
    if cached_path is not None and cached_path.exists():
        try:
            with Image.open(cached_path) as img:
                img.load()  # exists()만으로는 못 잡는 반쪽짜리/깨진 파일을 여기서 걸러낸다
                return img.convert("RGB")
        except (OSError, ValueError):
            pass  # 깨진 캐시 파일 — 아래에서 원본부터 다시 디코딩해 복구한다

    with Image.open(p) as img:
        img = img.convert("RGB")
        if size is not None:
            img = img.resize((size, size), resample=Image.BILINEAR)

    if cached_path is not None:
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cached_path.with_name(f".{cached_path.name}.tmp{os.getpid()}")
        img.save(tmp_path, format="JPEG", quality=92)
        os.replace(tmp_path, cached_path)  # 원자적 교체 — 동시에 읽는 다른 프로세스가 반쪽을 못 봄

    return img


def build_tile_cache(
    patch_paths: list[Path],
    size: int | None = TILE_CACHE_SIZE,
    disk_cache_dir: Path | None = TILE_DISK_CACHE_DIR,
    workers: int = 1,
) -> dict[Path, Image.Image]:
    """
    patch_paths를 전부 디코딩해 RAM에 PIL Image로 캐싱한다(학습 시작 시 1회만 호출) —
    디코딩 비용을 매 epoch이 아니라 학습 전체 기간 중 딱 1회로 줄인다. val/test/external은
    이 함수로 캐싱하지 않고 PATCH_TRANSFORM_512로 그때그때 디코딩+리사이즈한다.

    disk_cache_dir가 주어지면(기본값 TILE_DISK_CACHE_DIR) 디코딩 결과를 JPEG으로 디스크에도
    남긴다 — 다음 실행이 같은 슬라이드를 다시 쓸 때 프리로드 단계를 단축한다.

    workers>1이면 ThreadPoolExecutor로 병렬 디코딩한다(기본 1=순차).
    """
    cache: dict[Path, Image.Image] = {}
    disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else None

    def _load_one(p: Path) -> tuple[Path, Image.Image]:
        cached_path = _disk_cache_path(p, disk_cache_dir) if disk_cache_dir is not None else None
        return p, _decode_with_disk_cache(p, cached_path, size)

    # mininterval=30 — 비-TTY 환경(SLURM 로그 등)에서 tqdm 갱신이 매번 새 줄로 쌓이는 것을 방지.
    if workers <= 1:
        for p in tqdm(patch_paths, desc="Preloading tiles", unit="tile", mininterval=30):
            _, img = _load_one(p)
            cache[p] = img
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for p, img in tqdm(executor.map(_load_one, patch_paths), total=len(patch_paths),
                                desc="Preloading tiles", unit="tile", mininterval=30):
                cache[p] = img
    return cache


class TileLRUCache:
    """
    patch_path -> PIL Image(RGB) LRU 캐시, 항목 개수 상한(maxsize) 있음 — build_tile_cache()의
    무제한 전량 RAM 캐싱 대신 RAM 사용량을 제한해야 하는 경우(train_multi.py) 쓰는 대안.

    캐시가 꽉 찬 상태에서 새 타일을 넣으면 가장 오래전에 접근된 항목부터 제거한다(OrderedDict
    move_to_end + popitem(last=False)). RAM에서 밀려나도 디스크 캐시(TILE_DISK_CACHE_DIR)에는
    남아있어 다음 접근이 원본 디코딩보다는 빠르다.
    """

    def __init__(
        self, maxsize: int, size: int | None = TILE_CACHE_SIZE,
        disk_cache_dir: Path | None = TILE_DISK_CACHE_DIR,
    ):
        self.maxsize = maxsize
        self.size = size
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else None
        self._cache: OrderedDict[Path, Image.Image] = OrderedDict()

    def _load(self, p: Path) -> Image.Image:
        cached_path = _disk_cache_path(p, self.disk_cache_dir) if self.disk_cache_dir is not None else None
        return _decode_with_disk_cache(p, cached_path, self.size)

    def get(self, p: Path) -> Image.Image:
        if p in self._cache:
            self._cache.move_to_end(p)
            return self._cache[p]
        img = self._load(p)
        self._cache[p] = img
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return img

    def preload(self, patch_paths: list[Path]) -> None:
        """
        maxsize까지만 채운다 — 전량 시도하지 않는다.
        """
        for p in tqdm(patch_paths[: self.maxsize], desc="Preloading tiles (LRU-bounded)",
                      unit="tile", mininterval=30):
            self.get(p)


def _parse_coord(name: str) -> tuple[int, int]:
    m = _COORD_RE.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def list_patch_paths(node_dir: Path) -> list[Path]:
    """
    슬라이드 디렉터리의 패치 파일을 정렬된 순서로 나열.

    data/extract_features.py가 features.pt를 만들 때도 이 순서를 그대로 써야
    캐싱된 feature 행(row)과 패치(coords)가 어긋나지 않는다.
    """
    return sorted(list(node_dir.glob("*.png")) + list(node_dir.glob("*.jpg")))

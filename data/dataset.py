"""
TCGA-PAAD / CPTAC-PDA WSI 생존(OS) 데이터셋 — 환자(case) 단위 MIL.

환자당 다중 슬라이드를 리스트로 묶는다. 슬라이드→환자 매핑은 preprocess.py 산출물인
slide_index_task*.csv의 case_id 컬럼을 그대로 쓴다.

각 아이템 = 환자(case) 1명이 보유한 모든 슬라이드 리스트(dict).
DataLoader는 batch_size=1 + collate_fn=lambda batch: batch[0] 로 사용해야 한다.

반환 형식(각 원소는 dict):
    patch_paths / features: precomputed 여부에 따라 둘 중 하나만 존재
    coords:      (N, 2) int64   [row, col]
    case_id:     str
    slide_id:    str
    dataset:     "tcga" | "cptac"
    OS_time:     (1,) float32
    OS_event:    (1,) int64   (1=사망, 0=생존/censored)
    age_years / sex_idx: with_clinical=True일 때만 존재
    rna:         with_rna=True일 때만 존재 ((G,) float32, 코호트 내부 z-score 정규화)

os_labels_{tcga,cptac}.csv에 없는 case는 제외한다. with_clinical=True면
clinical_{tcga,cptac}.csv에 없는 case, with_rna=True면 rna_{tcga,cptac}.csv에
없는 case도 추가로 제외된다.

train/val/test는 case 단위 6:2:2 stratified split이다(OS_event 기준, _stratified_case_split
참조). split="all"이면 코호트 전체를 external test로 쓸 수 있다(train.py --external).
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DataConfig
from data.patch_utils import (
    FEATURES_FILENAME, FEATURES_NORM_FILENAME, FEATURES_UNI_FILENAME, FEATURES_UNI2_FILENAME,
    FEATURES_UNI2OFFICIAL_FILENAME, COORDS_UNI2OFFICIAL_FILENAME,
    FEATURES_UNI2NATIVE_FILENAME, COORDS_UNI2NATIVE_FILENAME,
    PATCH_TRANSFORM, list_patch_paths, _parse_coord,
)

from models.clinical_encoder import SEX_TO_IDX, STAGE_FIELDS, encode_stage_value, encode_margin_value

FEATURES_FILENAME_BY_BACKBONE = {
    "resnet50":      FEATURES_FILENAME,                 # ResNet-50 Lunit
    "uni":           FEATURES_UNI_FILENAME,             # UNI-h
    "uni2":          FEATURES_UNI2_FILENAME,            # UNI2-h
    "resnet50_norm": FEATURES_NORM_FILENAME,            # Macenko stain-normalized
    "uni2official":  FEATURES_UNI2OFFICIAL_FILENAME,    # UNI2-h 공식 pretrained checkpoint
    "uni2native":    FEATURES_UNI2NATIVE_FILENAME,      # UNI2-h 자체 pretrained checkpoint
}

# feature 파일과 짝을 이루는 별도 coords 파일에서 읽어야 하는 backbone들 — _load_slide 참조.
_PAIRED_COORDS_FILENAME_BY_BACKBONE = {
    "uni2official": COORDS_UNI2OFFICIAL_FILENAME,
    "uni2native":   COORDS_UNI2NATIVE_FILENAME,
}

class CohortPaths(NamedTuple):
    os_labels:          Path
    clinical:           Path
    rna:                Path
    patches_root_attr:  str

COHORT_PATHS = {
    "tcga":  CohortPaths(
        os_labels=Path("data/os_labels_tcga.csv"),
        clinical=Path("data/clinical_tcga.csv"),
        rna=Path("data/rna_tcga.csv"),
        patches_root_attr="patches_root_tcga",
    ),
    "cptac": CohortPaths(
        os_labels=Path("data/os_labels_cptac.csv"),
        clinical=Path("data/clinical_cptac.csv"),
        rna=Path("data/rna_cptac.csv"),
        patches_root_attr="patches_root_cptac",
    ),
}

# INT1500 — data/select_rnaseq_genes.py --intersection 산출물(TCGA-only/CPTAC-only 순위 교집합 top1500 유전자 id).
INT1500_GENE_IDS_PATH = Path("data/rna_gene_selection_intersection/selected_genes_top_1500.csv")
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2  # 나머지 0.2는 test


def _load_slide_index(patches_root: Path) -> pd.DataFrame:
    """
    data/wsi_preprocess.py가 --num-tasks 샤드별로 나눠 쓴 slide_index_task*.csv를 모두 합친다.
    """
    paths = sorted(patches_root.glob("slide_index_task*.csv"))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _stratified_case_split(case_df: pd.DataFrame, seed: int) -> dict:
    """
    OS_event별로 case를 6:2:2(train/val/test)로 나눈다. (Single split)
    """
    rng = np.random.RandomState(seed)
    split_of_case = {}
    for _, group in case_df.groupby("OS_event"):
        case_ids = group.index.to_numpy().copy()
        rng.shuffle(case_ids)
        n       = len(case_ids)
        n_train = min(round(n * TRAIN_FRAC), n)
        n_val   = min(round(n * VAL_FRAC), n - n_train)
        for i, case_id in enumerate(case_ids):
            if i < n_train:
                split_of_case[case_id] = "train"
            elif i < n_train + n_val:
                split_of_case[case_id] = "val"
            else:
                split_of_case[case_id] = "test"
    return split_of_case


def _stratified_kfold_assignment(case_df: pd.DataFrame, seed: int, n_folds: int) -> dict:
    """
    OS_event별로 case를 n_folds개 fold에 균등 배정한다.
    """
    rng = np.random.RandomState(seed)
    fold_of_case = {}
    for _, group in case_df.groupby("OS_event"):
        case_ids = group.index.to_numpy().copy()
        rng.shuffle(case_ids)
        for i, case_id in enumerate(case_ids):
            fold_of_case[case_id] = i % n_folds
    return fold_of_case


def _stratified_binary_split(case_df: pd.DataFrame, seed: int, frac: float) -> dict:
    """
    OS_event별로 case를 frac:(1-frac) 비율로 "train"/"val" 둘로 나눈다.
    """
    rng = np.random.RandomState(seed)
    split_of_case = {}
    for _, group in case_df.groupby("OS_event"):
        case_ids = group.index.to_numpy().copy()
        rng.shuffle(case_ids)
        n       = len(case_ids)
        n_train = min(round(n * frac), n)
        for i, case_id in enumerate(case_ids):
            split_of_case[case_id] = "train" if i < n_train else "val"
    return split_of_case


def _kfold_case_split(case_df: pd.DataFrame, seed: int, n_folds: int, fold_idx: int) -> dict:
    """
    K-fold 교차검증 split. case_df를 n_folds개로 나눠 fold_idx번째를 test로 쓰고,
    나머지는 TRAIN_FRAC:VAL_FRAC 비율로 train/val로 나눈다.

    Args:
        case_df:  index=case_id, columns=["OS_event"]
        seed:     fold 배정과 train/val 재분할에 공통으로 쓰는 셔플 시드
        n_folds:  fold 개수
        fold_idx: 이번 호출에서 test로 쓸 fold 번호(0-based)
    Returns:
        {case_id: "train"|"val"|"test"}
    """
    fold_of_case = _stratified_kfold_assignment(case_df, seed, n_folds)
    is_test = case_df.index.map(lambda cid: fold_of_case[cid] == fold_idx)
    test_df, remaining_df = case_df[is_test], case_df[~is_test]

    split_of_case = {case_id: "test" for case_id in test_df.index}
    train_val_frac = TRAIN_FRAC / (TRAIN_FRAC + VAL_FRAC)
    split_of_case.update(_stratified_binary_split(remaining_df, seed, frac=train_val_frac))
    return split_of_case


class WSISurvivalDataset(Dataset):
    """
    Args:
        cfg:           DataConfig (patches_root_tcga/cptac, precomputed, seed)
        dataset:       "tcga" | "cptac"
        split:         "train" | "val" | "test" | "all" ("all"은 split 없이 dataset 전체 반환)
        transform:     패치에 적용할 transform (precomputed=False일 때만 사용)
        with_clinical: True면 clinical_{tcga,cptac}.csv를 case_id로 inner-join해
                       age_years/sex_idx, AJCC 병기(T/N/M)+grade(STAGE_FIELDS),
                       residual_disease(절제연)를 추가한다(미상 값은 -1).
        with_rna:      True면 rna_{tcga,cptac}.csv를 case_id로 inner-join해 rna를 추가한다.
                       컬럼은 INT1500_GENE_IDS_PATH로 추린 유전자만 쓴다.
        feature_backbone: precomputed=True일 때 읽을 캐싱된 feature 파일의 backbone —
                       "resnet50" | "uni" | "uni2" | "resnet50_norm" | "uni2official" | "uni2native".
        fold:          주어지면(0-based) K-fold split(_kfold_case_split)을 쓴다.
                       None(기본)이면 단일 6:2:2 split(_stratified_case_split).
        n_folds:       fold 개수(기본 5). fold=None이면 무시.

    아이템 단위 = 환자 1명. __getitem__은 그 환자가 가진 모든 슬라이드의 dict 리스트를 반환한다.
    """

    def __init__(
        self,
        cfg: DataConfig,
        dataset: str = "cptac",
        split: str = "train",
        transform=None,
        with_clinical: bool = False,
        with_rna: bool = False,
        feature_backbone: str = "resnet50",
        fold: int | None = None,
        n_folds: int = 5,
    ):
        self.transform        = transform or PATCH_TRANSFORM
        self.precomputed      = cfg.precomputed
        self.with_clinical    = with_clinical
        self.with_rna         = with_rna
        self.feature_backbone   = feature_backbone
        self.features_filename  = FEATURES_FILENAME_BY_BACKBONE[feature_backbone]
        self.root = Path(getattr(cfg, COHORT_PATHS[dataset].patches_root_attr))
        self.rna_lookup = {}

        slide_df = _load_slide_index(self.root)
        slide_df = slide_df[(slide_df["status"] == "ok") & (slide_df["n_tiles_kept"] > 0)].copy()
        slide_df["dataset"] = dataset

        os_df  = pd.read_csv(COHORT_PATHS[dataset].os_labels)
        merged = slide_df.merge(os_df[["case_id", "OS_time", "OS_event"]], on="case_id", how="inner")

        if with_clinical:
            clinical_cols = ["case_id", "age_years", "sex", *STAGE_FIELDS, "residual_disease"]
            clinical_df = pd.read_csv(COHORT_PATHS[dataset].clinical)[clinical_cols]
            merged = merged.merge(clinical_df, on="case_id", how="inner")

        if with_rna:
            target_ids = set(pd.read_csv(INT1500_GENE_IDS_PATH)["gene_id"])
            rna_df = pd.read_csv(COHORT_PATHS[dataset].rna)
            gene_cols  = [c for c in rna_df.columns if c in target_ids]
            rna_matrix = rna_df[gene_cols].to_numpy(dtype="float32")

            self.rna_lookup.update(zip(rna_df["case_id"], rna_matrix))
            merged = merged.merge(rna_df[["case_id"]], on="case_id", how="inner")

        has_patches = merged["slide_id"].apply(self._has_patches)
        all_items = merged[has_patches].reset_index(drop=True)

        if all_items.empty:
            joined = ["os_labels"]
            if with_clinical:
                joined.append("clinical")
            if with_rna:
                joined.append("rna")

        if split == "all":
            # external test용 — 코호트 전체를 split 없이 그대로 쓴다.
            self.items = all_items.reset_index(drop=True)
        else:
            case_df = all_items.groupby("case_id").agg(OS_event=("OS_event", "first"))
            if fold is not None:
                # K-fold — fold 번째를 test로, 나머지를 다시 60:20 비율로 train/val 배정
                split_of_case = _kfold_case_split(case_df, seed=cfg.seed, n_folds=n_folds, fold_idx=fold)
            else:
                # case 단위 6:2:2 stratified split — OS_event별로 seed 고정 셔플 후 배정
                split_of_case = _stratified_case_split(case_df, seed=cfg.seed)
            all_items["_split"] = all_items["case_id"].map(split_of_case)
            self.items = all_items[all_items["_split"] == split].reset_index(drop=True)

        self.cases = sorted(self.items["case_id"].unique())

    def __len__(self) -> int:
        return len(self.cases)

    def _has_patches(self, slide_id: str) -> bool:
        d = self.root / "tiles" / slide_id
        if self.precomputed:
            return (d / self.features_filename).exists()
        return (next(d.glob("*.jpg"), None) or next(d.glob("*.png"), None)) is not None

    def _load_slide(self, row) -> dict:
        slide_dir = self.root / "tiles" / row["slide_id"]

        if self.feature_backbone in _PAIRED_COORDS_FILENAME_BY_BACKBONE:
            # patch grid가 자체 JPG 추출본과 달라 짝을 이루는 coords 파일에서 직접 읽는다.
            patch_paths = None
            coords = torch.load(slide_dir / _PAIRED_COORDS_FILENAME_BY_BACKBONE[self.feature_backbone], weights_only=True)
        else:
            patch_paths = list_patch_paths(slide_dir)
            coords = torch.tensor(
                [_parse_coord(p.name) for p in patch_paths],
                dtype=torch.long,
            )
            coords[:, 0] -= coords[:, 0].min()
            coords[:, 1] -= coords[:, 1].min()

        item = {
            "coords":   coords,
            "case_id":  row["case_id"],
            "slide_id": row["slide_id"],
            "dataset":  row["dataset"],
            "OS_time":  torch.tensor([row["OS_time"]], dtype=torch.float32),
            "OS_event": torch.tensor([row["OS_event"]], dtype=torch.long),
        }

        if self.with_clinical:
            item["age_years"] = torch.tensor(row["age_years"], dtype=torch.float32)
            item["sex_idx"]   = torch.tensor(SEX_TO_IDX[row["sex"]], dtype=torch.long)
            for field in STAGE_FIELDS:
                ord_val = encode_stage_value(field, row[field])
                item[field] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)
            ord_val = encode_margin_value(row["residual_disease"])
            item["margin_ord"] = torch.tensor(-1 if ord_val is None else ord_val, dtype=torch.long)

        if self.with_rna:
            item["rna"] = torch.from_numpy(self.rna_lookup[row["case_id"]])

        if self.precomputed:
            features = torch.load(slide_dir / self.features_filename, weights_only=True)
            item["features"] = features
        else:
            item["patch_paths"] = patch_paths

        return item

    def __getitem__(self, idx: int) -> list:
        case_id   = self.cases[idx]
        case_rows = self.items[self.items["case_id"] == case_id]
        return [self._load_slide(row) for _, row in case_rows.iterrows()]

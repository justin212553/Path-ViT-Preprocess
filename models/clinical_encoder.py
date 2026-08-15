"""
ClinicalEncoder — 임상 정보(age, sex, [선택]병기(T/N/M)+grade) MLP 인코더

data/clinical_{tcga,cptac}.csv 의 age_years, sex 컬럼을 WSI 임베딩과 late-fusion할 수 있는
D차원 벡터로 변환한다. patch_vit_fusion.py의 ClusterHistogramBranch(Path B)와 같은 역할 —
서로 다른 모달리티(연속형 age, 범주형 sex)를 하나의 임베딩으로 합쳐 risk_head 입력에
concat할 수 있게 한다.

[입력 전처리]
  age : (age_years - mean) / std 로 z-score 정규화. mean/std는 학습 코호트 내부에서 계산해
        buffer로 고정한다 — extract_rna_clinical.py가 RNA feature에 적용한 "데이터셋 내부
        z-score 정규화" 관례와 동일하게, age_stats_from_csv()로 학습 코호트 clinical.csv에서
        직접 계산해 생성자에 전달한다.
  sex : male=0, female=1 이진 인코딩. 두 코호트(clinical_tcga.csv, clinical_cptac.csv) 모두
        male/female만 존재함을 확인했으므로 그 외 값은 지원하지 않는다.

[병기(staging) — use_staging=True, train.py --clinical-staging]
extract_rna_clinical.py가 raw clinical.tsv에서 추가로 뽑아둔 AJCC 병기(T/N/M)와 종양 등급
(grade)을 clinical branch 입력에 추가한다 - "age/sex만 쓰라"는 기존 지시가 있었기 때문에
(이유 불명) 이 옵션을 켰을 때/껐을 때 두 버전을 다 만들 수 있도록 age/sex 전용 ClinicalEncoder는
그대로 남기고 use_staging으로 선택하게 한다. T/N/M/grade는 전부 순서형 범주(예: T1<T2<T3<T4)라
one-hot 대신 정수 순서값을 age와 같은 방식으로 z-score 정규화한다. TX/NX/MX/GX(AJCC상 "판정
불가", 결측과 구분됨)와 실제 결측(NaN)은 둘 다 "미상"으로 취급해 평균값(z=0)으로 대체하고,
그 필드가 실제로 알려져 있었는지를 별도의 0/1 플래그로 함께 넣어 모델이 대체값과 진짜 관측값을
구분할 수 있게 한다(필드당 (z_score, known_flag) 2차원 x 4필드 = 8차원 추가).
"""
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

SEX_TO_IDX = {"male": 0, "female": 1}

# AJCC 병기/등급 문자열 -> 순서형 정수. 매핑에 없는 값(TX/NX/MX/GX 등 "판정 불가" 카테고리)과
# NaN(실제 결측)은 전부 encode_stage_value()가 None을 반환해 "미상"으로 통일 처리된다.
STAGE_FIELDS = ("ajcc_t", "ajcc_n", "ajcc_m", "tumor_grade")
_STAGE_ORDINAL_MAPS = {
    "ajcc_t": {"Tis": 0, "T1": 1, "T1a": 1, "T1b": 1, "T1c": 1, "T2": 2, "T3": 3, "T4": 4},
    "ajcc_n": {"N0": 0, "N1": 1, "N1a": 1, "N1b": 1, "N2": 2},
    "ajcc_m": {"M0": 0, "M1": 1, "M1a": 1, "M1b": 1},
    "tumor_grade": {"G1": 1, "G2": 2, "G3": 3, "G4": 4},
}
# ClinicalEncoder 버퍼 이름(짧게)과 StagePredictionHead가 참조하는 stage_stats 키를 연결.
_STAGE_BUFFER_NAMES = {"ajcc_t": "t", "ajcc_n": "n", "ajcc_m": "m", "tumor_grade": "grade"}

# 2026-08-04 추가 — 절제연 상태(residual_disease: R0=완전절제 < R1 < R2=잔존종양). STAGE_FIELDS
# (T/N/M/grade, --clinical-staging)와 별도의 독립 플래그(--clinical-margin, M5_R)로 둔다 —
# T/N/M/grade와 한 묶음으로 켜고 끄는 --clinical-staging에 합치면 margin 단독 효과를 격리해서
# 볼 수 없다. RX("판정 불가")는 TX/NX/MX/GX와 같은 관례로 "미상"(encode_margin_value가 None
# 반환) 처리.
_MARGIN_ORDINAL_MAP = {"R0": 0, "R1": 1, "R2": 2}


def encode_margin_value(raw) -> int | None:
    """residual_disease 원본 문자열(예: "R1")을 순서형 정수로 변환한다. encode_stage_value와
    동일한 "미상은 None" 관례."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return _MARGIN_ORDINAL_MAP.get(raw)


def margin_stats_from_df(df: pd.DataFrame) -> tuple[float, float]:
    """residual_disease 순서형 정수의 평균/표준편차(z-score 정규화용) — stage_stats_from_df와
    동일한 역할의 margin 전용 버전."""
    ordinals = df["residual_disease"].map(encode_margin_value).dropna().astype(float)
    return float(ordinals.mean()), float(ordinals.std(ddof=0))


def margin_stats_from_csv(csv_path: str | Path) -> tuple[float, float]:
    """clinical_{tcga,cptac}.csv 파일 하나에서 margin_stats_from_df를 계산하는 편의 함수."""
    return margin_stats_from_df(pd.read_csv(csv_path))


def age_stats_from_csv(csv_path: str | Path) -> tuple[float, float]:
    """clinical_{tcga,cptac}.csv의 age_years 평균/표준편차를 계산한다(z-score 정규화용)."""
    age_years = pd.read_csv(csv_path)["age_years"].astype(float)
    return float(age_years.mean()), float(age_years.std(ddof=0))


def encode_sex(sex: pd.Series | list[str]) -> torch.Tensor:
    """sex 컬럼(male/female 문자열)을 이진 인덱스(long) 텐서로 변환한다."""
    return torch.tensor([SEX_TO_IDX[s] for s in sex], dtype=torch.long)


def encode_stage_value(field: str, raw) -> int | None:
    """AJCC 병기/등급 원본 문자열(예: "T3")을 순서형 정수로 변환한다.

    TX/NX/MX/GX("판정 불가")나 NaN(실제 결측)처럼 _STAGE_ORDINAL_MAPS에 없는 값은 전부
    None("미상")을 반환한다 - 둘을 구분해봐야 예후 예측 관점에서 의미가 없다(둘 다 "이
    환자의 병기를 신뢰성 있게 알 수 없다"는 동일한 상태).
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    return _STAGE_ORDINAL_MAPS[field].get(raw)


def stage_stats_from_df(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """clinical 테이블(들)을 합친 DataFrame에서 STAGE_FIELDS별 순서형 정수의 평균/표준편차를
    계산한다(z-score 정규화용, age_stats_from_csv와 동일한 역할). "미상"(encode_stage_value가
    None을 반환하는 값)은 통계 계산에서 제외한다 - dataset="both"처럼 두 코호트를 합쳐 계산할
    때는 두 CSV를 pd.concat한 뒤 이 함수를 호출하면 된다(age_mean/age_std를 train.py가 "both"에서
    계산하는 방식과 동일한 관례)."""
    stats = {}
    for field in STAGE_FIELDS:
        ordinals = df[field].map(lambda v: encode_stage_value(field, v)).dropna().astype(float)
        stats[field] = (float(ordinals.mean()), float(ordinals.std(ddof=0)))
    return stats


def stage_stats_from_csv(csv_path: str | Path) -> dict[str, tuple[float, float]]:
    """clinical_{tcga,cptac}.csv 파일 하나에서 stage_stats_from_df를 계산하는 편의 함수."""
    return stage_stats_from_df(pd.read_csv(csv_path))


class ClinicalEncoder(nn.Module):
    """
    age/sex (2,) [+ 선택: T/N/M/grade (8,)] → 임베딩 (D,) 두 층 MLP.

    [학습 범위]
    age_mean/age_std, {t,n,m,grade}_mean/std : 고정 — 학습 코호트에서 사전 계산된 정규화 통계
    mlp                                      : 학습 — 위 통계로 정규화한 입력을 risk 예측에
                                                유용한 임베딩으로 변환

    [use_staging] True면 stage_stats(STAGE_FIELDS별 (mean, std), stage_stats_from_csv 참조)가
    필수다. 필드별로 (z_score, known_flag) 2차원씩 총 8차원을 age/sex 뒤에 이어붙인다 —
    known_flag가 있어 "미상이라 평균으로 대체된 값"과 "실제로 관측된 평균값"을 모델이
    구분할 수 있다.
    """

    def __init__(
        self, embed_dim: int, age_mean: float, age_std: float, hidden_dim: int = 64,
        use_staging: bool = False, stage_stats: dict[str, tuple[float, float]] | None = None,
        dropout: float = 0.0, sex_onehot: bool = False,
        use_margin: bool = False, margin_stats: tuple[float, float] | None = None,
        use_age_sex: bool = True,
    ):
        super().__init__()
        self.register_buffer("age_mean", torch.tensor(age_mean, dtype=torch.float32))
        self.register_buffer("age_std", torch.tensor(age_std, dtype=torch.float32))

        self.use_staging = use_staging
        # 2026-07-29: sex를 스칼라 이진 인덱스(0/1) 대신 레퍼런스(m2_pathology_clinical_mil.py::
        # ClinicalEmbedding, clinical_dim=3)처럼 one-hot(2차원)으로 인코딩하는 ablation
        # (train_light.py --sex-onehot, --M7 전용). 표현력은 수학적으로 동일하지만(선형 레이어
        # 뒤에서 sex_male+sex_female=1 제약으로 스칼라 인코딩과 완전히 등가), one-hot은 유효
        # 파라미터 하나(스칼라 가중치)를 비식별 파라미터 둘(w_male, w_female)로 표현해 초기화
        # 분산과 Adam 최적화 경로가 달라진다 — 이게 실제 성능 분산에 영향을 주는지 검증.
        self.sex_onehot = sex_onehot
        # 2026-08-04: age/sex를 아예 빼고 margin(또는 staging) 단독 신호만 보는 ablation
        # (train_light.py --no-age-sex, --clinical-margin과 함께 M5_R_ONLY) — age는 코호트 간
        # 방향이 안 맞고(TCGA HR>1, CPTAC HR<1) margin 신호를 오히려 희석시킬 수 있다는 가설 검증.
        self.use_age_sex = use_age_sex
        input_dim = (3 if sex_onehot else 2) if use_age_sex else 0
        if use_staging:
            if stage_stats is None:
                raise ValueError("use_staging=True면 stage_stats가 필요합니다 (stage_stats_from_csv 참조).")
            for field in STAGE_FIELDS:
                mean, std = stage_stats[field]
                short = _STAGE_BUFFER_NAMES[field]
                self.register_buffer(f"{short}_mean", torch.tensor(mean, dtype=torch.float32))
                self.register_buffer(f"{short}_std", torch.tensor(std, dtype=torch.float32))
            input_dim += 2 * len(STAGE_FIELDS)  # 필드당 (z_score, known_flag)

        self.use_margin = use_margin
        if use_margin:
            if margin_stats is None:
                raise ValueError("use_margin=True면 margin_stats가 필요합니다 (margin_stats_from_csv 참조).")
            mean, std = margin_stats
            self.register_buffer("margin_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("margin_std", torch.tensor(std, dtype=torch.float32))
            input_dim += 2  # (z_score, known_flag)

        # 입력 (age_z, sex_bin[, T/N/M/grade z_score+known 8차원][, margin z_score+known 2차원])
        # → 임베딩 (D,): 두 층 MLP
        # 2026-07-29: dropout(기본 0.0=기존 동작 보존) — 레퍼런스(tabular_survival.py::
        # ClinicalEmbedding)는 clinical 브랜치 안에 Dropout(0.25)이 있는데 우리는 없었다는
        # 차이를 검증하기 위한 ablation(train_light.py --clinical-dropout, --M7 전용).
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self, age_years: torch.Tensor, sex_idx: torch.Tensor,
        stage_ord: dict[str, torch.Tensor] | None = None,
        margin_ord: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            age_years  : (N,) float — 원본 나이(연 단위)
            sex_idx    : (N,) long  — encode_sex()로 만든 0(male)/1(female) 인덱스
            stage_ord  : use_staging=True일 때 필수. {field: (N,) long} — encode_stage_value()로
                         만든 순서형 정수, "미상"은 -1(data/dataset.py가 이 규약으로 저장한다).
            margin_ord : use_margin=True일 때 필수. (N,) long — encode_margin_value()로 만든
                         순서형 정수(R0=0/R1=1/R2=2), "미상"은 -1(stage_ord와 동일 규약).
        Returns:
            z_clinical: (N, D) — 임상 정보 임베딩
        """
        feats = []
        if self.use_age_sex:
            age_z = (age_years.float() - self.age_mean) / self.age_std
            if self.sex_onehot:
                feats += [age_z, (sex_idx == 0).float(), (sex_idx == 1).float()]
            else:
                feats += [age_z, sex_idx.float()]
        if self.use_staging:
            for field in STAGE_FIELDS:
                short = _STAGE_BUFFER_NAMES[field]
                ordv  = stage_ord[field].float()
                known = (ordv >= 0).float()
                mean  = getattr(self, f"{short}_mean")
                std   = getattr(self, f"{short}_std")
                z = torch.where(ordv >= 0, (ordv - mean) / std, torch.zeros_like(ordv))
                feats.append(z)
                feats.append(known)
        if self.use_margin:
            ordv  = margin_ord.float()
            known = (ordv >= 0).float()
            z = torch.where(ordv >= 0, (ordv - self.margin_mean) / self.margin_std, torch.zeros_like(ordv))
            feats.append(z)
            feats.append(known)
        x = torch.stack(feats, dim=-1)  # (N, 2) / (N, 10) / (N, 4) / (N, 12)
        return self.mlp(x)

"""
TCGA-PAAD / CPTAC-PDA clinical.tsv → OS(overall survival) 레이블 추출 스크립트.

GDC clinical.tsv에는 OS 컬럼이 따로 없고, 아래 필드들로 직접 구성해야 한다:
    - demographic.vital_status            : Alive / Dead / Not Reported
    - demographic.days_to_death           : Dead인 경우 event time
    - diagnoses.days_to_last_follow_up    : Alive인 경우 censoring time

    OS_time  = Dead → days_to_death, Alive → days_to_last_follow_up
    OS_event = Dead → 1, Alive → 0
    vital_status가 "Not Reported"이거나 위 time 필드가 전부 결측인 case는 OS를 알 수 없으므로 제외한다.

clinical.tsv는 case(환자) 하나당 diagnosis/treatment 등으로 row가 여러 개 중복되므로
case_id(=cases.submitter_id, preprocess_cptac.py의 case_id와 동일 포맷)별 첫 값만 사용한다.

사용법:
    python -m data.extract_os_labels                    # tcga + cptac 모두
    python -m data.extract_os_labels --dataset cptac     # 하나만
"""
import argparse
from pathlib import Path

import pandas as pd

CLINIC_ROOTS = {
    "tcga":  Path("data/raw/TCGA_clinic/clinical.tsv"),
    "cptac": Path("data/raw/CPTAC_clinic/clinical.tsv"),
}
OUT_PATHS = {
    "tcga":  Path("data/os_labels_tcga.csv"),
    "cptac": Path("data/os_labels_cptac.csv"),
}
NA_VALUES = ["'--"]


def extract_os_labels(dataset: str) -> pd.DataFrame:
    clinical = pd.read_csv(CLINIC_ROOTS[dataset], sep="\t", na_values=NA_VALUES)
    cols = [
        "cases.submitter_id",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.days_to_last_follow_up",
    ]
    cases = clinical[cols].groupby("cases.submitter_id", as_index=False).first()

    is_dead  = cases["demographic.vital_status"] == "Dead"
    is_alive = cases["demographic.vital_status"] == "Alive"

    os_time = pd.Series(float("nan"), index=cases.index, dtype="float64")
    os_time[is_dead]  = cases.loc[is_dead,  "demographic.days_to_death"]
    os_time[is_alive] = cases.loc[is_alive, "diagnoses.days_to_last_follow_up"]

    labels = pd.DataFrame({
        "case_id":      cases["cases.submitter_id"],
        "dataset":      dataset,
        "vital_status": cases["demographic.vital_status"],
        "OS_time":      os_time,
        "OS_event":     is_dead.astype(int),
    })

    n_total = len(labels)
    labels = labels.dropna(subset=["OS_time"]).reset_index(drop=True)
    n_dropped = n_total - len(labels)
    if n_dropped:
        print(f"[{dataset}] OS_time 결측(vital_status 미상 등)으로 {n_dropped}/{n_total} case 제외")

    return labels


def main():
    parser = argparse.ArgumentParser(description="TCGA-PAAD / CPTAC-PDA clinical.tsv → OS 레이블 CSV")
    parser.add_argument("--dataset", type=str, default="both", choices=["tcga", "cptac", "both"])
    args = parser.parse_args()

    datasets = ["tcga", "cptac"] if args.dataset == "both" else [args.dataset]
    for dataset in datasets:
        labels = extract_os_labels(dataset)
        out_path = OUT_PATHS[dataset]
        labels.to_csv(out_path, index=False)
        n_dead = int(labels["OS_event"].sum())
        print(f"[{dataset}] {len(labels)} case → {out_path} (event=Dead {n_dead}, censored=Alive {len(labels) - n_dead})")


if __name__ == "__main__":
    main()

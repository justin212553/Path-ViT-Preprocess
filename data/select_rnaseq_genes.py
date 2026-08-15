"""
RNA-seq 유전자 재선정 — 생존 예측에 직접 최적화된 기준으로, leakage 없이 INT1500(상위 1500개
유전자)을 만든다. 매개변수 없이 실행하면 전체 파이프라인이 끝까지 돈다:

  1. 문헌 기반 curated seed gene(PDAC driver/subtype/EMT/stromal/immune/proliferation/
     hypoxia/DNA damage repair 8개 카테고리, PDAC_LITERATURE_GENE_SETS)
  2. TCGA/CPTAC 각각의 train split(data/dataset.py::WSISurvivalDataset로 가져와 실제 학습에
     쓰이는 split과 일치시킴 — val/test 라벨은 선정에 전혀 쓰지 않는다)만으로 독립적으로
     gene별 univariate Cox score test 수행(반대 코호트는 로드조차 하지 않는다)
  3. 두 코호트의 순위(각자 자기 라벨로만 계산된)가 겹치는 유전자만 채택(INT1500) — z-score를
     합치는 대신 순위 집합의 교집합만 보므로 어느 방향(TCGA->CPTAC/CPTAC->TCGA)에 써도
     leakage가 없다(build_intersection_ranking 참조)
  4. 최종 순위 안에서 문헌 curated gene을 먼저 배치하고, 남는 자리는 나머지 유전자의 Cox
     순위로 채운다
  5. 상위 1500개를 data/rna_gene_selection_intersection/selected_genes_top_1500.csv로 저장

Cox score test는 유전자 18,879개를 하나씩 fitting하면 느리므로, 벡터화된 score test(효율적
점수 U, Fisher 정보 I, z = U/sqrt(I))로 전체 유전자를 한 번에 계산한다 — 결과는 표준 Cox
partial likelihood의 score test와 동일하다.

사용법:
    python -m data.select_rnaseq_genes
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from config import DataConfig
from data.dataset import COHORT_PATHS, WSISurvivalDataset

COMMON_GENES_PATH = Path("data/common_genes.csv")
N_GENES = 1500  # INT1500 — 이 프로젝트가 실제로 쓰는 유일한 유전자 수

# 레퍼런스 scripts/select_rnaseq_gene_features.py::PDAC_LITERATURE_GENE_SETS 그대로 이식.
# PDAC driver/subtype/EMT/stromal/immune/proliferation/hypoxia/DNA damage repair 8개
# (Collisson 2011, Moffitt 2015, Bailey 2016, Waddell 2015, TCGA 2017, Puleo 2018).
PDAC_LITERATURE_GENE_SETS = {
    "core_driver_tumor_suppressor": [
        "KRAS", "TP53", "CDKN2A", "CDKN2B", "SMAD4", "ARID1A", "KDM6A", "RNF43",
        "GNAS", "TGFBR2", "STK11", "SMARCA4", "PIK3CA", "PTEN", "BRAF", "MYC",
    ],
    "dna_damage_repair_therapy": [
        "BRCA1", "BRCA2", "PALB2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51",
        "MLH1", "MSH2", "MSH6", "PMS2", "ERCC1",
    ],
    "classical_pancreatic_progenitor": [
        "GATA6", "HNF1A", "HNF4A", "HNF4G", "FOXA2", "FOXA3", "PDX1", "MNX1",
        "ONECUT1", "ONECUT2", "KRT19", "EPCAM", "CDH1", "MUC1", "MUC5AC",
        "CEACAM5", "CEACAM6", "CLDN4", "CLDN18", "TFF1", "TFF2", "AGR2",
    ],
    "basal_squamous_mesenchymal": [
        "KRT5", "KRT6A", "KRT6B", "KRT14", "KRT17", "KRT81", "TP63", "KLF5",
        "S100A2", "S100A4", "SERPINB3", "SERPINB4", "VIM", "CDH2", "ZEB1",
        "ZEB2", "SNAI1", "SNAI2", "TWIST1", "ITGA6", "LAMC2",
    ],
    "stroma_ecm_invasion": [
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "COL5A2", "COL6A1", "COL6A2",
        "COL6A3", "FN1", "SPARC", "POSTN", "THBS1", "ACTA2", "TAGLN", "FAP",
        "ITGA2", "ITGA3", "ITGB1", "ITGB4", "MMP2", "MMP7", "MMP9", "MMP11",
        "MMP14", "PLAU", "PLAUR", "LOX", "LUM", "DCN", "BGN", "MET",
    ],
    "immune_inflammation_tgf_beta": [
        "CD274", "PDCD1", "CTLA4", "CD8A", "CD8B", "CD3D", "CD3E", "FOXP3",
        "CD68", "CD163", "LYZ", "CXCL12", "CXCR4", "CXCL8", "IL6", "IL6R",
        "STAT3", "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "CCL2", "CCR2", "CSF1", "CSF1R",
    ],
    "proliferation_cell_cycle_apoptosis": [
        "MKI67", "TOP2A", "CCNB1", "CCND1", "CCNE1", "CDK1", "CDK2", "BIRC5",
        "AURKA", "AURKB", "PLK1", "MCM2", "MCM4", "MCM6", "PCNA", "BCL2", "BAX", "CASP3",
    ],
    "hypoxia_metabolism_acinar_program": [
        "HIF1A", "VEGFA", "CA9", "SLC2A1", "LDHA", "HK2", "ENO1", "ALDOA",
        "PNLIP", "CPA1", "CPA2", "CPB1", "CTRB1", "CTRB2", "CLPS", "PRSS1", "REG1A", "REG1B",
    ],
}


def cox_score_test_matrix(
    x: np.ndarray, time: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    전체 유전자(열)에 대해 벡터화된 Cox partial-likelihood score test.

    Args:
        x:     (N, G) — 환자 x 유전자 z-score 행렬
        time:  (N,)   — OS_time
        event: (N,)   — OS_event (1=사망)
    Returns:
        z, chi2_stat, p_value: 각 (G,) — 유전자별 score test 통계량
    """
    order = np.argsort(-time)
    x = x[order].astype(np.float64, copy=False)
    event = event[order].astype(bool, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # risk set(현재 시점에 아직 생존/미관측인 환자 집합)의 누적 평균/분산 — time 내림차순
    # 정렬 후 앞에서부터 누적하면 매 시점의 risk set이 지금까지의 전체가 된다.
    risk_count = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    risk_mean = np.cumsum(x, axis=0) / risk_count[:, None]
    risk_var = np.cumsum(x * x, axis=0) / risk_count[:, None] - risk_mean**2
    risk_var = np.clip(risk_var, 1e-12, None)

    event_x, event_mean, event_var = x[event], risk_mean[event], risk_var[event]
    u = (event_x - event_mean).sum(axis=0)       # 효율적 점수(efficient score)
    info = event_var.sum(axis=0)                  # Fisher 정보
    z = u / np.sqrt(np.clip(info, 1e-12, None))
    chi2_stat = z * z
    p_value = chi2.sf(chi2_stat, df=1)
    return z, chi2_stat, p_value


def _train_case_ids_single(cfg: DataConfig, cohort: str) -> list[str]:
    """
    --dataset {cohort} --external이 실제로 쓰는 train split(그 코호트 단독 6:2:2)과
    동일한 case 집합을 반환한다 — 반대 코호트는 이 함수 호출 전체에서 아예 참조되지 않는다.
    """
    ds = WSISurvivalDataset(cfg, dataset=cohort, split="train", with_rna=False)
    train_cases = set(ds.items["case_id"].drop_duplicates())
    rna_cases = set(pd.read_csv(COHORT_PATHS[cohort].rna)["case_id"])
    return sorted(train_cases & rna_cases)


def rank_genes_by_train_cox(cfg: DataConfig, single_cohort: str) -> pd.DataFrame:
    """
    single_cohort의 train split만 사용해(반대 코호트 파일은 로드조차 하지 않음) 유전자를
    Cox score test로 순위 매긴다.
    """
    rna = pd.read_csv(COHORT_PATHS[single_cohort].rna).set_index("case_id")
    os_labels = pd.read_csv(COHORT_PATHS[single_cohort].os_labels).set_index("case_id")
    train_cases = _train_case_ids_single(cfg, single_cohort)

    common_genes = sorted(rna.columns)
    cases = [c for c in train_cases if c in rna.index and c in os_labels.index]
    x = rna.loc[cases, common_genes].to_numpy(dtype=np.float64)
    time = os_labels.loc[cases, "OS_time"].to_numpy(dtype=np.float64)
    event = os_labels.loc[cases, "OS_event"].to_numpy(dtype=np.int64)
    z, chi2_stat, p_value = cox_score_test_matrix(x, time, event)
    print(f"  {single_cohort}: train n={len(cases)}, events={int(event.sum())}")

    rows = pd.DataFrame({"gene_id": common_genes})
    rows[f"{single_cohort}_train_n"] = len(cases)
    rows[f"{single_cohort}_train_events"] = int(event.sum())
    rows[f"{single_cohort}_cox_z"] = z
    rows[f"{single_cohort}_cox_p"] = p_value
    rows["cox_z"] = z
    rows["cox_p"] = p_value
    rows["direction"] = np.where(z >= 0, "higher_expr_higher_risk", "higher_expr_lower_risk")
    rows["_abs_z"] = np.abs(z)

    rows = rows.sort_values(["cox_p", "_abs_z"], ascending=[True, False]).drop(columns="_abs_z").reset_index(drop=True)
    rows.insert(0, "rank", np.arange(1, len(rows) + 1))
    return rows


def build_literature_table(ranked: pd.DataFrame) -> pd.DataFrame:
    """
    PDAC_LITERATURE_GENE_SETS의 각 유전자를 gene_id로 매핑하고, ranked에서의 Cox 순위를 붙인다.
    """
    common_genes = pd.read_csv(COMMON_GENES_PATH).drop_duplicates(subset="gene_name", keep="first")
    name_to_id = common_genes.set_index("gene_name")["gene_id"]
    rank_lookup = ranked.set_index("gene_id").to_dict(orient="index")

    records, seen = [], set()
    for category, symbols in PDAC_LITERATURE_GENE_SETS.items():
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            gene_id = name_to_id.get(symbol)
            info = rank_lookup.get(gene_id, {}) if gene_id is not None else {}
            records.append({
                "gene_symbol": symbol, "category": category, "gene_id": gene_id,
                "available": gene_id is not None and gene_id in rank_lookup,
                "cox_rank": info.get("rank"), "cox_z": info.get("cox_z"),
            })
    table = pd.DataFrame(records)
    return table.sort_values(["available", "cox_rank", "gene_symbol"], ascending=[False, True, True], na_position="last").reset_index(drop=True)


def build_intersection_ranking(target_n: int) -> pd.DataFrame:
    """
    TCGA-only/CPTAC-only 순위(각자의 라벨만으로 독립적으로 계산된 gene_cox_ranking.csv)를
    읽어, 두 순위의 상위 유전자 집합이 겹치는 부분만 남긴다 — 어느 방향(TCGA->CPTAC/
    CPTAC->TCGA)에 써도 leakage가 없고(Stouffer처럼 z-score를 합치지 않고 순위 집합의
    교집합만 본다), 한쪽 코호트에서 우연히 튄 유전자가 걸러진다.

    target_n개를 채우기 위해 필요한 최소 깊이(각 코호트 상위 몇 개씩 볼지)를 자동으로 찾는다.
    교집합 안에서는 두 코호트 p-value의 합(작을수록 상위)으로 다시 정렬해 상위 target_n개만
    최종 채택한다.
    """
    tcga_path = Path("data/rna_gene_selection_tcgaonly/gene_cox_ranking.csv")
    cptac_path = Path("data/rna_gene_selection_cptaconly/gene_cox_ranking.csv")
    lit_path = Path("data/rna_gene_selection_tcgaonly/literature_curated_genes.csv")
    tcga = pd.read_csv(tcga_path).sort_values("tcga_cox_p").reset_index(drop=True)
    cptac = pd.read_csv(cptac_path).sort_values("cptac_cox_p").reset_index(drop=True)

    # 문헌 curated 유전자는 두 코호트 순위가 우연히 겹치는지와 무관하게 항상 먼저 포함시킨다.
    curated_ids: list[str] = []
    if lit_path.exists():
        lit = pd.read_csv(lit_path)
        curated_ids = lit.loc[lit["available"], "gene_id"].dropna().drop_duplicates().tolist()
    curated_set = set(curated_ids)
    n_fill = max(target_n - len(curated_ids), 0)

    depth = max(target_n, len(curated_ids))
    while True:
        depth = min(depth * 2, len(tcga))
        inter = (set(tcga.head(depth)["gene_id"]) & set(cptac.head(depth)["gene_id"])) - curated_set
        if len(inter) >= n_fill or depth >= len(tcga):
            break

    combined = (
        tcga[["gene_id", "tcga_cox_p"]].merge(cptac[["gene_id", "cptac_cox_p"]], on="gene_id")
    )
    combined = combined[combined["gene_id"].isin(inter)].copy()
    combined["combined_p_sum"] = combined["tcga_cox_p"] + combined["cptac_cox_p"]
    combined = combined.sort_values("combined_p_sum").reset_index(drop=True)
    fill_ids = combined.head(n_fill)["gene_id"].tolist()

    ordered_ids = curated_ids + fill_ids
    lookup = combined.set_index("gene_id")[["tcga_cox_p", "cptac_cox_p"]].to_dict(orient="index")
    # curated 유전자는 교집합 계산(combined)에 아예 안 걸렸을 수도 있어(자기 코호트 상위권이
    # 아니었던 경우) tcga/cptac 원본에서 개별 p-value를 따로 채운다.
    tcga_p = tcga.set_index("gene_id")["tcga_cox_p"].to_dict()
    cptac_p = cptac.set_index("gene_id")["cptac_cox_p"].to_dict()
    rows = []
    for gid in ordered_ids:
        info = lookup.get(gid, {})
        rows.append({
            "gene_id": gid,
            "tcga_cox_p": info.get("tcga_cox_p", tcga_p.get(gid)),
            "cptac_cox_p": info.get("cptac_cox_p", cptac_p.get(gid)),
            "is_literature_curated": gid in curated_set,
        })
    selected = pd.DataFrame(rows)
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected.attrs["depth_used"] = depth
    return selected


def main():
    cfg = DataConfig()

    for cohort in ("tcga", "cptac"):
        opposite = "cptac" if cohort == "tcga" else "tcga"
        print(f"Ranking genes by {cohort}-only train-split Cox score test "
              f"({opposite} not loaded at all)...")
        ranked = rank_genes_by_train_cox(cfg, single_cohort=cohort)
        out_dir = Path(f"data/rna_gene_selection_{cohort}only")
        out_dir.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(out_dir / "gene_cox_ranking.csv", index=False)
        print(f"  -> {out_dir / 'gene_cox_ranking.csv'} ({len(ranked)} genes)")

        literature_table = build_literature_table(ranked)
        literature_table.to_csv(out_dir / "literature_curated_genes.csv", index=False)
        n_available = int(literature_table["available"].sum())
        print(f"  -> literature curated genes: {len(literature_table)} total, {n_available} found in common RNA-seq")

    print(f"TCGA-only/CPTAC-only 순위 교집합 계산 중 (목표 {N_GENES}개)...")
    selected = build_intersection_ranking(N_GENES)
    out_dir = Path("data/rna_gene_selection_intersection")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"selected_genes_top_{N_GENES}.csv"
    selected[["rank", "gene_id", "tcga_cox_p", "cptac_cox_p", "is_literature_curated"]].to_csv(out_path, index=False)
    n_curated_in_sel = int(selected["is_literature_curated"].sum())
    print(f"  -> {out_path} ({len(selected)}개, 문헌 curated {n_curated_in_sel}개 + "
          f"교집합 {len(selected) - n_curated_in_sel}개, 깊이 top-"
          f"{selected.attrs.get('depth_used', '?')} each에서 교집합 확보)")


if __name__ == "__main__":
    main()

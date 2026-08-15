# PMA 데이터 파이프라인

전처리 스크립트를 raw 데이터부터 학습 입력까지 만드는 순서대로 정리한 문서.
모든 명령은 프로젝트 루트(`config.py`가 있는 위치)에서 실행한다.

## 실행 방법
`preprocess_pipeline.ipynb`를 순차적으로 실행한다.

## 전체 구조

```
raw WSI(.svs)              raw clinical.tsv           raw RNA tsv
     │                           │                          │
     ▼                           ▼                          │
[A] wsi_preprocess.py      [B] extract_os_labels.py         │
     │                           │                          │
     │                           ▼                          │
     │                     os_labels_{tcga,cptac}.csv       │
     │                           │                          │
     │                           └──────────┬───────────────┘
     │                                      ▼
     │                            [C] extract_rna_clinical.py
     │                                      │
     │                    rna_{tcga,cptac}.csv, clinical_{tcga,cptac}.csv
     │                                      │
     └──────────────────┬───────────────────┘
                         ▼
              [D] select_rnaseq_genes.py
                         │
        rna_gene_selection_intersection/selected_genes_top_1500.csv (INT1500)
                         │
                         ▼
              data/dataset.py::WSISurvivalDataset  ← 학습(train.py)이 이걸 씀
```

## 단계별 명령

### [A] WSI 타일링 + feature 추출
다른 단계와 독립적이라 가장 먼저(또는 B·C와 병렬로) 실행 가능. 가장 오래 걸리는 단계.

```
python -m data.wsi_preprocess --dataset tcga
python -m data.wsi_preprocess --dataset cptac
```

- 입력: `data/tcga_paad_wsi/*.svs`, `data/cptac_pda_wsi/*.svs`
- 출력: `data/patches_{tcga,cptac}/tiles/<slide_id>/*.jpg` + `features.pt`(기본 backbone=resnet50) + `slide_index_task*.csv`
- `--tiles-only`를 주면 타일링만 하고 feature 추출은 생략(다른 backbone으로 따로 뽑고 싶을 때)

### [B] 생존 라벨(OS) 추출
[A]와 독립적.

```
python -m data.extract_os_labels
```

- 입력: `data/raw/{TCGA,CPTAC}_clinic/clinical.tsv`
- 출력: `data/os_labels_{tcga,cptac}.csv`

### [C] RNA + clinical 추출
**[B]가 먼저 끝나야 함** — OS 라벨과 inner join하기 때문.

```
python -m data.extract_rna_clinical
```

- 입력: `data/raw/{TCGA,CPTAC}_RNA/<file_uuid>/*.tsv`, `data/raw/{TCGA,CPTAC}_clinic/clinical.tsv`, `data/os_labels_{tcga,cptac}.csv`([B] 산출물)
- 출력: `data/rna_{tcga,cptac}.csv`, `data/clinical_{tcga,cptac}.csv`, `data/common_genes.csv`

### [D] INT1500 유전자 선정
**[A][B][C]가 전부 끝나야 함** — train split을 구하려면 patches([A])가, Cox 랭킹을 매기려면 RNA/OS 라벨([B][C])이 필요.

```
python -m data.select_rnaseq_genes
```

매개변수 없음. 내부적으로 다음을 순서대로 전부 수행한다:
1. TCGA train split만으로 Cox 순위 계산 → `data/rna_gene_selection_tcgaonly/`
2. CPTAC train split만으로 Cox 순위 계산 → `data/rna_gene_selection_cptaconly/`
3. 두 순위의 교집합(상위 1500개, INT1500) → `data/rna_gene_selection_intersection/selected_genes_top_1500.csv`

여기까지 끝나면 `data/dataset.py::WSISurvivalDataset`이 바로 사용 가능한 상태가 된다.

## 다른 backbone feature

[A]는 기본 backbone(resnet50)으로만 feature를 뽑는다. UNI/UNI2-h 등 다른 backbone이 필요하면
[A] 완료 후 별도로:

```
python -m utils.extract_features --dataset tcga --backbone uni2
python -m utils.extract_features --dataset cptac --backbone uni2
```

from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class ModelConfig:
    # 표준 관례(embed_dim=256/num_heads=4)보다 작게 잡는다 — 이 프로젝트 코호트(TCGA-PAAD/
    # CPTAC-PDA)는 WSI-MIL 레퍼런스(CLAM/TransMIL) 대비 표본이 훨씬 적어 과적합 위험이 크다.
    embed_dim:              int   = 64
    num_heads:              int   = 2
    num_transformer_layers: int   = 1
    dropout:                float = 0.3    # Transformer/ViT 표준 기본값
    num_landmarks:          int   = 128   # Nystrom attention landmark 수 (근사 정밀도/속도 트레이드오프)
    # 대형 WSI(N 수만 패치) backward 메모리 절감용. 끄면 메모리↑ 속도↑
    grad_checkpoint:        bool  = False
    # 실시간 augmentation(--tile-augment) 경로에서 타일 디코딩+증강에 쓰는 스레드 수
    # (models/vit_m1.py::_patch_tokens, train.py --tile-decode-workers). forward() 안에서
    # 도는 CPU 작업이라 DataConfig.num_workers(DataLoader용)와는 별개다.
    tile_decode_workers:    int   = 4
    use_nystrom:             bool  = True   # False면 Nystrom 근사 대신 일반 self-attention(nn.MultiheadAttention) 사용
    use_spatial_embed:       bool  = True   # 좌표 임베딩(spatial embedding) 사용 여부
    # 절대좌표 임베딩을 상대offset 기반 attention bias(Swin류, models/vit_encoder.py::
    # RelativeBiasFullAttention)로 교체하는 실험용 플래그(train.py --rel-bias-attention).
    # True면 use_nystrom/use_spatial_embed는 자동으로 False로 강제된다(ViT_M1.__init__).
    use_rel_bias_attn:       bool  = False
    # use_rel_bias_attn(dense, O(N^2))을 kNN 그래프로 제한한 희소 버전(models/vit_encoder.py::
    # KNNBiasAttention, train.py --knn-bias-attention). use_rel_bias_attn과 배타적
    # (둘 다 True면 use_rel_bias_attn 우선).
    use_knn_bias_attn:       bool  = False
    knn_attn_k:               int   = 8
    knn_attn_edge_dropout:    float = 0.2
    # Nystrom(전역)을 대체하는 대신, 같은 레이어에서 kNN(국소)과 병렬로 더하는 hybrid
    # (models/vit_encoder.py::HybridLocalGlobalAttention, train.py --hybrid-attention).
    # use_rel_bias_attn/use_knn_bias_attn과 달리 use_nystrom/use_spatial_embed를 강제로 끄지
    # 않는다 — global 경로(Nystrom)가 계속 절대좌표 임베딩을 활용해야 하므로.
    use_hybrid_attn:          bool  = False
    # 학습 파라미터 없는 spatial feature(models/spatial_features.py)를 risk_head의 5번째
    # 관점으로 추가하는 옵션(train.py --spatial-autocorr/--attn-dispersion, ViT_PMA 전용).
    use_spatial_autocorr:     bool  = False
    use_attn_dispersion:      bool  = False
    # KNNBiasAttention의 학습되는 RelativePositionBias(MLP)를 고정(학습 파라미터 없는)
    # 거리감쇠 커널 bias=-dist/tau로 교체(train.py --knn-fixed-bias-attention).
    use_knn_fixed_bias_attn:  bool  = False
    knn_bias_tau:              float = 50.0
    # PSA-MIL(WACV 2026, arXiv:2503.16284) "learnable distance-decayed prior"의 경량 버전 —
    # tau를 고정 상수 대신 head별로 학습되는 스칼라로 둔다(train.py --learnable-tau,
    # --knn-fixed-bias-attention과 함께 사용).
    knn_bias_learnable_tau:   bool  = False


@dataclass
class DataConfig:
    wsi_root_tcga: str          = "data/tcga_paad_wsi"
    wsi_root_cptac: str         = "data/cptac_pda_wsi"
    patches_root_tcga: str      = "data/patches_tcga"
    patches_root_cptac: str     = "data/patches_cptac"
    num_workers: int            = 0
    precomputed: bool           = True
    seed: int                   = 42  # case 단위 train/val/test stratified split 재현성 (data/dataset.py 참조)


@dataclass
class TrainConfig:
    epochs:                int   = 30
    lr:                    float = 1e-5
    weight_decay:          float = 1e-1
    device:                str   = "cuda"
    seed:                  int   = 42
    # gradient accumulation 단위 = 환자 1명(보유한 모든 노드 누적 후 1 step, train.py 참조)
    warmup_epochs:         int   = 3       # linear LR warmup → cosine decay (epochs의 10%, 표준 warmup 비율)
    cnn_chunk_size:        int   = 64       # 대형 WSI(수천 패치)에서 CNN OOM 방지용 서브배치
    # Cox PH loss는 위험집합(risk set) 비교를 위해 여러 환자를 한 배치로 묶어야 한다.
    # 값이 클수록 risk set 추정이 안정적이지만, 환자별 forward activation을 배치가 찰 때까지
    # 모두 메모리에 들고 있어야 하므로 GPU 메모리 사용량도 함께 늘어난다.
    cox_batch_size:        int   = 16


@dataclass
class LightTrainConfig:
    """
    WSI 없이 Clinical/RNA만 쓰는 모델(M5/M6/M6X/M7, train_light.py) 학습 설정.

    TrainConfig보다 lr을 높게 잡는 이유: 여기 모델들은 ViT self-attention이 포함된 WSI
    스택이 아니라 작은 MLP(clinical/RNA 인코더 + risk_head)뿐이라, WSI 스택 학습 안정성을
    위해 낮춘 TrainConfig.lr 수준으로 낮출 필요가 없다.

    embed_dim/dropout(모델 폭)은 여기 두지 않고 ModelConfig(cfg.model)를 그대로 쓴다 —
    train_light.py와 train.py의 결과 차이가 "아키텍처"가 아니라 "이 학습 설정(lr/schedule)"
    때문임을 분리해 비교할 수 있게 한다.
    """
    epochs:                int   = 30
    lr:                    float = 1e-3
    weight_decay:          float = 1e-2
    device:                str   = "cuda"
    seed:                  int   = 42
    warmup_epochs:         int   = 3
    cox_batch_size:        int   = 16


@dataclass
class Config:
    model: ModelConfig      = field(default_factory=ModelConfig)
    data:  DataConfig       = field(default_factory=DataConfig)
    train: TrainConfig      = field(default_factory=TrainConfig)
    light: LightTrainConfig = field(default_factory=LightTrainConfig)

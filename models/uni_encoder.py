"""
UNI Feature Extractor
- 대규모 병리 이미지로 사전학습된 범용(general-purpose) 조직학적 특징 포착
- Mahmood Lab UNI(ViT-L/16) pretrained backbone으로 feature 추출 후 embed_dim으로 projection

가중치는 HuggingFace Hub에서 자동 다운로드 (최초 1회, gated repo 접근 승인 필요).
    모델: MahmoodLab/UNI
    캐시: ~/.cache/huggingface/hub/
"""
import torch
import torch.nn as nn
import timm

BACKBONE_DIM = 1024
UNI_HF_ID    = "hf_hub:MahmoodLab/UNI"


def _build_backbone(pretrained: bool) -> nn.Module:
    return timm.create_model(
        UNI_HF_ID,
        pretrained=pretrained,
        num_classes=0,
        init_values=1e-5,       # LayerScale 파라미터 포함 (체크포인트 요구사항)
        dynamic_img_size=True,  # 224 외 해상도 입력도 허용(positional embedding 보간)
    )


class UNIEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True, with_backbone: bool = True):
        """
        Args:
            with_backbone: False면 backbone을 생성하지 않는다
        """
        super().__init__()
        self.backbone = _build_backbone(pretrained) if with_backbone else None
        self.proj = nn.Sequential(
            nn.Linear(BACKBONE_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_patches, 3, 224, 224)
        Returns:
            features: (N_patches, embed_dim)
        """
        if self.backbone is None:
            raise RuntimeError("backbone이 없는 UNIEncoder(with_backbone=False)입니다 — "
                               "forward_pooled()으로 사전 추출된 feature를 전달하세요.")
        pooled = self.backbone(x)  # (N_patches, 1024) — UNI는 ViT라 이미 pooled 출력
        return self.proj(pooled)

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (N_patches, 1024) - backbone까지 미리 계산해 캐싱해둔 feature
        Returns:
            features: (N_patches, embed_dim)
        """
        return self.proj(pooled)

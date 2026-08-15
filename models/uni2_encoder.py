"""
UNI2-h Feature Extractor
- 대규모(200M+ 타일, H&E+IHC 혼합) 병리 이미지로 사전학습된 범용 조직학적 특징 포착 (UNI의 상위 호환 버전)
- Mahmood Lab UNI2-h(ViT-H/14) pretrained backbone으로 feature 추출 후 embed_dim으로 projection

가중치는 HuggingFace Hub의 gated repo에서 다운로드 (최초 1회, 접근 승인 필요).
    모델: MahmoodLab/UNI2-h
    캐시: ~/.cache/huggingface/hub/
"""
import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked

BACKBONE_DIM = 1536
UNI2H_HF_ID  = "hf_hub:MahmoodLab/UNI2-h"

_TIMM_KWARGS = dict(
    img_size=224,
    patch_size=14,
    depth=24,
    num_heads=24,
    init_values=1e-5,
    embed_dim=BACKBONE_DIM,
    mlp_ratio=2.66667 * 2,
    num_classes=0,
    no_embed_class=True,
    mlp_layer=SwiGLUPacked,
    act_layer=torch.nn.SiLU,
    reg_tokens=8,
    dynamic_img_size=True,  # 224 외 해상도 입력도 허용(positional embedding 보간)
    dynamic_img_pad=True,   # patch_size=14로 나누어떨어지지 않는 입력 크기를 자동 패딩
)


def _build_backbone(pretrained: bool) -> nn.Module:
    return timm.create_model(UNI2H_HF_ID, pretrained=pretrained, **_TIMM_KWARGS)


class UNI2hEncoder(nn.Module):
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
            x: (N_patches, 3, H, W)
        Returns:
            features: (N_patches, embed_dim)
        """
        if self.backbone is None:
            raise RuntimeError("backbone이 없는 UNI2hEncoder(with_backbone=False)입니다 — "
                               "forward_pooled()으로 사전 추출된 feature를 전달하세요.")
        pooled = self.backbone(x)  # (N_patches, 1536) — ViT라 이미 pooled 출력(reg_tokens 제외)
        return self.proj(pooled)

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (N_patches, 1536) - backbone까지 미리 계산해 캐싱해둔 feature
        Returns:
            features: (N_patches, embed_dim)
        """
        return self.proj(pooled)

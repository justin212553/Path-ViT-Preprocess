"""
CNN Feature Extractor
- 패치 수준의 Local Morphological Features 및 세포핵 밀집도 포착
- Lunit SwAV pretrained ResNet50 backbone으로 feature map 추출 후 embed_dim으로 projection

가중치는 HuggingFace Hub에서 자동 다운로드 (최초 1회).
    모델: 1aurent/resnet50.lunit_swav
    캐시: ~/.cache/huggingface/hub/
"""
import torch
import torch.nn as nn
import timm

BACKBONE_DIM   = 2048
LUNIT_SWAV_ID  = "hf_hub:1aurent/resnet50.lunit_swav"


def _build_backbone(pretrained: bool) -> nn.Module:
    return timm.create_model(
        LUNIT_SWAV_ID,
        pretrained=pretrained,
        num_classes=0,
        global_pool="",   # (B, 2048, H/32, W/32) feature map 반환
    )


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True, with_backbone: bool = True):
        """
        Args:
            with_backbone: False면 backbone을 생성하지 않는다
        """
        super().__init__()

        self.backbone = _build_backbone(pretrained) if with_backbone else None

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
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
        feat_map = self.backbone(x)
        return self.proj(self.pool(feat_map))

    def forward_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (N_patches, 2048) - backbone+pool까지 미리 계산해 캐싱해둔 feature
        Returns:
            features: (N_patches, embed_dim)
        """
        return self.proj(pooled)

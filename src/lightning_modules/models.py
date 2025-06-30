import torch
from torch import nn
from typing import Literal, Optional, Type
from transformers import (
    ViTModel, ViTConfig,
    DeiTModel, DeiTConfig,
    Dinov2Model, Dinov2Config,
    Dinov2WithRegistersModel, Dinov2WithRegistersConfig,
    CLIPVisionModel, CLIPVisionConfig
)

def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.Dropout):
        module.p = dropout_p


class VisionTransformer(nn.Module):
    def __init__(
        self,
        backbone_cls: Type[nn.Module],
        config_cls: Type,
        model_name: str,
        head_dim: int,
        backbone_dropout: float = 0.0,
        freeze_backbone: bool = False,
        load_pretrained_backbone: bool = False,
        config_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        config_kwargs = config_kwargs or {}

        if load_pretrained_backbone:
            self.backbone = backbone_cls.from_pretrained(model_name, **config_kwargs)
        else:
            config = config_cls.from_pretrained(model_name, **config_kwargs)
            self.backbone = backbone_cls(config)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if backbone_dropout > 0.0:
            self.backbone.apply(lambda m: set_encoder_dropout_p(m, dropout_p=backbone_dropout))

        self.head = nn.Linear(self.backbone.config.hidden_size, head_dim)

    def forward(self, x: torch.Tensor, output_attentions: bool = False):
        out = self.backbone(x, output_attentions=output_attentions)
        x = self._get_embedding(out)
        x = self.head(x)
        if output_attentions and "attentions" in out:
            return x, out["attentions"]
        return x

    def _get_embedding(self, backbone_output):
        return backbone_output["pooler_output"]


# === Model-specific wrappers ===

class VanillaViT(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            backbone_cls=ViTModel,
            config_cls=ViTConfig,
            model_name="google/vit-base-patch16-224", #WinKawaks/vit-small-patch16-224 ; google/vit-base-patch16-224
            **kwargs
        )


class DeiT(VisionTransformer):
    def __init__(self, model_size: Literal["tiny", "small", "base"], **kwargs):
        assert model_size in ["tiny", "small", "base"]
        model_name = f"facebook/deit-{model_size}-distilled-patch16-224"
        super().__init__(
            backbone_cls=DeiTModel,
            config_cls=DeiTConfig,
            model_name=model_name,
            **kwargs
        )


class DINO(VisionTransformer):
    def __init__(self, model_size: Literal["small", "base"], patch_size: Literal[8, 16], **kwargs):
        assert model_size in ["small", "base"]
        assert patch_size in [8, 16]
        model_name = f"facebook/dino-vit{model_size[0]}{patch_size}"
        super().__init__(
            backbone_cls=ViTModel,
            config_cls=ViTConfig,
            model_name=model_name,
            **kwargs
        )


class DINOv2(VisionTransformer):
    def __init__(self, model_size: Literal["small", "base"], with_registers: bool = True, **kwargs):
        assert model_size in ["small", "base"]
        if with_registers:
            cls, cfg_cls = Dinov2WithRegistersModel, Dinov2WithRegistersConfig
            model_name = f"facebook/dinov2-with-registers-{model_size}"
        else:
            cls, cfg_cls = Dinov2Model, Dinov2Config
            model_name = f"facebook/dinov2-{model_size}"
        super().__init__(
            backbone_cls=cls,
            config_cls=cfg_cls,
            model_name=model_name,
            **kwargs
        )


class VisionCLIP(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            backbone_cls=CLIPVisionModel,
            config_cls=CLIPVisionConfig,
            model_name="openai/clip-vit-base-patch16",
            **kwargs
        )

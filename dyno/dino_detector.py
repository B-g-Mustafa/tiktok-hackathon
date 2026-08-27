"""
DINO Vision Foundation Model Detector for Robust AIGC Detection.
Supports DINOv2 (ViT-L/14, ViT-B/14) and DINOv3 architectures (< 2B parameter limit).
Includes multi-layer intermediate token harvesting and LoRA/Dual-Stream integration.
"""

from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MLPHead(nn.Module):
    """Multi-Layer Perceptron Classification Head with LayerNorm for FP16 numerical stability."""
    def __init__(
        self,
        in_features: int,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.2,
        activation: str = "gelu",
    ):
        super().__init__()
        act_fn = nn.GELU() if activation.lower() == "gelu" else nn.ReLU()
        
        layers: List[nn.Module] = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                act_fn,
                nn.Dropout(p=dropout),
            ])
            curr_dim = h_dim

        # Final projection to binary logit
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).squeeze(-1)
        return torch.nan_to_num(out, nan=0.0, posinf=15.0, neginf=-15.0)


class DINODetector(nn.Module):
    """
    Robust AIGC Detector using DINO Foundation Model.
    Architecture:
    Input (3, H, W) -> DINO ViT Backbone -> [Multi-Layer Intermediate Tokens] -> LayerNorm -> MLP Head -> Logit
    """
    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-with-registers-large",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        mlp_hidden_dims: List[int] = [512, 256],
        mlp_dropout: float = 0.2,
        use_cls_and_patch_pool: bool = True,
        intermediate_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_cls_and_patch_pool = use_cls_and_patch_pool
        self.freeze_backbone_flag = freeze_backbone
        self.intermediate_layers = intermediate_layers

        # 1. Load Backbone
        self.backbone, self.embed_dim = self._load_backbone(backbone_name, pretrained)

        # Feature dimension after pooling / multi-layer concatenation
        if intermediate_layers and len(intermediate_layers) > 0:
            feature_dim = self.embed_dim * len(intermediate_layers)
        elif use_cls_and_patch_pool:
            feature_dim = self.embed_dim * 2
        else:
            feature_dim = self.embed_dim

        self.norm = nn.LayerNorm(feature_dim)

        # 2. MLP Head
        self.head = MLPHead(
            in_features=feature_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
        )

        if freeze_backbone:
            self.freeze_backbone()

        # Modern non-reentrant gradient checkpointing for DDP stability
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except Exception:
                try:
                    self.backbone.gradient_checkpointing_enable()
                except Exception:
                    pass

    def _load_backbone(self, model_name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Load vision backbone via Transformers, Timm, or PyTorch Hub."""
        # Resolve common DINOv3 shortcuts
        if model_name in ["dinov3", "dinov3-large", "dinov3-l", "facebook/dinov3-large"]:
            model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        elif model_name in ["dinov3-base", "dinov3-b", "facebook/dinov3-base"]:
            model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"

        # Try Hugging Face Transformers (DINOv3 / DINOv2)
        if TRANSFORMERS_AVAILABLE:
            try:
                if pretrained:
                    model = AutoModel.from_pretrained(model_name)
                else:
                    cfg = AutoConfig.from_pretrained(model_name)
                    model = AutoModel.from_config(cfg)
                embed_dim = getattr(model.config, "hidden_size", 1024)
                return model, embed_dim
            except Exception:
                pass

        # Try timm
        if TIMM_AVAILABLE:
            try:
                timm_name = model_name.replace("facebook/", "").replace("-", "_")
                if "dinov3" in timm_name:
                    timm_name = "vit_large_patch16_dinov3" if "large" in timm_name or "vitl" in timm_name else "vit_base_patch16_dinov3"
                elif "registers" in timm_name or "reg" in timm_name:
                    timm_name = "vit_large_patch14_reg4_dinov2.lvd142m"
                elif "dinov2" in timm_name and "large" in timm_name:
                    timm_name = "vit_large_patch14_dinov2.lvd142m"
                elif "dinov2" in timm_name and "base" in timm_name:
                    timm_name = "vit_base_patch14_dinov2.lvd142m"
                    
                model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
                embed_dim = model.num_features
                return model, embed_dim
            except Exception:
                pass

        # Try torch.hub (DINOv3 / DINOv2)
        try:
            if "dinov3" in model_name:
                hub_name = "dinov3_vitl16" if "large" in model_name or "vitl" in model_name else "dinov3_vitb16"
                model = torch.hub.load("facebookresearch/dinov3", hub_name, pretrained=pretrained)
            elif "reg" in model_name or "register" in model_name:
                hub_name = "dinov2_vitl14_reg"
                model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained)
            elif "large" in model_name:
                hub_name = "dinov2_vitl14"
                model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained)
            else:
                hub_name = "dinov2_vitb14"
                model = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained)
            embed_dim = model.embed_dim
            return model, embed_dim
        except Exception:
            pass

        # Fallback to cached facebook/dinov2-large if specific variant is unavailable
        if model_name != "facebook/dinov2-large":
            try:
                if TRANSFORMERS_AVAILABLE:
                    model = AutoModel.from_pretrained("facebook/dinov2-large")
                    embed_dim = getattr(model.config, "hidden_size", 1024)
                    print(f"--> [SUCCESS] Loaded cached facebook/dinov2-large foundation backbone.")
                    return model, embed_dim
            except Exception:
                pass

        # Minimal fallback ViT encoder if offline
        print(f"Warning: Could not load {model_name} from online sources. Using local placeholder ViT encoder.")
        fallback = nn.Sequential(
            nn.Conv2d(3, 1024, kernel_size=16, stride=16),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return fallback, 1024

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.freeze_backbone_flag = True

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone_flag = False

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized pooled embedding from backbone."""
        patch_size = getattr(self.backbone, "patch_size", None)
        if patch_size is None and hasattr(self.backbone, "config"):
            patch_size = getattr(self.backbone.config, "patch_size", None)
        if patch_size is None:
            patch_size = 16 if ("16" in self.backbone_name or "v3" in self.backbone_name) else 14
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]

        h, w = x.shape[-2:]
        if h % patch_size != 0 or w % patch_size != 0:
            target_h = int(round(h / patch_size)) * patch_size
            target_w = int(round(w / patch_size)) * patch_size
            target_h = max(patch_size, target_h)
            target_w = max(patch_size, target_w)
            x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)

        # Multi-layer Intermediate Token Harvesting (Hugging Face)
        if self.intermediate_layers and hasattr(self.backbone, "config"):
            try:
                outputs = self.backbone(x, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (B, N, D)
                cls_tokens = []
                for layer_idx in self.intermediate_layers:
                    actual_idx = layer_idx if layer_idx < len(hidden_states) else -1
                    cls_tokens.append(hidden_states[actual_idx][:, 0])
                feat = torch.cat(cls_tokens, dim=-1)
                return self.norm(torch.nan_to_num(feat, nan=0.0))
            except Exception:
                pass

        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if isinstance(feats, dict):
                if "x_norm_clstoken" in feats and "x_norm_patchtokens" in feats:
                    cls_tok = feats["x_norm_clstoken"]
                    patch_tok = feats["x_norm_patchtokens"].mean(dim=1)
                    feat = torch.cat([cls_tok, patch_tok], dim=-1)
                else:
                    feat = feats.get("last_hidden_state", list(feats.values())[0])
                    if feat.dim() == 3:
                        cls_tok = feat[:, 0]
                        patch_tok = feat[:, 1:].mean(dim=1)
                        feat = torch.cat([cls_tok, patch_tok], dim=-1) if self.use_cls_and_patch_pool else cls_tok
            elif isinstance(feats, torch.Tensor):
                if feats.dim() == 3:
                    cls_tok = feats[:, 0]
                    patch_tok = feats[:, 1:].mean(dim=1)
                    feat = torch.cat([cls_tok, patch_tok], dim=-1) if self.use_cls_and_patch_pool else cls_tok
                else:
                    feat = feats
            return self.norm(feat)

        # Standard Hugging Face forward
        if hasattr(self.backbone, "config"):
            outputs = self.backbone(x)
            hidden_states = outputs.last_hidden_state  # (B, N, D)
            cls_token = hidden_states[:, 0]
            if self.use_cls_and_patch_pool and hidden_states.size(1) > 1:
                patch_tokens = hidden_states[:, 1:].mean(dim=1)
                feat = torch.cat([cls_token, patch_tokens], dim=-1)
            else:
                feat = cls_token
            return self.norm(feat)

        # Torch Hub forward features or fallback
        if hasattr(self.backbone, "get_intermediate_layers"):
            if self.intermediate_layers:
                out = self.backbone.get_intermediate_layers(x, n=self.intermediate_layers, return_class_token=True)
                cls_tokens = [tok for _, tok in out]
                feat = torch.cat(cls_tokens, dim=-1)
                return self.norm(feat)
            else:
                out = self.backbone.get_intermediate_layers(x, n=1, return_class_token=True)
                patch_tokens, cls_token = out[0]
                if self.use_cls_and_patch_pool:
                    feat = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)
                else:
                    feat = cls_token
                return self.norm(feat)

        # Fallback
        feat = self.backbone(x)
        if self.use_cls_and_patch_pool and feat.size(-1) != self.embed_dim * 2:
            feat = torch.cat([feat, feat], dim=-1)
        return self.norm(feat)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.freeze_backbone_flag:
            with torch.no_grad():
                features = self.extract_features(x)
        else:
            features = self.extract_features(x)

        logit = self.head(features)

        if return_features:
            return logit, features
        return logit

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logit = self.forward(x)
            return torch.sigmoid(logit)


def build_detector(config: Dict[str, Any]) -> nn.Module:
    """Factory function to build detector from config dictionary with Dual-Stream & LoRA support."""
    from .lora_dino import apply_lora_to_dino
    from .frequency_branch import DualStreamDetector

    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "facebook/dinov2-large")
    pretrained = model_cfg.get("pretrained", True)
    freeze_backbone = model_cfg.get("freeze_backbone", True)
    mlp_cfg = model_cfg.get("mlp_head", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [512, 256])
    dropout = mlp_cfg.get("dropout", 0.2)
    intermediate_layers = model_cfg.get("intermediate_layers", None)

    # 1. Base DINOv2 Spatial Detector
    dino_model = DINODetector(
        backbone_name=backbone,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        mlp_hidden_dims=hidden_dims,
        mlp_dropout=dropout,
        intermediate_layers=intermediate_layers,
    )

    # 2. Inject LoRA if enabled
    lora_cfg = model_cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        dino_model = apply_lora_to_dino(
            dino_model,
            r=lora_cfg.get("r", 16),
            alpha=lora_cfg.get("alpha", 32.0),
            dropout=lora_cfg.get("dropout", 0.05),
        )

    # 3. Wrap in DualStreamDetector if dual-stream is requested
    is_dual = model_cfg.get("type") == "dual_stream" or model_cfg.get("dual_stream", {}).get("enabled", False)
    if is_dual:
        freq_dim = model_cfg.get("dual_stream", {}).get("freq_dim", 256)
        model = DualStreamDetector(
            dino_detector=dino_model,
            freq_dim=freq_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        return model

    return dino_model

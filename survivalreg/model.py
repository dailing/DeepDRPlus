
# %%
from torch.nn import Module
import torch.nn as nn
from torchvision.models.resnet import resnet50, resnet18, resnet101
from torchvision.models import convnext_tiny
import torch
import torch.nn.functional as F


def _feature_map_forward_transform(output):
    output = F.adaptive_avg_pool2d(output, output_size=(1, 1))
    output = output.squeeze(-1).squeeze(-1)
    return output


def _backbone_resnet18(*args, **kwargs):
    model = resnet18(*args, **kwargs)
    model.fc = nn.Identity()
    return model, 512, [
        (model.layer1[0], _feature_map_forward_transform),
        (model.layer1[1], _feature_map_forward_transform),
        (model.layer2[0], _feature_map_forward_transform),
        (model.layer2[1], _feature_map_forward_transform),
        (model.layer3[0], _feature_map_forward_transform),
        (model.layer3[1], _feature_map_forward_transform),
        (model.layer4[0], _feature_map_forward_transform),
        (model.layer4[1], _feature_map_forward_transform),
    ]


def _backbone_resnet50(*args, **kwargs):
    model = resnet50(*args, **kwargs)
    model.fc = nn.Identity()
    return model, 2048, [
        (model.layer1[0], _feature_map_forward_transform),
        (model.layer1[2], _feature_map_forward_transform),
        (model.layer2[1], _feature_map_forward_transform),
        (model.layer2[3], _feature_map_forward_transform),
        (model.layer3[2], _feature_map_forward_transform),
        (model.layer3[5], _feature_map_forward_transform),
        (model.layer4[1], _feature_map_forward_transform),
        (model.layer4[2], _feature_map_forward_transform),
    ]


def _backbone_resnet101(*args, **kwargs):
    model = resnet101(*args, **kwargs)
    model.fc = nn.Identity()
    return model, 2048, None


def _backbone_convnext_tiny(*args, **kwargs):
    model = convnext_tiny(*args, **kwargs)
    model.classifier = nn.Identity()
    return model, 768, None


class ModelProgression(Module):
    def __init__(self, backbone='convnext_tiny', output_size=20):
        super().__init__()
        backbone, feat_size, intermediate_layers = globals(
        )[f'_backbone_{backbone}'](pretrained=True)
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.LayerNorm(feat_size, eps=1e-6, elementwise_affine=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(feat_size, output_size, bias=True),
        )
        if intermediate_layers is not None:
            for i, (model, transform) in enumerate(intermediate_layers):
                self._record_output(f'_feat_{i:03d}', model, transform)
        self.forward_feat = {}

    def forward(self, x):
        self.forward_feat.clear()
        feat = self.backbone(x)
        feat = feat.view(feat.shape[0], -1)
        out = self.fc(feat)
        return out

    def _record_output(self, name: str, module: torch.nn.Module, transform=_feature_map_forward_transform):
        def wrap(model, _, output):
            # print('wrap')
            if transform is not None:
                output = transform(output)
            if name in self.forward_feat:
                print("WARNING: overwrite existing output")
            self.forward_feat[name] = output.detach().cpu()
        module.register_forward_hook(wrap)


# %%
if __name__ == "__main__":
    m = ModelProgression(backbone='resnet50')
    output = m(torch.randn(1, 3, 448, 448))
    print(output.shape)

# %%

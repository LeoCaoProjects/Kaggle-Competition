"""The reference CNN and both MobileNetV2 training policies."""

from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class ReferenceCNN(nn.Module):
    """Three convolution/pooling blocks followed by an eight-logit head."""

    def __init__(self):
        super().__init__()
        # Padding preserves spatial size before each pool halves it.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # One spatial average per feature channel.
            nn.Flatten(),
            nn.Dropout(0.3),  # Active only in training mode.
            nn.Linear(128, 8),  # Raw logits: CrossEntropyLoss includes log-softmax.
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def require(condition, message):
    if not condition:
        raise ValueError(message)


class FrozenMobileNetV2(nn.Module):
    """Official pretrained features with an eight-logit, trainable classifier."""
    def __init__(self, weights=None):
        super().__init__()
        self.network = mobilenet_v2(weights=weights)
        self.network.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.network.last_channel, 8))
        for parameter in self.network.features.parameters():
            parameter.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen weights alone do not freeze BatchNorm running statistics.
        self.network.features.eval()
        return self

    def forward(self, images):
        return self.network(images)

class PartiallyFineTunedMobileNetV2(nn.Module):
    """Train final feature modules and classifier while keeping all BatchNorm frozen."""
    def __init__(self):
        super().__init__()
        self.network = mobilenet_v2(weights=None)
        self.network.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.network.last_channel, 8))
        names = list(dict(self.network.features.named_children()))
        require(names == [str(i) for i in range(19)], "Unexpected torchvision feature-module layout")
        self.selected_names = names[-3:]
        require(self.selected_names == ["16", "17", "18"]
                and [type(self.network.features[int(i)]).__name__ for i in self.selected_names]
                == ["InvertedResidual", "InvertedResidual", "Conv2dNormActivation"], "Final modules changed")
        self.requires_grad_(False)
        for name in self.selected_names:
            self.network.features[int(name)].requires_grad_(True)
        self.network.classifier.requires_grad_(True)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.requires_grad_(False)
        require(sum(p.numel() for p in self.parameters()) == 2234120
                and sum(p.numel() for p in self.parameters() if p.requires_grad) == 1205128,
                "Architecture/trainable parameter count changed")
        self.train()

    def train(self, mode=True):
        # Independent from E1.train(): selected features must be allowed to train.
        super().train(mode)
        for name, module in self.network.features.named_children():
            module.train(mode and name in self.selected_names)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images):
        return self.network(images)


def build_model(experiment, pretrained=False):
    """Saved checkpoints always use pretrained=False; only new E1 training downloads weights."""
    if experiment == "E0":
        return ReferenceCNN()
    if experiment == "E1":
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        return FrozenMobileNetV2(weights=weights)
    if experiment == "E2" and not pretrained:
        return PartiallyFineTunedMobileNetV2()
    raise ValueError("Choose E0, E1 or E2; E2 starts from the saved E1 checkpoint")

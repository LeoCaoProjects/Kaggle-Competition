"""The fixed, randomly initialised reference CNN for ENGE707 Assessment 3."""

from torch import nn


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

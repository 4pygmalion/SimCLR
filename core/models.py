import torch
import torchvision


class ResNetEncoder(torch.nn.Module):
    def __init__(self, out_features: int):
        super(ResNetEncoder, self).__init__()
        self.out_features = out_features
        self.backbone = torchvision.models.resnet18(
            weights=False, num_classes=out_features
        )
        self._attach_projection_head()

    def _attach_projection_head(self) -> None:
        hidden_dim = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.out_features),
        )

        return

    def forward(self, x):
        return self.backbone(x)

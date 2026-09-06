import torch
import torch.nn as nn


class OceanEmbedFast(nn.Module):
    """
    Small CNN for rapid subsurface temperature reconstruction.

    Input:
        [batch, 7, 9, 9]

    Seven channels:
        0 = SST
        1 = SSS
        2 = SSH/SLA
        3 = U surface current
        4 = V surface current
        5 = U surface wind
        6 = V surface wind

    Output:
        [batch, 15]

    Fifteen temperature depths:
        0, 5, 10, 20, 30, 50, 75, 100,
        125, 150, 200, 300, 500, 700, 1000 m
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 15)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = OceanEmbedFast()

    test_input = torch.randn(4, 7, 9, 9)
    output = model(test_input)

    print("Model test successful")
    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)
    print("Parameters  :", sum(p.numel() for p in model.parameters()))
import torch
import torch.nn as nn


architecture_config = [
    # Model Architecture
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    'M',
    [
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        4,
    ],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        2,
    ],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.leakyrelu(x)


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(
                    CNNBlock(
                        in_channels, out_channels=x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                )
                in_channels = x[1]
            elif type(x) == str:
                layers.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
            elif type(x) == list:
                conv1 = x[0]  # Tuple
                conv2 = x[1]  # Tuple
                num_repeats = x[2]  # Integer
                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3],
                        ),
                        CNNBlock(
                            conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_class):
        S, B, C = split_size, num_boxes, num_class
        return nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(1024 * S * S, 496),  # in original paper, there is 4096
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(496, S * S * (C + B * 5))
        )

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(x)


def main():
    model = YOLOv1(3, split_size=7, num_boxes=2, num_class=20)
    x = torch.randn(4, 3, 448, 448)
    print(f"input shape is {x.shape}")
    out = model(x)
    print(f"output shape is {out.shape}")


if __name__ == "__main__":
    main()








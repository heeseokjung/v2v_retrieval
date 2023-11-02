import torch
import torch.nn as nn


def main():
    k = 8
    d = 384

    x = torch.randn(16, d, k)

    n = 22
    conv_transpose = nn.ConvTranspose1d(in_channels=d, out_channels=d, kernel_size=n - k + 1)

    output = conv_transpose(x)

    print(f"output: {output.shape}")


if __name__ == "__main__":
    main()
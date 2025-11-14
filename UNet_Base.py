import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(AttentionBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(x) * ca
        return sa


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(ResidualConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5 if dropout else 0.0)

        # Si in_channels != out_channels, on applique une projection pour le résidu
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

    def forward(self, x):
        residual = x
        x = self.conv(x)

        # Adapter le résidu si nécessaire
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        x = x + residual
        return self.relu(self.dropout(x))



class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        self.encode = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels, dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(1, 32)
        self.enc2 = _EncoderBlock(32, 64)
        self.enc3 = _EncoderBlock(64, 128)
        self.enc4 = _EncoderBlock(128, 256, dropout=True)

        self.center = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.GroupNorm(8, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
        )

        self.dec4 = _DecoderBlock(512, 256, 128)
        self.dec3 = _DecoderBlock(256, 128, 64)
        self.dec2 = _DecoderBlock(128, 64, 32)
        self.dec1 = nn.Sequential(
            ResidualConvBlock(64, 32),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

        self.att1 = AttentionBlock(32)
        self.att2 = AttentionBlock(64)
        self.att3 = AttentionBlock(128)
        self.att4 = AttentionBlock(256)

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        center = self.center(enc4)

        dec4 = self.dec4(torch.cat([self.att4(enc4), F.interpolate(center, enc4.size()[2:], mode='bilinear')], dim=1))
        dec3 = self.dec3(torch.cat([self.att3(enc3), F.interpolate(dec4, enc3.size()[2:], mode='bilinear')], dim=1))
        dec2 = self.dec2(torch.cat([self.att2(enc2), F.interpolate(dec3, enc2.size()[2:], mode='bilinear')], dim=1))
        dec1 = self.dec1(torch.cat([self.att1(enc1), F.interpolate(dec2, enc1.size()[2:], mode='bilinear')], dim=1))

        return F.interpolate(dec1, x.size()[2:], mode='bilinear')

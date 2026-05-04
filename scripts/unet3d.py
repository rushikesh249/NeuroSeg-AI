import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=groups, num_channels=out_ch)

        self.dropout = nn.Dropout3d(p=dropout) if dropout and dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.skip_conv = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        else:
            self.skip_conv = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.gn2(out)
        skip = self.skip_conv(x) if self.skip_conv is not None else x
        return F.leaky_relu(out + skip, negative_slope=0.01)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=4, base_features=32, num_classes=4, dropout=0.2, deep_supervision: bool = False, groups: int = 8):
        super().__init__()
        # encoder - deeper 4-level residual UNet
        self.enc1 = ResidualBlock(in_channels, base_features, groups=groups, dropout=dropout)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock(base_features, base_features * 2, groups=groups, dropout=dropout)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualBlock(base_features * 2, base_features * 4, groups=groups, dropout=dropout)
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = ResidualBlock(base_features * 4, base_features * 8, groups=groups, dropout=dropout)
        self.pool4 = nn.MaxPool3d(2)
        self.bottleneck = ResidualBlock(base_features * 8, base_features * 16, groups=groups, dropout=dropout)

        self.deep_supervision = deep_supervision

        self.up4 = UpConv(base_features * 16, base_features * 8)
        self.dec4 = ResidualBlock(base_features * 16, base_features * 8, groups=groups)

        self.up3 = UpConv(base_features * 8, base_features * 4)
        self.dec3 = ResidualBlock(base_features * 8, base_features * 4, groups=groups)

        self.up2 = UpConv(base_features * 4, base_features * 2)
        self.dec2 = ResidualBlock(base_features * 4, base_features * 2, groups=groups)

        self.up1 = UpConv(base_features * 2, base_features)
        self.dec1 = ResidualBlock(base_features * 2, base_features, groups=groups)

        self.final_conv = nn.Conv3d(base_features, num_classes, kernel_size=1)
        self.dropout = nn.Dropout3d(p=dropout)

        # deep supervision heads (1x1 convs) if enabled
        if self.deep_supervision:
            self.ds4 = nn.Conv3d(base_features * 8, num_classes, kernel_size=1)
            self.ds3 = nn.Conv3d(base_features * 4, num_classes, kernel_size=1)
            self.ds2 = nn.Conv3d(base_features * 2, num_classes, kernel_size=1)

        self._init_weights()

    def forward(self, x):
        # x: (B, C=4, D, H, W)
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)

        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)

        enc3 = self.enc3(p2)
        p3 = self.pool3(enc3)

        enc4 = self.enc4(p3)
        p4 = self.pool4(enc4)

        bottleneck = self.bottleneck(p4)
        bottleneck = self.dropout(bottleneck)

        up4 = self.up4(bottleneck)
        cat4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec4(cat4)

        up3 = self.up3(dec4)
        cat3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.up2(dec3)
        cat2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(cat1)

        out_main = self.final_conv(dec1)

        if self.deep_supervision:
            # produce auxiliary outputs and upsample them to main prediction spatial size
            ds4 = self.ds4(dec4)
            ds3 = self.ds3(dec3)
            ds2 = self.ds2(dec2)

            # ensure shapes match main output
            ds4_up = F.interpolate(ds4, size=out_main.shape[2:], mode='trilinear', align_corners=False)
            ds3_up = F.interpolate(ds3, size=out_main.shape[2:], mode='trilinear', align_corners=False)
            ds2_up = F.interpolate(ds2, size=out_main.shape[2:], mode='trilinear', align_corners=False)

            # return main + deep supervision outputs for loss handling by trainer
            return out_main, [ds2_up, ds3_up, ds4_up]

        return out_main

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # quick smoke test
    m = UNet3D(in_channels=4, base_features=16, num_classes=4)
    x = torch.randn(1, 4, 64, 128, 128)
    out = m(x)
    print('out', out.shape)
    print('params', count_parameters(m))

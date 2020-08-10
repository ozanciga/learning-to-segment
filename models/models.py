import torch
from torch import nn


class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class domain_classifier(torch.nn.Module):
    def __init__(self, nf, n_classes):
        super(domain_classifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(n_classes, 8, 1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, n_classes, 1),
            Flatten()
        )

    def forward(self, x):

        x = torch.nn.functional.adaptive_max_pool2d(x, (1, 1))

        return torch.flatten(x, 1)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class C1(nn.Module):
    def __init__(self, num_class, fc_dim, segSize):
        super(C1, self).__init__()

        self.segSize = segSize
        self.cbr = nn.Sequential(
            *conv3x3_bn_relu(fc_dim, fc_dim // 4, 1),
            nn.UpsamplingBilinear2d(scale_factor=2)
            )
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, x):

        x = self.cbr(x)
        l = self.conv_last(x)  # low res output

        h = nn.functional.interpolate(
            l,
            size=self.segSize,
            mode='nearest'
        )

        return l, h


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.AvgPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.inc = inconv(3, 64)
        self.down1 = down(64, 128)  # 256
        self.down2 = down(128, 256)  # 128
        self.down3 = down(256, 512)  # 64
        self.down4 = down(512, 1024)  # 32

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5

class SimpleSegment(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegment, self).__init__()

        def conv_block(in_filters, out_filters):
            block = [
                torch.nn.Conv2d(in_filters, out_filters, 3, 4, 1),
                torch.nn.BatchNorm2d(out_filters),
                torch.nn.ReLU(True),
                # torch.nn.Dropout(0.05),
            ]
            return block

        self.model = nn.Sequential(
            *conv_block(3, 8),
            *conv_block(8, 8),
            *conv_block(8, 16),
            *conv_block(16, 16),
            nn.AvgPool2d((2, 2)),
        )
        self.conv_last = nn.Conv2d(16, 32, 3, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        x = self.model(x)
        x = self.conv_last(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)

        return x



class Classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Classifier, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // 4),
            nn.ReLU(True),
            nn.Linear(num_features // 4, num_classes)
        )

    def forward(self, x):

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


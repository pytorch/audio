import torch.nn as nn


class Swish(nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)


def conv3x3(in_planes, out_planes, stride=1):
    """conv3x3.
    :param in_planes: int, number of channels in the input sequence.
    :param out_planes: int,  number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def downsample_basic_block(inplanes, outplanes, stride):
    """downsample_basic_block.
    :param inplanes: int, number of channels in the input sequence.
    :param outplanes: int, number of channels produced by the convolution.
    :param stride: int, size of the convolving kernel.
    """
    return  nn.Sequential(
        nn.Conv1d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=stride,
            bias=False,
        ),
        nn.BatchNorm1d(outplanes),
    )


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        relu_type="relu",
    ):
        """__init__.
        :param inplanes: int, number of channels in the input sequence.
        :param planes: int,  number of channels produced by the convolution.
        :param stride: int, size of the convolving kernel.
        :param downsample: boolean, if True, the temporal resolution is downsampled.
        :param relu_type: str, type of activation function.
        """
        super(BasicBlock1D, self).__init__()

        assert relu_type in ["relu","prelu", "swish"]

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)

        # type of ReLU is an input option
        if relu_type == "relu":
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        elif relu_type == "swish":
            self.relu1 = Swish()
            self.relu2 = Swish()
        else:
            raise NotImplementedError
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T)
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet1D(nn.Module):

    def __init__(self,
        block,
        layers,
        relu_type="swish",
        a_upsample_ratio=1,
    ):
        """__init__.
        :param block: torch.nn.Module, class of blocks.
        :param layers: List, customised layers in each block.
        :param relu_type: str, type of activation function.
        :param a_upsample_ratio: int, The ratio related to the \
            temporal resolution of output features of the frontend. \
            a_upsample_ratio=1 produce features with a fps of 25.
        """
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.relu_type = relu_type
        self.downsample_block = downsample_basic_block
        self.a_upsample_ratio = a_upsample_ratio

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.inplanes,
            kernel_size=80,
            stride=4,
            padding=38,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.inplanes)

        if relu_type == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif relu_type == "prelu":
            self.relu = nn.PReLU(num_parameters=self.inplanes)
        elif relu_type == "swish":
            self.relu = Swish()

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(
            kernel_size=20//self.a_upsample_ratio,
            stride=20//self.a_upsample_ratio,
        )


    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer.
        :param block: torch.nn.Module, class of blocks.
        :param planes: int,  number of channels produced by the convolution.
        :param blocks: int, number of layers in a block.
        :param stride: int, size of the convolving kernel.
        """

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block(
                inplanes=self.inplanes,
                outplanes=planes*block.expansion,
                stride=stride,
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                relu_type=self.relu_type,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    relu_type=self.relu_type,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """forward.
        :param x: torch.Tensor, input tensor with input size (B, C, T)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x


class Conv1dResNet(nn.Module):
    """Conv1dResNet
    """

    def __init__(self, relu_type="swish", a_upsample_ratio=1):
        """__init__.
        :param relu_type: str, Activation function used in an audio front-end.
        :param a_upsample_ratio: int, The ratio related to the \
            temporal resolution of output features of the frontend. \
            a_upsample_ratio=1 produce features with a fps of 25.
        """
        
        super(Conv1dResNet, self).__init__()
        self.a_upsample_ratio=a_upsample_ratio
        self.trunk = ResNet1D(
            BasicBlock1D,
            [2, 2, 2, 2],
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio
        )

    def forward(self, xs_pad):
        """forward.
        :param xs_pad: torch.Tensor, batch of padded input sequences (B, Tmax, idim)
        """
        B, T, C = xs_pad.size()
        xs_pad = xs_pad[:,:T//640*640,:]
        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = self.trunk(xs_pad)
        # -- from B x C x T to B x T x C
        xs_pad = xs_pad.transpose(1, 2)
        return xs_pad


def audio_resnet():
    return Conv1dResNet()

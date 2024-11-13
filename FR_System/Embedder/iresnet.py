"""
This file was is originally from:
https://onedrive.live.com/?authkey=%21AFZjr283nwZHqbA&id=4A83B6B633B029CC%215577&cid=4A83B6B633B029CC

Originally, all options were:
__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']
"""
import copy
import torch
from torch import nn
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'iresnet34': 'https://sota.nizhib.ai/pytorch-insightface/iresnet34-5b0d0e90.pth',
    'iresnet50': 'https://sota.nizhib.ai/pytorch-insightface/iresnet50-7f187506.pth',
    'iresnet100': 'https://sota.nizhib.ai/pytorch-insightface/iresnet100-73e07ba7.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.
    :param in_planes: Required. Type: int. Layer input dimensions.
    :param out_planes: Required. Type: int. Layer output dimensions.
    :param stride: Optional. Type: int. Layer stride. If not given, it is equal to 1.
    :param groups: Optional. Type: int. Layer groups. If not given, it is equal to 1.
    :param dilation: Optional. Type: int. Layer dilation. If not given, it is equal to 1.
    :return: nn.Conv2d layer.
    """
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution
    :param in_planes: Required. Type: int. Layer input dimensions.
    :param out_planes: Required. Type: int. Layer output dimensions.
    :param stride: Optional. Type: int. Layer stride. If not given, it is equal to 1.
    :return: nn.Conv2d layer.
    """
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    """
    IBasicBlock class.
    Composed out of:
        - Batch normalization with epsilon=1e-05 on num neurons.
        - Convolution layer with kernel=(3,3), stride=(1,1), padding=(1,1) with num neurons.
        - Batch normalization with epsilon=1e-05 on num neurons.
        - PReLU on num neurons.
        - Convolution layer with kernel=(3,3), stride=(2,2), padding=(1,1) with num neurons.
        - Batch normalization with epsilon=1e-05 on num neurons.
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        """
        Constructor.
        :param inplanes: Required. Type: int. Block input neurons dimensions.
        :param planes: Required. Type: int. Block inner neurons dimensions.
        :param stride: Optional. Type: int. Block stride. If not given, it is equal to 1.
        :param downsample: Optional. Type: nn.Sequential. If not given, downsample is not used.
        :param groups: Optional. Type: int. Layer groups. If not given, it is equal to 1.
        :param base_width: Optional. Type: int. The base width. If not given, it is equal to 64.
        :param dilation: Optional. Type: int. Layer dilation. If not given, it is equal to 1.
        """
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=2e-05, momentum=0.9)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-05, momentum=0.9)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Applied the IBasicBlock on the given input.
        :param x: Required. Type: Tensor. The input sample.
        :return: Tensor. The output of the IBasicBlock.
        """
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet(nn.Module):
    """
    IResNet class.
    """
    fc_scale = 7 * 7
    def __init__(self,
                 block, layers, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None):
        """
        Constructor.
        :param block: Required. Type: Object. The block to use. For example, AFBackbone.IBasicBlock.
        :param layers: Required. Type: list of int. The sizes of layers to use. For example, [3, 13, 30, 3].
        :param num_features: Optional. Type: int. The output size. If not given, it is equal to 512.
        :param zero_init_residual: Optional. Type: boolean. Whether to use zero_init_residual. If not given,
                                   it is equal to False.
        :param groups: Optional. Type: int. Layer groups. If not given, it is equal to 1.
        :param width_per_group: Optional. Type: int. The base width. If not given, it is equal to 64.
        :param replace_stride_with_dilation: Optional. Type: list of boolean. Whether to replace stride with dilation.
                                              If not given, it is equal to None.
        """
        super(IResNet, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=2e-05, momentum=0.9)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=2e-05, momentum=0.9)
        self.dropout = nn.Dropout2d(p=0.4, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=2e-05, momentum=0.9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Used to make the blocks.
        :param block: Required. Type: Object. The block to use. For example, AFBackbone.IBasicBlock.
        :param planes: Required. Type: int. Block inner neurons dimensions.
        :param blocks: Required. Type: int. Number of blocks.
        :param stride: Optional. Type: int. Block stride. If not given, it is equal to 1.
        :param dilate: Optional. Type: boolean. Whether to use dilate. If not given, it is equal to False.
        :return:
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=2e-05, momentum=0.9),
            )
        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride,
                            downsample,
                            self.groups,
                            self.base_width,
                            previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes,
                                planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Applied the IResNet on the given input.
        :param x: Required. Type: Tensor. The input sample.
        :return: Tensor. The output of the IResNet.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        return x


def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    """
    Load an iresnet module.
    :param arch: Requiered. Type: str. The iresnet specific module to use. The options are ['iresnet34',
    'iresnet50', 'iresnet100'].
    :param block: Required. Type: Object. The block to use. For example, AFBackbone.IBasicBlock.
    :param layers: Required. Type: list of int. The sizes of layers to use. For example, [3, 13, 30, 3].
    :param pretrained: Required. Type: boolean. Whether to use a pretrained model.
    :param progress: Required. Type: boolean. Whether or not to display a progress bar to stderr.
    :param kwargs: Optional. Additional arguments.
    :return: IResNet object. iresnet model.
    """
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        if arch == "iresnet100":
            model.load_state_dict(copy.deepcopy(torch.load("/sise/home/royek/iresnet100.pth")))
        if arch == 'iresnet50':
            model.load_state_dict(copy.deepcopy(torch.load("/sise/home/guyelov/model_ir_se50.pth",map_location='cpu')))
    return model


def iresnet34(pretrained=False, progress=True, **kwargs):
    """
    Load the iresnet34 module.
    :param pretrained: Optional. Type: boolean. Whether to use a pretrained model. Default is False.
    :param progress: Optional. Type: boolean. Whether or not to display a progress bar to stderr. Default is True.
    :param kwargs: Optional. Additional arguments.
    :return: IResNet object. iresnet model.
    """
    print('using irestnet34')
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    """
    Load the iresnet50 module.
    :param pretrained: Optional. Type: boolean. Whether to use a pretrained model. Default is False.
    :param progress: Optional. Type: boolean. Whether or not to display a progress bar to stderr. Default is True.
    :param kwargs: Optional. Additional arguments.
    :return: IResNet object. iresnet model.
    """
    print('using irestnet50')
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained, progress,
                    **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    """
    Load the iresnet100 module.
    :param pretrained: Optional. Type: boolean. Whether to use a pretrained model. Default is False.
    :param progress: Optional. Type: boolean. Whether or not to display a progress bar to stderr. Default is True.
    :param kwargs: Optional. Additional arguments.
    :return: IResNet object. iresnet model.
    """
    print('using irestnet100')
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, progress,
                    **kwargs)
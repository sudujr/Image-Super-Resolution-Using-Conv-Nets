import torch
from torch import nn
import torchvision
import math


class SRResNet(nn.Module):
    """
    SRResNet model for Deep Super Resolution
    """

    def __init__(self):
        super(SRResNet, self).__init__()
        scaling_factor = int(4)
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=9,batch_norm=False, activation='PReLu')
        #16 residual blocks with skip connecttions
        self.residual_blocks = nn.Sequential(*[ResidualBlock(kernel_size=3, n_channels=64) for i in range(16)])
        self.convblock2 = ConvBlock(in_channels=64, out_channels=64,kernel_size=3,batch_norm=True, activation=None)
        # Upscaling is done by sub-pixel convolution, with each such block upscaling by a factor of 2
        subpixel_conv_blocks = int(math.log2(4))
        self.subpixel_conv_blocks = nn.Sequential(
              *[SubpixelConvLayer(kernel_size=3, n_channels=64, scaling_factor=2) for i
              in range(subpixel_conv_blocks)])
        self.convblock3 = ConvBlock(in_channels=64, out_channels=3, kernel_size=9,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_img):
        """
        :param lr_img: low-resolution input images, a tensor of size (N, 3, w, h)
        :return: super-resolution output images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        """
        output = self.convblock1(lr_img)
        residual = output
        output = self.residual_blocks(output)
        output = self.convblock2(output)
        output = output + residual
        output = self.subpixel_conv_blocks(output)
        sr_img = self.convblock3(output)

        return sr_img


class ResidualBlock(nn.Module):
    """
    A residual block with two convolutional blocks with a skip connection across them.
    """

    def __init__(self, kernel_size=3, n_channels=64):
        super(ResidualBlock, self).__init__()

        self.convblock1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                             batch_norm=True, activation='PReLu')
        self.convblock2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,
                                             batch_norm=True, activation=None)

    def forward(self, input):
        residual = input
        output = self.convblock1(input)
        output = self.convblock2(output)
        output = output + residual
        return output

class ConvBlock(nn.Module):
    """
    A convolutional block with a convolution layer, Batch Norm layer and Acivation Layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super(ConvBlock, self).__init__()

        activation = activation.lower()
        layers = []
        #single conv layer
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=kernel_size // 2))
        #batch norm layer
        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        #activation layer
        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'swish':
            layers.append(nn.SiLU())

        # Put together the convolutional block as a sequence of the layers in this container
        self.cblock = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation.

        """
        output = self.cblock(input)

        return output

class SubpixelConvLayer(nn.Module):
    """
    A subpixel convolutional block with convolutional, pixel-shuffle, and PReLU activation layers.
    """

    def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):

        super(SubpixelConvLayer, self).__init__()

        # A convolutional layer upsamples with scaling factor^2, followed by pixel shuffle and PReLU activation function
        self.convlayer = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2),
                              kernel_size=kernel_size, padding=kernel_size // 2)
        # Pixel shuffler as mentioned in the SRGAN paper for upsampling
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=scaling_factor)
        self.act = nn.PReLU()

    def forward(self, input):

        output = self.convlayer(input)  # (N, n_channels * scaling factor^2, w, h)
        output = self.pixelshuffle(output)  # (N, n_channels, w * scaling factor, h * scaling factor)
        output = self.act(output)  # (N, n_channels, w * scaling factor, h * scaling factor)

        return output

class Discriminator(nn.Module):
    """
    The discriminator in the SRGAN, as defined in the paper.
    """

    def __init__(self):

        super(Discriminator, self).__init__()

        in_channels = 3
        kernel_size = 3
        n_channels = 64
        n_blocks = 8

        #Odd number of ConvBlocs increase the number of channels but have the same image size
        #Even Number of ConvBlocs have the same channels but the images size is halved.
        blocks = []
        for i in range(n_blocks):
            out_channels = (n_channels if i is 0 else in_channels * 2) if i % 2 is 0 else in_channels
            blocks.append(
                ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=1 if i % 2 is 0 else 2, batch_norm=i is not 0, activation='LeakyReLu'))
            in_channels = out_channels
        self.conv_blocks = nn.Sequential(*blocks)


        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(out_channels * 6 * 6, 1024)
        self.act = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, imgs):

        batch_size = imgs.size(0)
        output = self.conv_blocks(imgs)
        output = self.adaptive_pool(output)
        output = self.fc1(output.view(batch_size, -1))
        output = self.act(output)
        output = self.fc2(output)

        return output




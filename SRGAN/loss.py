import torch
import torch.nn as nn
from torchvision.models import vgg19

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        height_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :height - 1, :]), 2).sum()
        width_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :width - 1]), 2).sum()
        out = 2 * (height_tv / count_h + width_tv / count_w) / batch_size
        return out

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model = model.eval()

        for param in model.parameters():
            param.requires_grad = False

        self.freezed_vgg = model
        self.mse_loss = nn.MSELoss()

    def forward(self, source, target):
        source_feature = self.freezed_vgg(source)
        target_feature = self.freezed_vgg(target)
        loss = self.mse_loss(source_feature, target_feature)
        return loss
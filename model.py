
import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, pretrained, requires_grad):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
            
        if requires_grad == True:
            for param in self.model.parameters():
                param.requires_grad = True
        elif requires_grad == False:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.l0 = nn.Linear(2048, 256)

    def forward(self, x):
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        l0 = self.l0(x)
        return l0

model = ResNet50(pretrained=True, requires_grad=False)

import torch
import torch.nn as nn
from torchvision import models, transforms

class HierarchyCNN(nn.Module):
    def __init__(self, use_pretrained=True):
        super(HierarchyCNN, self).__init__()
        # coarse
        self.coarse_backbone, self.coarse_avgpool, self.coarse_classifier = self._make_vgg_backbone('coarse', use_pretrained)
        # fine
        self.fine_backbone, self.fine_avgpool, self.fine_classifier = self._make_vgg_backbone('fine', use_pretrained)
        
        self.fine_11conv = nn.Conv2d(1024, 512, (1, 1))
        
    
    def _make_vgg_backbone(self, type_, use_pretrained):
        if type_ == 'coarse':
            vgg = models.vgg13_bn(pretrained = use_pretrained)
            classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 20)
            )
        elif type_ == 'fine':
            vgg = models.vgg16_bn(pretrained = use_pretrained)
            classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 100)
            )
        backbone = vgg._modules['features']
        avg_pool = vgg._modules['avgpool']
        classifier = classifier.apply(self._init_weight)
        
        return backbone, avg_pool, classifier

    def _init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        # coarse
        coarse_out = self.coarse_backbone(x)
        coarse_cat = coarse_out.clone().detach()
        coarse_out = self.coarse_avgpool(coarse_out)
        coarse_out = torch.flatten(coarse_out, 1)        
        coarse_out = self.coarse_classifier(coarse_out)
        
        # fine
        fine_out = self.fine_backbone(x)
        fine_out = torch.cat((fine_out, coarse_cat), dim=1)
        fine_out = self.fine_11conv(fine_out)
        fine_out = self.fine_avgpool(fine_out)
        fine_out = torch.flatten(fine_out, 1)
        fine_out = self.fine_classifier(fine_out)
        
        return coarse_out, fine_out
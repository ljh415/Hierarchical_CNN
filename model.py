import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from collections import OrderedDict

class HierarchyCNN(nn.Module):
    def __init__(self, backbone='vgg', use_pretrained=True):
        super(HierarchyCNN, self).__init__()
        self.backbone = backbone
        if self.backbone == 'vgg':
            # coarse
            self.coarse_backbone, self.coarse_avgpool, self.coarse_classifier = self._make_vgg_backbone('coarse', use_pretrained)
            # fine
            self.fine_backbone, self.fine_avgpool, self.fine_classifier = self._make_vgg_backbone('fine', use_pretrained)
            
            self.fine_11conv = nn.Conv2d(1024, 512, (1, 1))
        elif self.backbone == 'wide_resnet_50':
            # coarse
            self.coarse_backbone, self.coarse_avgpool, self.coarse_classifier = self._make_wrs_backbone('coarse', use_pretrained)
            # fine
            self.fine_backbone, self.fine_avgpool, self.fine_classifier = self._make_wrs_backbone('fine', use_pretrained)
            self.fine_11conv = nn.Conv2d(3072, 2048, (1, 1))
            # self.fine_11conv = nn.Conv2d(4096, 2048, (1, 1))  # exp2
        
        
    def _make_wrs_backbone(self, type_, use_pretrained):
        wrn50 = models.wide_resnet50_2(pretrained = use_pretrained)
        
        if type_ == 'coarse':
            wrn50_backbone = nn.Sequential(OrderedDict(list(wrn50._modules.items())[:-3]))
            classifier =  nn.Linear(in_features=1024, out_features=20)
            
            # wrn50_backbone = nn.Sequential(OrderedDict(list(wrn50._modules.items())[:-2]))  # exp2 
            # classifier =  nn.Linear(in_features=2048, out_features=20)  # exp2
            
        elif type_ == 'fine':
            wrn50_backbone = nn.Sequential(OrderedDict(list(wrn50._modules.items())[:-2]))
            classifier = nn.Linear(in_features=2048, out_features=100)
            
        if use_pretrained:
            wrn50_backbone.requires_grad = False
            
            # if type_ == 'coarse':
            #     for layer_name, layer_ in wrn50_backbone.named_parameters():  # exp1
            #         if 'layer3' in layer_name :
            #             layer_.requires_grad = True
            #         else :
            #             layer_.requires_grad = False
            # else :
            #     wrn50_backbone.requires_grad = False
            
            # if type_ == 'coarse':
            #     for module_name, modules in wrn50_backbone.named_children():  # exp2
            #         if module_name == 'layer4':
            #             modules.requires_grad = True
            #         else :
            #             modules.requires_grad = False
            # else :
            #     wrn50_backbone.requires_grad = False
                
        avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        classifier = classifier.apply(self._init_weight)
        
        return wrn50_backbone, avg_pool, classifier
    
    def _make_vgg_backbone(self, type_, use_pretrained):
        if type_ == 'coarse':
            vgg = models.vgg13_bn(pretrained = use_pretrained)
            num_classes = 20

        elif type_ == 'fine':
            vgg = models.vgg16_bn(pretrained = use_pretrained)
            num_classes = 100
            
        backbone = vgg.features
        if use_pretrained:
            backbone.requires_grad = False
        avg_pool = vgg.avgpool
        classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
        classifier = classifier.apply(self._init_weight)
        
        return backbone, avg_pool, classifier

    def _init_weight(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x, only_fine=False):
        
        if only_fine:
            fine_out = self.fine_backbone(x)
            fine_out = self.fine_avgpool(fine_out)
            fine_out = torch.flatten(fine_out, 1)
            fine_out = self.fine_classifier(fine_out)
            
            return fine_out
        
        else :
            # coarse
            coarse_out = self.coarse_backbone(x)
            coarse_cat = coarse_out.clone().detach()
            coarse_out = self.coarse_avgpool(coarse_out)
            coarse_out = torch.flatten(coarse_out, 1)
            coarse_out = self.coarse_classifier(coarse_out)
            
            # fine
            fine_out = self.fine_backbone(x)
            if self.backbone == 'wide_resnet_50':
                coarse_cat = F.max_pool2d(coarse_cat, (3,3), 2, 1)  # exp2
            fine_out = torch.cat((fine_out, coarse_cat), dim=1)
            fine_out = self.fine_11conv(fine_out)
            fine_out = self.fine_avgpool(fine_out)
            fine_out = torch.flatten(fine_out, 1)
            fine_out = self.fine_classifier(fine_out)
            
            return coarse_out, fine_out
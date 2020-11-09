# Import Modules
import os
import json
import numpy as np
from efficientnet_pytorch import EfficientNet

# Import PyTorch
import torch
import torchvision
import torch.nn.functional as F
from torch import nn

class conv_model(nn.Module):
    def __init__(self, efficientnet_not_use, efficient_model_number, letter_model_path):
        super(conv_model, self).__init__()
        
        with open(os.path.join(letter_model_path, 'hyperparameter.json'), 'r') as f:
            letter_model_info = json.load(f)
            efficientnet_not_use = letter_model_info['efficientnet_not_use']
            efficientnet_model_number = letter_model_info['efficientnet_model_number']
            efficientnet_model_name = f'efficientnet-b{efficientnet_model_number}'
            if not efficientnet_not_use:
                self.letter_model = EfficientNet.from_pretrained(efficientnet_model_name, 
                                                            num_classes=26)
                self.letter_model.load_state_dict(torch.load(os.path.join(letter_model_path, 
                                                                     'model.pt')))
            else:
                self.letter_model = models.resnext50_32x4d(pretrained=False, num_classes=26)
                self.letter_model.load_state_dict(torch.load(os.path.join(letter_model_path, 
                                                                     'model.pt')))
        
        #
        self.efficient_net = EfficientNet.from_name(f'efficientnet-b{efficient_model_number}')
        self.batch_norm_2d = nn.BatchNorm2d(256, eps=1e-3, momentum=1e-2) # 256 -> 2304
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.last_dropout = nn.Dropout(0.1)
        self.last_linear = nn.Linear(2304, 10)

        self.test_linear1 = nn.Linear(2304, 512)
        self.test_linear2 = nn.Linear(2304, 512)
        self.test_bilinear = nn.Bilinear(512, 512, 256)
        self.test_last_linear = nn.Linear(256, 10)
        
    def forward(self, input_img):
        feature_extracted_ = self.test_linear1(self.efficient_net.extract_features(input_img).view(-1, 11, 11,2304))
        letter_feature_extracted_ = self.test_linear2(self.letter_model.extract_features(input_img).view(-1, 11, 11,2304))
        # feature_extracted_ = self.efficient_net.extract_features(input_img)
        # letter_feature_extracted_ = self.letter_model.extract_features(input_img)
        # feature_ = feature_extracted_ - letter_feature_extracted_
        feature_ = self.test_bilinear(feature_extracted_, letter_feature_extracted_).view(-1, 256, 11, 11)
        output = self.last_dropout(self.adaptive_avgpool(self.batch_norm_2d(feature_)))
        output = output.view(-1, 256)
        model_output = swish(self.test_last_linear(output))
        return model_output

def swish(x): 
    return x * torch.sigmoid(x) 
def mish(x): 
    return x * torch.tanh(F.softplus(x))
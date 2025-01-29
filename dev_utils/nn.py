import torch
import torch.nn as nn
import torchvision.models as models

def resnet18_for_single_channel():
    model = models.resnet18(weights=None)

    # Modifying the input layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modifying the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def resnet50_for_single_channel():
    model = models.resnet50(weights=None)

    # Modifying the input layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modifying the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def resnet18_aux_batch_norm():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model
    
class AdvPropResNet(nn.Module):
    def __init__(self, base_model):
        super(AdvPropResNet, self).__init__()
        self.base_model = base_model
        self.adv_bn = nn.ModuleList()
        self.clean_bn = nn.ModuleList()

        self.adv_bn.append(nn.BatchNorm2d(64))
        self.clean_bn.append(nn.BatchNorm2d(64))

    def forward(self, x, is_adv=False):
        bn_counter = 0
        x = self.base_model.conv1(x)
        if is_adv:
            x = self.adv_bn[bn_counter](x)
        else:
            x = self.clean_bn[bn_counter](x)
        x = nn.ReLU(inplace=True)(x)
        bn_counter += 1

        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        layers = [self.base_model.layer1, self.base_model.layer2, self.base_model.layer3, self.base_model.layer4]
        for layer in layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv2d):
                    x = module(x)
                    if is_adv:
                        x = self.adv_bn[bn_counter](x)
                    else:
                        x = self.clean_bn[bn_counter](x)
                    x = nn.ReLU(inplace=True)(x)
                    bn_counter += 1
                elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.ReLU) or isinstance(module, nn.Identity):
                    continue
                else:
                    x = module(x)
                    
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x
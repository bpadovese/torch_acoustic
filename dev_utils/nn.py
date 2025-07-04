import torch
import torch.nn as nn
import torchvision.models as models
import copy

def resnet18_for_single_channel():
    model = models.resnet18(weights=None)

    # Modifying the input layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modifying the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def get_efficientnet_b0_model(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    
    # Modify first conv layer to accept single-channel input
    model.features[0][0] = nn.Conv2d(
        in_channels=1,  # for 1-channel spectrograms
        out_channels=32,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )
    
    # Modify final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

def get_resnet34_model(num_classes=2):
    model = models.resnet34(weights=None)
    
    # Modify first conv layer for single-channel input
    model.conv1 = nn.Conv2d(
        in_channels=1,  # single-channel spectrograms
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    
    # Modify final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
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
    """
    AdvPropResNet: A wrapper around a ResNet model that introduces 
    separate Batch Normalization (BN) layers for clean and adversarial inputs.

    This follows the Adversarial Propagation (AdvProp) approach, where:
      - Clean examples use the original (clean) BN layers.
      - Adversarial examples use separate auxiliary (adv) BN layers.
      - During inference (model.eval()), only clean BN layers are used.


    Args:
        base_model (nn.Module): A ResNet model to wrap.
    
    Methods:
        forward(x, is_adv=False):
            Runs a forward pass with the appropriate BN layers.
            Uses adversarial BN only if training and is_adv=True.
    """

    def __init__(self, base_model):
        """
        Initializes AdvPropResNet by:
        - Creating copies of BatchNorm layers for clean and adversarial examples.
        - Storing these layers in dictionaries for fast retrieval.
        """
        super(AdvPropResNet, self).__init__()
        self.base_model = copy.deepcopy(base_model)  # Preserve original model structure

        # Store mappings of replaced BatchNorm layers
        self.adv_bn_layers = nn.ModuleDict()  # Adversarial BN layers
        self.clean_bn_layers = nn.ModuleDict()  # Clean BN layers (original)

        # Replace BatchNorm2d layers and store references
        self._replace_bn_layers(self.base_model)

    def _replace_bn_layers(self, module, prefix=""):
        """
        Recursively finds all BatchNorm2d layers in the ResNet model and 
        stores separate clean and adversarial versions.

        Args:
            module (nn.Module): The module (or submodule) to process.
            prefix (str): A string representing the module's path in the model hierarchy.
        """
        for name, child in module.named_children():
            full_name = f"{prefix}_{name}" if prefix else name

            if isinstance(child, nn.BatchNorm2d):
                # Create separate adversarial and clean batch norms
                self.adv_bn_layers[full_name] = nn.BatchNorm2d(child.num_features)
                self.clean_bn_layers[full_name] = copy.deepcopy(child)  # Keep a copy of the original BN

            else:
                self._replace_bn_layers(child, full_name)  # Recurse into deeper layers

    def _apply_bn(self, module, prefix="", is_adv=False):
        """
        Recursively replaces BN layers in the ResNet model with either:
        - Adversarial BN (`adv_bn_layers`) if training and is_adv=True.
        - Clean BN (`clean_bn_layers`) otherwise.

        This function **does NOT modify the model permanently**—it only switches 
        batch normalization layers temporarily during the forward pass. The 
        previous states are stored in the adv_bn_layers ModuleDict and clean_bn_layers.
        
        Args:
            module (nn.Module): The module to modify.
            prefix (str): The hierarchical name of the module.
            is_adv (bool): Whether to use adversarial BN layers.
        """
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if full_name in self.adv_bn_layers:  # If it's a replaced BN layer
                bn_layer = self.adv_bn_layers[full_name] if is_adv else self.clean_bn_layers[full_name]
                setattr(module, name, bn_layer)  # Replace BN instance
            else:
                self._apply_bn(child, full_name, is_adv)  # Recurse

    def forward(self, x, is_adv=False):
        """
        Forward pass with dynamic BN swapping.

        - During training:
          - Uses adversarial BN layers if `is_adv=True`.
          - Uses clean BN layers otherwise.
        - During inference (`model.eval()`), always uses clean BN layers.

        Args:
            x (torch.Tensor): The input tensor (batch of images).
            is_adv (bool): Whether the input is adversarial (default: False).
        
        Returns:
            torch.Tensor: The model output (predicted logits or class scores).
        """
        self._apply_bn(self.base_model, is_adv=is_adv)  # Apply stored BN layers
        return self.base_model(x)  # Forward pass through unchanged structure
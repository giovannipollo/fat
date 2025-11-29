from .mobilenet import MobileNetV1
from .resnet_cifar import ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
from .resnet_imagenet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG11, VGG13, VGG16, VGG19

MODELS = {
    # MobileNet
    "mobilenetv1": MobileNetV1,
    # ResNet (CIFAR-specific)
    "resnet20": ResNet20,
    "resnet32": ResNet32,
    "resnet44": ResNet44,
    "resnet56": ResNet56,
    "resnet110": ResNet110,
    # ResNet (ImageNet-style, adapted for small images)
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    # VGG
    "vgg11": VGG11,
    "vgg13": VGG13,
    "vgg16": VGG16,
    "vgg19": VGG19,
}


def get_model(config: dict):
    """
    Get model based on configuration.
    
    Args:
        config: Configuration dictionary with model settings
        
    Returns:
        nn.Module: The initialized model
    """
    model_name = config["model"]["name"].lower()
    num_classes = config["model"].get("num_classes", 10)
    in_channels = config["model"].get("in_channels", 3)
    
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    return MODELS[model_name](num_classes=num_classes, in_channels=in_channels)

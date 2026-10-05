from __future__ import annotations

import torch
from PIL import Image
from torchvision import models, transforms

CLASSES = ["Anthracnose", "Downy_Mildew", "Healthy", "Mosaic_Virus"]
THAI_LABELS = {
    "Anthracnose": "โรคแอนแทรคโนส",
    "Downy_Mildew": "โรคราน้ำค้าง",
    "Healthy": "ใบปกติ",
    "Mosaic_Virus": "โรคไวรัสใบด่าง",
}
IMAGE_SIZE = 224

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_model(pretrained: bool = False, architecture: str = "mobilenet_v3_small") -> torch.nn.Module:
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(CLASSES))
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return model


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    return INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0)

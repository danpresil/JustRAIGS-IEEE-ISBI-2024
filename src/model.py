import os
from pathlib import Path
from typing import Dict, Tuple, Union

import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from helper import DEFAULT_GLAUCOMATOUS_FEATURES

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_path(*paths: str) -> str:
    """Return the first existing path from the given candidates."""
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(str(paths))


def _load_model(model_name: str, checkpoint: str) -> torch.nn.Module:
    """Load a timm model and apply sigmoid to the classifier head."""
    state_dict = torch.load(checkpoint, map_location=DEVICE)
    model = timm.create_model(model_name, pretrained=False, num_classes=11)
    if "efficient" in model_name:
        model.classifier = nn.Sequential(model.classifier, nn.Sigmoid())
    elif "convnext" in model_name:
        model.head.fc = nn.Sequential(model.head.fc, nn.Sigmoid())
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop out black borders from an RGB image."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
        return img
    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


def torch_flip_lr(images: torch.Tensor) -> torch.Tensor:
    return TF.hflip(images)


def torch_shift(images: torch.Tensor, shift: int, axis: int = 1) -> torch.Tensor:
    return torch.roll(images, shift, dims=[axis])


def torch_rotate(images: torch.Tensor, angle: float) -> torch.Tensor:
    return TF.rotate(images, angle)


_test_transform = albumentations.Compose([
    albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


MODEL_CONFIGS = [
    {
        "image_size": 1024,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9317073170731708.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9317073170731708.pt",
        ],
        "justification_thresholds": [0.4646, 0.305, 0.5114, 0.6, 0.5521, 0.6768, 0.5418, 0.6, 0.5862, 0.6539],
    },
    {
        "image_size": 896,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9472817133443163.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9472817133443163.pt",
        ],
        "justification_thresholds": [0.3681, 0.2146, 0.5, 0.6, 0.6495, 0.6203, 0.3608, 0.6, 0.5062, 0.6324],
    },
    {
        "image_size": 896,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9496183206106871.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9496183206106871.pt",
        ],
        "justification_thresholds": [0.4282, 0.4478, 0.7328, 0.6262, 0.5153, 0.6, 0.6165, 0.6, 0.6388, 0.7586],
    },
    {
        "image_size": 1024,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9346092503987241.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9346092503987241.pt",
        ],
        "justification_thresholds": [0.2539, 0.6276, 0.3802, 0.6, 0.5487, 0.6125, 0.7946, 0.1933, 0.6079, 0.5555],
    },
    {
        "image_size": 896,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9362363919129082.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9362363919129082.pt",
        ],
        "justification_thresholds": [0.1341, 0.5891, 0.5615, 0.6, 0.6, 0.6, 0.5272, 0.6, 0.6708, 0.6541],
    },
    {
        "image_size": 896,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_1_weight_best_0.9382113821138212.pt"),
            "/opt/algorithm/checkpoints/epoch_1_weight_best_0.9382113821138212.pt",
        ],
        "justification_thresholds": [0.508, 0.37, 0.614, 0.676, 0.697, 0.75, 0.754, 0.171, 0.531, 0.814],
    },
    {
        "image_size": 896,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9347079037800687.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9347079037800687.pt",
        ],
        "justification_thresholds": [0.508, 0.37, 0.614, 0.676, 0.697, 0.75, 0.754, 0.171, 0.531, 0.814],
    },
    {
        "image_size": 896,
        "model_name": "convnext_tiny",
        "weight_paths": [
            str(Path(__file__).resolve().parent / "weights/epoch_0_weight_best_0.9538714991762768.pt"),
            "/opt/algorithm/checkpoints/epoch_0_weight_best_0.9538714991762768.pt",
        ],
        "justification_thresholds": [0.508, 0.37, 0.614, 0.676, 0.697, 0.75, 0.754, 0.171, 0.531, 0.814],
    },
]


MODEL_INFOS = []
for cfg in MODEL_CONFIGS:
    weight_path = _resolve_path(*cfg["weight_paths"])
    model = _load_model(cfg["model_name"], weight_path)
    MODEL_INFOS.append(
        {
            "image_size": cfg["image_size"],
            "model": model,
            "justification_thresholds": cfg["justification_thresholds"],
        }
    )


def _prepare_tta(np_img: np.ndarray) -> torch.Tensor:
    """Apply transforms and test-time augmentations to an image."""
    tensor = _test_transform(image=np_img)["image"]
    imgs_valid_f = torch_flip_lr(tensor)
    imgs_valid_w0 = torch_shift(tensor, -3, axis=2)
    imgs_valid_w1 = torch_shift(tensor, 3, axis=2)
    imgs_valid_h0 = torch_shift(tensor, -3, axis=1)
    imgs_valid_h1 = torch_shift(tensor, 3, axis=1)
    imgs_valid_r0 = torch_rotate(tensor, -10)
    imgs_valid_r1 = torch_rotate(tensor, 10)
    return torch.stack([
        tensor,
        imgs_valid_f,
        imgs_valid_w0,
        imgs_valid_w1,
        imgs_valid_h0,
        imgs_valid_h1,
        imgs_valid_r0,
        imgs_valid_r1,
    ]).to(DEVICE)


def predict(image: Union[np.ndarray, Image.Image]) -> Tuple[bool, float, Dict[str, bool]]:
    """Run the ensemble on an image and return primitive Python outputs."""
    if isinstance(image, Image.Image):
        np_img = np.array(image)
    elif isinstance(image, np.ndarray):
        np_img = image
    else:
        raise TypeError("image must be a NumPy array or PIL.Image")

    np_img = crop_image_from_gray(np_img)
    np_img_896 = cv2.resize(np_img, (896, 896))
    np_img_1024 = cv2.resize(np_img, (1024, 1024))

    tta_896 = _prepare_tta(np_img_896)
    tta_1024 = _prepare_tta(np_img_1024)

    referable_glaucoma_proba_list = []
    justification_labels_list = []
    with torch.no_grad():
        for info in MODEL_INFOS:
            model = info["model"]
            if info["image_size"] == 896:
                model_output = model(tta_896)
            elif info["image_size"] == 1024:
                model_output = model(tta_1024)
            else:
                raise NotImplementedError
            model_output = model_output.mean(0).cpu().numpy()
            referable_glaucoma_proba_list.append(float(model_output[0]))
            justification_labels_list.append(
                model_output[1:] > np.array(info["justification_thresholds"])
            )

    ensemble_referable_glaucoma_proba = float(np.mean(referable_glaucoma_proba_list))
    ensemble_justification_vote = np.array(justification_labels_list).astype(int).sum(0)

    threshold = np.mean([info["justification_thresholds"][0] for info in MODEL_INFOS])
    is_referable_glaucoma = bool(ensemble_referable_glaucoma_proba > float(threshold))

    features = {
        k: bool(int(ensemble_justification_vote[idx] >= 3))
        for idx, (k, _) in enumerate(DEFAULT_GLAUCOMATOUS_FEATURES.items())
    }

    return is_referable_glaucoma, ensemble_referable_glaucoma_proba, features


__all__ = ["predict"]

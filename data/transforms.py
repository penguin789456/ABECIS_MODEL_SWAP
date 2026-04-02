"""
Augmentation pipelines using Albumentations.

Train pipeline:  geometric + photometric augmentations + normalise + ToTensor
Val / test:      normalise + ToTensor only
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — reasonable for concrete crack imagery
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_train_transforms(patch_size: int = 512) -> A.Compose:
    return A.Compose(
        [
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            # Expanded rotation range: cracks appear at many orientations
            # border_mode=0 (constant black) avoids spurious crack continuities
            A.Rotate(limit=45, border_mode=0, p=0.5),
            # Photometric
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            # CLAHE: equalise local contrast — improves visibility of low-contrast cracks
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            # Sharpen: enhance fine crack edges
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # Normalise then convert to CHW tensor
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    return A.Compose(
        [
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ]
    )


def get_test_transforms() -> A.Compose:
    """Identical to val; named separately for clarity in inference scripts."""
    return get_val_transforms()

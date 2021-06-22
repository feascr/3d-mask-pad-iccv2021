from albumentations.augmentations.transforms import RGBShift
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from . import register_preprocessor
from .augmentations import Resize, PermutePatches


@register_preprocessor("CompetitionPreprocessorV1")
class CombinedPreprocessorV1:
    def __init__(self, config):
        self._use_HSV = config.use_HSV

        self.resize = Resize(
            config.backbone.image_size, 
            config.backbone.image_size
        )

        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.5),
                A.RandomBrightness(p=0.5),
                A.RandomContrast(p=0.5),
            ], p=0.5),
            A.Rotate(limit=10, p=0.5),
        ])

        if self._use_HSV:
            if config.HSV_norm == 'same':
                HSV_mean = config.backbone.pretrained_mean
                HSV_std = config.backbone.pretrained_std
            else:
                HSV_mean = [0., 0., 0.]
                HSV_std = [1., 1., 1.]
            norm_mean = config.backbone.pretrained_mean + HSV_mean
            norm_std = config.backbone.pretrained_std + HSV_std
        else:
            norm_mean = config.backbone.pretrained_mean
            norm_std = config.backbone.pretrained_std

        self.final_transfroms = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])

    def train(self, image):
        image = self.train_transforms(image=image)['image']
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.resize(image)
        image = self.final_transfroms(image=image)['image']
        return image

    def inference(self, image):
        image = self.resize(image)
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.final_transfroms(image=image)['image']
        return image

@register_preprocessor("CompetitionPreprocessorV2")
class CombinedPreprocessorV2:
    def __init__(self, config):
        self._use_HSV = config.use_HSV

        self.resize = Resize(
            config.backbone.image_size, 
            config.backbone.image_size
        )

        self.train_transforms = A.Compose(
            [
                # FLip
                A.HorizontalFlip(p=0.5),
                # Blur
                A.OneOf([
                    A.Blur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.5),
                # Noise
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0),
                ], p=0.3),
                # Sepia
                A.ToSepia(p=0.1),
                # Color/Contrast/Brightness
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
                    A.Sequential([
                        A.RandomBrightness(limit=0.4, p=0.75),
                        A.RandomContrast(limit=0.4, p=0.75)
                    ], p=1.0)
                ], p=0.8),
                # Shift, scale, rotate
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1, 
                            rotate_limit=20, 
                            border_mode=cv2.BORDER_CONSTANT, 
                            p=0.85
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1, 
                            rotate_limit=90, 
                            border_mode=cv2.BORDER_CONSTANT, 
                            p=0.15
                        ),
                    ], p=0.5),
            ]
        )

        if self._use_HSV:
            if config.HSV_norm == 'same':
                HSV_mean = config.backbone.pretrained_mean
                HSV_std = config.backbone.pretrained_std
            else:
                HSV_mean = [0., 0., 0.]
                HSV_std = [1., 1., 1.]
            norm_mean = config.backbone.pretrained_mean + HSV_mean
            norm_std = config.backbone.pretrained_std + HSV_std
        else:
            norm_mean = config.backbone.pretrained_mean
            norm_std = config.backbone.pretrained_std

        self.final_transfroms = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])

    def train(self, image):
        image = self.train_transforms(image=image)['image']
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.resize(image)
        image = self.final_transfroms(image=image)['image']
        return image

    def inference(self, image):
        image = self.resize(image)
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.final_transfroms(image=image)['image']
        return image


@register_preprocessor("CompetitionPreprocessorV3")
class CombinedPreprocessorV3:
    def __init__(self, config):
        self._use_HSV = config.use_HSV

        self.resize = Resize(
            config.backbone.image_size, 
            config.backbone.image_size
        )

        self.train_transforms = A.Compose(
            [
                # FLip
                A.HorizontalFlip(p=0.5),
                # Blur
                A.OneOf([
                    A.Blur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.5),
                # Noise
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0),
                ], p=0.3),
                # Sepia
                A.ToSepia(p=0.1),
                # Color/Contrast/Brightness
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
                    A.Sequential([
                        A.RandomBrightness(limit=0.4, p=0.75),
                        A.RandomContrast(limit=0.4, p=0.75)
                    ], p=1.0)
                ], p=0.8),
                # Shift, scale, rotate
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1, 
                            rotate_limit=20, 
                            border_mode=cv2.BORDER_CONSTANT, 
                            p=0.85
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1, 
                            rotate_limit=90, 
                            border_mode=cv2.BORDER_CONSTANT, 
                            p=0.15
                        ),
                    ], p=0.5),
                
            ]
        )

        if self._use_HSV:
            if config.HSV_norm == 'same':
                HSV_mean = config.backbone.pretrained_mean
                HSV_std = config.backbone.pretrained_std
            else:
                HSV_mean = [0., 0., 0.]
                HSV_std = [1., 1., 1.]
            norm_mean = config.backbone.pretrained_mean + HSV_mean
            norm_std = config.backbone.pretrained_std + HSV_std
        else:
            norm_mean = config.backbone.pretrained_mean
            norm_std = config.backbone.pretrained_std

        self.additional_transforms = [
            PermutePatches(patches_per_y=3, patches_per_x=3, p=0.5)
        ]

        self.final_transfroms = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])

    def train(self, image):
        image = self.train_transforms(image=image)['image']
        for add_tr in self.additional_transforms:
            image = add_tr(image)
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.resize(image)
        image = self.final_transfroms(image=image)['image']
        return image

    def inference(self, image):
        image = self.resize(image)
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.final_transfroms(image=image)['image']
        return image


@register_preprocessor("CompetitionPreprocessorV4")
class CombinedPreprocessorV3:
    def __init__(self, config):
        self._use_HSV = config.use_HSV

        self.resize = Resize(
            config.backbone.image_size, 
            config.backbone.image_size
        )

        self.train_transforms = A.Compose(
            [
                # FLip
                A.HorizontalFlip(p=0.5),
                # Blur
                A.OneOf([
                    A.Blur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=(3, 5), p=1.0),
                    A.MedianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.5),
                # Noise
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 50), mean=0, p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=1.0),
                ], p=0.3),
                # Sepia
                A.ToSepia(p=0.1),
                # Color/Contrast/Brightness
                A.OneOf([
                    A.OneOf([
                        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
                        A.RGBShift(p=1.0)
                    ], p=1.0),
                    A.Sequential([
                        A.RandomBrightness(limit=0.4, p=0.75),
                        A.RandomContrast(limit=0.4, p=0.75)
                    ], p=1.0)
                ], p=0.8),
                # Shift, scale, rotate
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1, 
                            rotate_limit=20, 
                            border_mode=cv2.BORDER_CONSTANT, 
                            p=0.85
                        ),
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.1, 
                            rotate_limit=90, 
                            border_mode=cv2.BORDER_CONSTANT, 
                            p=0.15
                        ),
                    ], p=0.5),
                
            ]
        )

        if self._use_HSV:
            if config.HSV_norm == 'same':
                HSV_mean = config.backbone.pretrained_mean
                HSV_std = config.backbone.pretrained_std
            else:
                HSV_mean = [0., 0., 0.]
                HSV_std = [1., 1., 1.]
            norm_mean = config.backbone.pretrained_mean + HSV_mean
            norm_std = config.backbone.pretrained_std + HSV_std
        else:
            norm_mean = config.backbone.pretrained_mean
            norm_std = config.backbone.pretrained_std

        self.additional_transforms = [
            PermutePatches(patches_per_y=3, patches_per_x=3, p=0.5)
        ]

        self.final_transfroms = A.Compose([
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2()
        ])

    def train(self, image):
        image = self.train_transforms(image=image)['image']
        for add_tr in self.additional_transforms:
            image = add_tr(image)
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.resize(image)
        image = self.final_transfroms(image=image)['image']
        return image

    def inference(self, image):
        image = self.resize(image)
        if self._use_HSV:
            image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image = np.concatenate((image, image_HSV), axis=2)
        image = self.final_transfroms(image=image)['image']
        return image
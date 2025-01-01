import cv2
import numpy as np
import torch
from torchvision import transforms
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Resize(object):
    def __init__(self, size):
        # Expect size to be (height, width)
        self.size = size

    def __call__(self, image, mask):
        # Resize using the (width, height) order that OpenCV expects
        image = cv2.resize(image, (self.size[1], self.size[0]))  # Swap width and height for cv2.resize
        mask = cv2.resize(mask, (self.size[1], self.size[0]), interpolation=cv2.INTER_NEAREST)  # Swap width and height
        return image, mask

class RandomRotation(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            rotate_degree = random.uniform(-self.degree, self.degree)
            h, w = img.shape[:2]
            img_center = (w / 2, h / 2)

            rotation_mat = cv2.getRotationMatrix2D(img_center, rotate_degree, 1.0)
            img = cv2.warpAffine(img, rotation_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, rotation_mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

        return img, mask

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        return img, mask

class RandomBrightness(object):
    def __init__(self, delta=32, p=0.5):
        assert delta >= 0
        assert delta <= 255
        self.delta = delta
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.array(image, dtype=np.float32)
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        return image, mask

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, p=0.5):
        self.lower = lower
        self.upper = upper
        self.p = p
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, mask):
        if random.random() < self.p:
            image = np.array(image, dtype=np.float32)
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        return image, mask

class PhotometricDistort(object):
    def __init__(self, p=0.5):
        self.pd = [
            RandomContrast(),
            RandomBrightness(),
        ]
        self.rand_brightness = RandomBrightness()
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            im = image.copy().astype(np.float32)
            if random.random() < 0.5:
                for op in self.pd:
                    im, mask = op(im, mask)
            else:
                for op in self.pd[::-1]:
                    im, mask = op(im, mask)
            image = im.astype(np.uint8)
        return image, mask

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = image.astype(np.float32)
        image = (image / 255.0 - self.mean) / self.std
        return image, mask

class ToTensor(object):
    def __call__(self, img, mask):
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        return img, mask
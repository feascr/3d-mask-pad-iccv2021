import cv2
import random
import numpy as np


class Resize:
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation
    
    def __call__(self, img):
        img = cv2.resize(img, (self.width, self.height), interpolation=self.interpolation)
        return img

class PermutePatches:
    def __init__(self, patches_per_y=3, patches_per_x=3, p=0.5):
        self.patches_per_y = patches_per_y
        self.patches_per_x = patches_per_x
        self.p = 0.5
    
    def __call__(self, img):
        if random.random() < self.p:
            return img
        image_width = img.shape[1]
        image_height =  img.shape[0]
        image_channels = img.shape[2]
        
        tile_width = image_width // self.patches_per_x
        tile_height = image_height // self.patches_per_y
        
        tiles_pos = list(range(self.patches_per_y * self.patches_per_x))
        random.shuffle(tiles_pos)

        new_img = np.zeros_like(img, shape=(tile_height * self.patches_per_y, tile_width * self.patches_per_x, image_channels))
        for i in range(self.patches_per_y):
            for j in range(self.patches_per_x):
                tile = img[
                    (i * tile_height):((i + 1) * tile_height), 
                    (j * tile_width):((j + 1) * tile_width),
                    :
                ]
                
                tile_pos = tiles_pos.pop()
                new_img[
                    (tile_pos // 3 * tile_height):((tile_pos // 3 + 1) * tile_height),
                    (tile_pos % 3 * tile_width):((tile_pos % 3 + 1) * tile_width),
                    :
                ] = tile
        return new_img
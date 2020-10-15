import Utils
from FixedEncoder import FixedEncoder
import PIL
import numpy as np
import torch

fixedEncoder = FixedEncoder('ResNet18', 12)

filename1 = 'PMC493266_002_00.png'
image_path1 = Utils.create_abs_path('pubtabnet/train/' + filename1)
filename2 = 'PMC493271_005_00.png'
image_path2 = Utils.create_abs_path('pubtabnet/train/' + filename2)

image1 = PIL.Image.open(image_path1)
image2 = PIL.Image.open(image_path2)
images = [image1, image2]

input_images = [fixedEncoder.preprocess(image) for image in images]
input_images = torch.stack(input_images)
print(input_images.shape)

features_map = fixedEncoder.encode(input_images)
print(features_map.shape)

print(features_map.nbytes)
print(np.array(image1).nbytes)
print(np.array(image2).nbytes)


# %%
from training.dataset_tool import create_dataset
import os
import numpy as np
import PIL
# %%
image_path = '/home/kunato/dataset/fashion_video/crop'
data_dir = create_dataset(image_path, resolution=512)
training_images = []
for fname in os.listdir(data_dir):
    if fname.endswith('.jpg'):
        training_images.append(
            np.array(PIL.Image.open(os.path.join(data_dir, fname))))
imgs = np.reshape(training_images, [5, 20, *training_images[0].shape])
imgs = np.concatenate(imgs, axis=1)
imgs = np.concatenate(imgs, axis=1)
PIL.Image.fromarray(imgs).resize((1000, 250), PIL.Image.ANTIALIAS)

# %%

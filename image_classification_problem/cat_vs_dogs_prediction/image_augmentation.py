import os
import random

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

image_generator = ImageDataGenerator(rotation_range=30,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest'
                                     )

fix, ax = plt.subplots(2, 3, figsize=(20, 10))
all_images = []

# Train data path
BASE_DIR = os.getcwd()
train_folder_path = f'{BASE_DIR}/data/PetImages/Train/Dog/'
_, _, dog_images = next(os.walk(train_folder_path))
random_img = random.sample(dog_images, 1)[0]
random_img = plt.imread(f'{BASE_DIR}/data/PetImages/Train/Dog/' + random_img)
all_images.append(random_img)

random_img = random_img.reshape((1,) + random_img.shape)
sample_augmented_images = image_generator.flow(random_img)

for _ in range(5):
    augmented_imgs = sample_augmented_images.next()
    for img in augmented_imgs:
        all_images.append(img.astype('uint8'))

for idx, img in enumerate(all_images):
    ax[int(idx / 3), idx % 3].imshow(img)
    ax[int(idx / 3), idx % 3].axis('off')
    if idx == 0:
        ax[int(idx / 3), idx % 3].set_title('Original Image')
    else:
        ax[int(idx / 3), idx % 3].set_title(f'Augmented Image {idx}')

plt.show()

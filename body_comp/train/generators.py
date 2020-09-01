import numpy as np

from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator


class SegmentationSequence(Sequence):

    def __init__(self, images, masks, batch_size, jitter=False):
        self.masks = masks
        self.images = images
        self.batch_size = batch_size
        self.shuffled_indices = np.random.permutation(self.images.shape[0])
        self.jitter = jitter
        if self.jitter:
            self.jitter_datagen = ImageDataGenerator(rotation_range=5,
                                                     width_shift_range=0.05,
                                                     height_shift_range=0.05,
                                                     fill_mode="nearest")

    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __getitem__(self, idx):

        # The shuffled indices in this batch
        batch_inds = self.shuffled_indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.jitter:

            batch_images_list = []
            batch_masks_list = []

            for i in batch_inds:
                # Stack mask and image together to ensure that they are transformed
                # in exactly the same way
                stacked = np.dstack([self.images[i, :, :, :].astype(np.uint8), self.masks[i, :, :, :]])
                transformed = self.jitter_datagen.random_transform(stacked)

                batch_images_list.append(transformed[:, :, 0].astype(float))
                batch_masks_list.append(transformed[:, :, 1])

            batch_images = np.dstack(batch_images_list)
            batch_images = np.transpose(batch_images[:, :, :, np.newaxis], [2, 0, 1, 3])
            batch_masks = np.dstack(batch_masks_list)
            batch_masks = np.transpose(batch_masks[:, :, :, np.newaxis], [2, 0, 1, 3])

        else:

            # Slice images and labels for this batch
            batch_images = self.images[ batch_inds, :, :, :]
            batch_masks = self.masks[ batch_inds, :, :, :]

        return (batch_images, batch_masks)

    def on_epoch_end(self):
        # Shuffle the dataset indices again
        self.shuffled_indices = np.random.permutation(self.images.shape[0])

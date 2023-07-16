import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images_labels = [
        (i, os.path.join(path, image))
        for i, path in zip(labels, paths)
        for image in sampler(os.listdir(path))
    ]
    if shuffle:
        random.shuffle(images_labels)
    return images_labels


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.

    Description of IterableDataset: https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
    """

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        config={},
        device=torch.device("cpu"),
        cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_size: size of meta batch size (e.g. number of functions)
            batch_type: train/val/test
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train : num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val :]
        self.device = device
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)  # misc.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / 255.0
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Args:
            does not take any arguments
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and
                2. label batch has shape [K+1, N, N]
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.
            
            3. The value for `self.num_samples_per_class` will be set to K+1 
            since for K-shot classification you need to sample K supports and 
            1 query.
        """

        #############################
        #### YOUR CODE GOES HERE ####
        sampled_character_paths = random.sample(self.folders, self.num_classes)
        labels = np.eye(self.num_classes)

        # Format the support images and labels. We do not want to shuffle the support images.
        support_images_labels = get_images(sampled_character_paths, labels,
                                      nb_samples=self.num_samples_per_class-1, shuffle=False)
        support_label_batch = np.vstack([item[0] for item in support_images_labels]).reshape(self.num_classes,
                                                                             self.num_samples_per_class-1,
                                                                             self.num_classes)

        support_label_batch = torch.tensor(np.moveaxis(support_label_batch, [0, 1, 2], [1, 0, 2]))
        support_image_stack = np.vstack([self.image_file_to_array(filename=item[1], dim_input=self.dim_input)
                                 for item in support_images_labels]).reshape(self.num_classes,
                                                                     self.num_samples_per_class-1,
                                                                     self.dim_input)
        support_image_stack = torch.tensor(np.moveaxis(support_image_stack, [0, 1, 2], [1, 0, 2]))

        # Format the query images and labels. We want to shuffle the support images.
        query_images_labels = get_images(sampled_character_paths, labels,
                                           nb_samples=1, shuffle=True)

        query_label_batch = np.vstack([item[0] for item in query_images_labels]).reshape(self.num_classes, 1,
                                                                                         self.num_classes)
        query_label_batch = torch.tensor(np.moveaxis(query_label_batch, [0, 1, 2], [1, 0, 2]))

        query_image_stack = np.vstack([self.image_file_to_array(filename=item[1], dim_input=self.dim_input)
                                         for item in query_images_labels]).reshape(self.num_classes, 1,
                                                                                     self.dim_input)
        query_image_stack = torch.tensor(np.moveaxis(query_image_stack, [0, 1, 2], [1, 0, 2]))

        # concat the support and query labels and images
        image_stack = torch.cat([support_image_stack, query_image_stack], dim=0)
        label_batch = torch.cat([support_label_batch, query_label_batch], dim=0)

        return (image_stack, label_batch)
        #############################

    def __iter__(self):
        while True:
            yield self._sample()

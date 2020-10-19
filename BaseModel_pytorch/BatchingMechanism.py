from Storage import Storage
import torch
import numpy as np
import random


class BatchingMechanism:

    def __init__(self, dataset_split, number_examples, batch_size, storage_size):

        self.number_examples = number_examples
        self.batch_size = batch_size

        self.storage_size = storage_size

        self.batching_storage_info = None

        self.storage = Storage(dataset_split)

    # based on the required number of examples and on the storage size
    # compute how many storages and how many examples per storage are needed
    # storages are considered sequentially (0, 1, 2...)
    # index are considered sequentially (0, 1, 2...)
    def initialise(self):

        # start with the total number of examples
        residual_number_of_examples = self.number_examples

        batching_storage_info = []
        storage_number = 0

        # until there are outstanding examples
        while residual_number_of_examples > 0:

            # if suitable, use an entire storage
            if residual_number_of_examples >= self.storage_size:

                # record the number of the storage and its size
                batching_storage_info.append((storage_number, self.storage_size))

                # reduce the outstanding examples
                residual_number_of_examples -= self.storage_size
                # move to the next container
                storage_number += 1

            else:

                # record the number of the storage and its size
                # the last container is only partially used
                batching_storage_info.append((storage_number, residual_number_of_examples))

                # reduce the outstanding examples
                residual_number_of_examples = 0

        # the output is a list of tuples indicating the number of the storage
        # and how many examples of that storage to use
        self.batching_storage_info = batching_storage_info

    # internal function to partition a list into chunks
    # as determined by the batch size
    def _divide_into_chunks(self, indices):

        for i in range(0, len(indices), self.batch_size):

            yield indices[i:i + self.batch_size]

    # build the batches: every batch is described by a tuple
    # the first element is the number of the storage and
    # the second element is a list of indices of examples
    # within the storage
    def build_batches(self, randomise):

        batches = []

        # loop through the list of containers to be used
        # we know the number of the container and how
        # many examples within it to be used
        for info in self.batching_storage_info:

            storage_number = info[0]
            number_examples = info[1]

            # full list of indices to be used
            indices_to_batch = list(range(number_examples))

            # optionally, shuffle the indices
            if randomise:

                random.shuffle(indices_to_batch)

            # partition the indices into chunks as determined by the batch size
            batches_indices = list(self._divide_into_chunks(indices_to_batch))

            # build the list of batches
            for batch_indices in batches_indices:

                batches.append((storage_number, batch_indices))

        # optionally shuffle the batches
        if randomise:

            random.shuffle(batches)

        return batches

    # trim the the array removing trailing columns
    # filled with '<pad>' for all examples
    @staticmethod
    def _trim_structural_tokens(padded_structural_tokens):

        # one boolean per column
        # True if at least one entry is not '<pad>' (=0)
        is_any_data = np.any(padded_structural_tokens != 0, axis=0)

        # indices of columns with at least one entry which is not '<pad>'
        indices = np.where(is_any_data)

        # first index of columns with all '<pad>'
        # all subsequent columns are filled with '<pad>'
        cut_off_index = np.max(indices) + 1

        # trim the columns with all '<pad>'
        trimmed_structural_tokens = padded_structural_tokens[:, :cut_off_index]

        return trimmed_structural_tokens

    # trim the the array removing trailing rows and trailing columns
    # filled with '<pad>' for all examples and all time-steps
    @staticmethod
    def _trim_cells_content_tokens(padded_cells_content_tokens):

        # one boolean per row
        # True if at least one entry is not '<pad>'
        is_any_data = np.any(padded_cells_content_tokens != 0, axis=2)

        # indices of rows with at least one entry which is not '<pad>'
        indices = np.where(is_any_data)[1]

        # first index of rows with all '<pad>'
        # all subsequent rows are filled with '<pad>'
        cut_off_index = np.max(indices) + 1

        # trim the rows with all '<pad>'
        trimmed_cells_content_tokens = padded_cells_content_tokens[:, :cut_off_index, :]

        # one boolean per column
        # True if at least one entry is not '<pad>'
        is_any_data = np.any(trimmed_cells_content_tokens != 0, axis=1)

        # indices of columns with at least one entry which is not '<pad>'
        indices = np.where(is_any_data)[1]

        # first index of columns with all '<pad>'
        # all subsequent columns are filled with '<pad>'
        cut_off_index = np.max(indices) + 1

        # trim the columns with all '<pad>'
        trimmed_cells_content_tokens = trimmed_cells_content_tokens[:, :, :cut_off_index]

        return trimmed_cells_content_tokens

    # given a batch, retrieve the corresponding data
    # the output are already tensors with the required data type
    def get_batch(self, batch):

        # get the essential data about the batch
        storage_number = batch[0]
        indices = batch[1]

        # open the required storage
        self.storage.open(storage_number)

        # get the features maps from the HDF5 file
        features_maps = np.array(list(map(self.storage.features_maps.__getitem__, indices))).astype(np.float32)
        features_maps = torch.from_numpy(features_maps)

        # get the structural tokens from the HDF5 file
        structural_tokens = np.array(list(map(self.storage.structural_tokens.__getitem__, indices))).astype(np.int64)
        structural_tokens = self._trim_structural_tokens(structural_tokens)
        structural_tokens = torch.from_numpy(structural_tokens)

        # get the triggers from the HDF5 file
        triggers = np.array(list(map(self.storage.triggers.__getitem__, indices))).astype(np.int64)
        triggers = torch.from_numpy(triggers)

        # get the cell content from the HDF5 file
        cells_content_tokens = np.array(list(map(self.storage.cells_content_tokens.__getitem__, indices))).astype(np.int64)
        cells_content_tokens = self._trim_cells_content_tokens(cells_content_tokens)
        cells_content_tokens = torch.from_numpy(cells_content_tokens)

        # close the storage
        self.storage.close()

        return features_maps, structural_tokens, triggers, cells_content_tokens

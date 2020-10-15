import Utils
import h5py


class Storage:

    def __init__(self):

        # the HDF5 File object
        # assigned when opening the storage
        self.storage = None

        # the HDF5 Dataset objects
        # assigned when opening the storage
        self.features_maps = None
        self.structural_tokens = None
        self.triggers = None
        self.cells_content_tokens = None

    def open(self, storage_suffix):

        # open the storage
        suffix = '{:0>2}'.format(storage_suffix)
        storage_path = Utils.create_abs_path('Dataset/dataset_' + suffix + '.hdf5')
        self.storage = h5py.File(storage_path, "r")

        # set up the containers
        # note: they can be accessed like if they were arrays
        # the process of fetching them from disk is transparent
        self.features_maps = self.storage['features maps']
        self.structural_tokens = self.storage['structural tokens']
        self.triggers = self.storage['triggers']
        self.cells_content_tokens = self.storage['cells content tokens']

    def get_data(self, indexing):

        data = self.features_maps[indexing]

        return data

    def close(self):

        # close storage
        self.storage.close()


storage = Storage()
storage.open(0)
storage.get_data([1:5])
storage.close()


class DataRetriever:

    def __init__(self):

        self.storage = Storage()

    def open(self, storage_suffix):

        self.storage.open(storage_suffix)

    def get_features_maps(self, indexing):

        features_maps = self.storage.features_maps[indexing]

        return features_maps

    def close(self):

        self.storage.close()

















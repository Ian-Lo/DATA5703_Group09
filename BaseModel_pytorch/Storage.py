import Utils
import h5py


class Storage:

    def __init__(self, dataset_split):

        self.dataset_split = dataset_split

        self.storage_number = None
        self.storage = None
        self.features_maps = None
        self.structural_tokens = None
        self.triggers = None
        self.cells_content_tokens = None

    # open a HDF5 storage file given its number (i.d. its numerical suffix)
    def open(self, storage_number):

        self.storage_number = storage_number

        # connect to the HFD5 file
        suffix = '{:0>3}'.format(self.storage_number)
        storage_filename = self.dataset_split + '_dataset_' + suffix + '.hdf5'
        storage_path = Utils.create_abs_path('Dataset/' + storage_filename)
        self.storage = h5py.File(storage_path, "r")

        # set up dataset objects
        self.features_maps = self.storage['features maps']
        self.structural_tokens = self.storage['structural tokens']
        self.triggers = self.storage['triggers']
        self.cells_content_tokens = self.storage['cells content tokens']

    # close a HDF5 file and release the member data
    def close(self):

        # close the storage
        self.storage.close()

        # release the member data
        self.storage_number = None
        self.storage = None
        self.features_maps = None
        self.structural_tokens = None
        self.triggers = None
        self.cells_content_tokens = None



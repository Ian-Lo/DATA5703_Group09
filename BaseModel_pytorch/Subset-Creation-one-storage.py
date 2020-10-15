import Utils
from FixedEncoder import FixedEncoder
import PIL
import jsonlines
import numpy as np
import pandas as pd
import h5py
import torch


# instantiate the fixed CNN encoder
# with ResNet-18, the features map will be (features_map_size * features_map_size, 512)
features_map_size = 12
fixedEncoder = FixedEncoder('ResNet18', features_map_size)

# load vocabularies
structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()
cell_content_token2integer, cell_content_integer2token = Utils.load_cell_content_vocabularies()

# HDF5 storage
# number of examples in each HDF5 file
storage_size = 100
# batch size for the encoder
# and all other processing steps
# choose a divisor of storage_size
chunk_size = 10

# total number of examples to process and store
# it needs to be a multiple of the storage size
num_examples = 200

# note: these numbers are a result of the way the images
# are preprocessed in the CNN encoder (e.g. 512 is determined by ResNet-18)
# shape of a feature map
h_fm = features_map_size * features_map_size
w_fm = 512

# note: these numbers are the *actual* criteria found in SubsetCriteria/subset_analysis.txt
# max length of structural tokens
w_st_tks = 154
# we add 1 to account for the future addition of the '<end>' token
w_st_tks += 1
# max shape of cells content tokens
h_cc_tks = 30
w_cc_tks = 50
# we add 1 to account for the future addition of the '<end>' token
w_cc_tks += 1

# placeholders for HDF5 storages
storage_features_maps = None
storage_structural_tokens = None
storage_triggers = None
storage_cells_content_tokens = None
data_features_maps = None
data_structural_tokens = None
data_triggers = None
data_cells_content_tokens = None

# storage suffix number
# used to name the storage
suffix = 0

# placeholders for the temporary storages used during the processing of a chunk
images = None
structural_tokens_store = None
triggers_store = None
cells_content_tokens_store = None

# index over the examples fitting the subsetting criteria
idx_example = 0

# index within the chunk
idx_chunk = 0

# open the annotations file
annotations_filename = Utils.create_abs_path('pubtabnet/PubTabNet_2.0.0.jsonl')
annotations = jsonlines.open(annotations_filename, 'r')

# create the metadata file
metadata_filename = Utils.create_abs_path('Dataset/metadata.jsonl')
metadata = jsonlines.open(metadata_filename, 'w')

# imgids of examples to use for the creation of the subset
# criteria are explored in Subset_Criteria_Analysis.py
subset_imgids = np.load('SubsetCriteria/subset_imgids.npy')

for annotation in annotations:

    imgid = annotation['imgid']

    # only process imgids fulfilling certain criteria
    if imgid in subset_imgids:

        # check if we need to create new H5DF storages
        if idx_example % storage_size == 0:

            # close previous HDF5 storages
            # except at the very beginning
            if idx_example != 0:

                storage_features_maps.close()
                storage_structural_tokens.close()
                storage_triggers.close()
                storage_cells_content_tokens.close()

            # get storage suffix
            suffix = idx_example // storage_size
            suffix = '{:0>2}'.format(suffix)

            # create new HDF5 storage
            storage_features_maps_path = Utils.create_abs_path('Dataset/features_maps_' + suffix + '.hdf5')
            storage_features_maps = h5py.File(storage_features_maps_path, "w")
            data_features_maps = storage_features_maps.create_dataset('data',
                                                                      shape=(storage_size, h_fm, w_fm),
                                                                      dtype=np.float16)

            # create new HDF5 storage
            storage_structural_tokens_path = Utils.create_abs_path('Dataset/structural_tokens_' + suffix + '.hdf5')
            storage_structural_tokens = h5py.File(storage_structural_tokens_path, "w")
            data_structural_tokens = storage_structural_tokens.create_dataset('data',
                                                                              shape=(storage_size, w_st_tks),
                                                                              dtype=np.uint8)

            # create new HDF5 storage
            storage_triggers_path = Utils.create_abs_path('Dataset/triggers_' + suffix + '.hdf5')
            storage_triggers = h5py.File(storage_triggers_path, "w")
            data_triggers = storage_triggers.create_dataset('data',
                                                            shape=(storage_size, h_cc_tks),
                                                            dtype=np.uint8)

            # create new HDF5 storage
            storage_cells_content_tokens_path = Utils.create_abs_path('Dataset/cells_content_tokens_' + suffix + '.hdf5')
            storage_cells_content_tokens = h5py.File(storage_cells_content_tokens_path, "w")
            data_cells_content_tokens = storage_cells_content_tokens.create_dataset('data',
                                                                                    shape=(storage_size, h_cc_tks, w_cc_tks),
                                                                                    dtype=np.uint16)

        # compute index within the chunk
        idx_chunk = idx_example % chunk_size

        # check if we need initialise temporary storages
        if idx_chunk == 0:

            images = []
            structural_tokens_store = np.zeros((chunk_size, w_st_tks))
            triggers_store = np.zeros((chunk_size, h_cc_tks))
            cells_content_tokens_store = np.zeros((chunk_size, h_cc_tks, w_cc_tks))

        # retrieve the image
        filename = annotation['filename']
        image_filename = Utils.create_abs_path('pubtabnet/train/' + filename)
        image = PIL.Image.open(image_filename)

        # compute index within storage
        idx_storage = idx_example % storage_size

        # assemble the metadata
        meta = {'imgid': imgid,
                'filename': filename,
                'storage': suffix,
                'index': idx_storage}
        # write the metadata
        metadata.write(meta)

        # put into temporary storage
        images.append(image)

        # retrieve the structural tokens
        structural_tokens = annotation['html']['structure']['tokens']

        # add the '<end>' token
        structural_tokens = structural_tokens + ['<end>']

        # encode the structural tokens
        structural_tokens_int = [structural_token2integer[tk] for tk in structural_tokens]
        structural_tokens_int = np.array(structural_tokens_int).astype(np.uint8)
        structural_tokens_int_padded = np.zeros(w_st_tks, dtype=np.uint8)
        structural_tokens_int_padded[:structural_tokens_int.shape[0]] = structural_tokens_int

        # put into temporary storage
        structural_tokens_store[idx_chunk] = structural_tokens_int_padded

        # find the trigger points where a new cell start
        # these are needed in order to know when to preserve
        # the hidden state of the first decoder
        triggers = [idx for idx, tk in enumerate(structural_tokens) if (tk == '<td>') | (tk == '>')]
        triggers = np.array(triggers).astype(np.uint8)
        triggers_padded = np.zeros(h_cc_tks, dtype=np.uint8)
        triggers_padded[:triggers.shape[0]] = triggers

        # put into temporary storage
        triggers_store[idx_chunk] = triggers_padded

        # retrieve the cells content tokens
        cells_content_tokens = [cell['tokens'] for cell in annotation['html']['cells']]

        # add the '<end>' token
        cells_content_tokens = [cell_content_tokens + ['<end>'] for cell_content_tokens in cells_content_tokens]

        # encode the cells content tokens
        cells_content_tokens_int = [[cell_content_token2integer[tk] for tk in cell_content_tokens] for cell_content_tokens in cells_content_tokens]
        cells_content_tokens_int = pd.DataFrame(cells_content_tokens_int).fillna(0).values.astype(np.uint16)
        cells_content_tokens_int_padded = np.zeros((h_cc_tks, w_cc_tks), dtype=np.uint16)
        cells_content_tokens_int_padded[:cells_content_tokens_int.shape[0], :cells_content_tokens_int.shape[1]] = cells_content_tokens_int

        # put into temporary storage
        cells_content_tokens_store[idx_chunk] = cells_content_tokens_int_padded

        # if the end of a chunk as been reached, add to HDF5 storage
        if idx_chunk == (chunk_size - 1):

            print(idx_example)

            # batch encoding of the images
            input_images = [fixedEncoder.preprocess(image) for image in images]
            input_images = torch.stack(input_images)
            features_maps = fixedEncoder.encode(input_images)

            # start
            start_index = (idx_example - chunk_size + 1) % storage_size
            end_index = (idx_example + 1) % storage_size
            if end_index == 0:
                end_index = storage_size

            # add to H5DF storage file
            data_features_maps[start_index:end_index] = features_maps
            data_structural_tokens[start_index:end_index] = structural_tokens_store
            data_triggers[start_index:end_index] = triggers_store
            data_cells_content_tokens[start_index:end_index] = cells_content_tokens_store

        # increment the counter
        idx_example += 1

        # check if we need to close the current HDF5 storages
        if idx_example % storage_size == 0:

            storage_features_maps.close()
            storage_structural_tokens.close()
            storage_triggers.close()
            storage_cells_content_tokens.close()

        # check if we have collected enough examples
        if idx_example == num_examples:
            break

# close the JSON files
annotations.close()
metadata.close()
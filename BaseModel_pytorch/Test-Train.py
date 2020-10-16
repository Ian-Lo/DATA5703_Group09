import Utils
from Encoder import Encoder
from DecoderStructural import DecoderStructural
from DecoderCellContent import DecoderCellContent
import h5py
import torch
import numpy as np
from time import perf_counter

structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()
cell_content_token2integer, cell_content_integer2token = Utils.load_cell_content_vocabularies()

suffix = '00'

storage_features_maps_path = Utils.create_abs_path('Dataset/features_maps_' + suffix + '.hdf5')
storage_features_maps = h5py.File(storage_features_maps_path, "r")

storage_structural_tokens_path = Utils.create_abs_path('Dataset/structural_tokens_' + suffix + '.hdf5')
storage_structural_tokens = h5py.File(storage_structural_tokens_path, "r")

storage_triggers_path = Utils.create_abs_path('Dataset/triggers_' + suffix + '.hdf5')
storage_triggers = h5py.File(storage_triggers_path, "r")

storage_cells_content_tokens_path = Utils.create_abs_path('Dataset/cells_content_tokens_' + suffix + '.hdf5')
storage_cells_content_tokens = h5py.File(storage_cells_content_tokens_path, "r")

features_maps = storage_features_maps['data']
structural_tokens = storage_structural_tokens['data']
triggers = storage_triggers['data']
cells_content_tokens = storage_cells_content_tokens['data']

# batch of examples
# TODO: you have overridden the storage by renaming!
features_map = features_maps[0:10].astype(np.float32)
features_map = torch.from_numpy(features_map)

structural_tokens = structural_tokens[0:10].astype(np.int64)
structural_tokens = torch.from_numpy(structural_tokens)

triggers = triggers[0:10]

cells_content_tokens = cells_content_tokens[0:10].astype(np.int64)
cells_content_tokens = torch.from_numpy(cells_content_tokens)

###---###

# set up the encoder for the features maps
features_map_size = features_map.size()[-1]
encoder_size = 200

encoder = Encoder(features_map_size, encoder_size)
encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()))

# set up the decoder for structural tokens
embedding_size = 15
structural_hidden_size = 100
structural_attention_size = 50

decoder_structural = DecoderStructural(structural_token2integer, embedding_size, encoder_size, structural_hidden_size, structural_attention_size)
decoder_structural_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_structural.parameters()))

# set up the decoder for cell content tokens
cell_content_hidden_size = 100
cell_content_attention_size = 50

decoder_cell_content = DecoderCellContent(cell_content_token2integer, embedding_size, encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size)
decoder_cell_content_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_cell_content.parameters()))

# set up training variables
epochs = 5


t1_start = perf_counter()

# what are we looping over? if j is a dummy variable, rename to "_"
for epoch in range(epochs):

    encoded_features_map = encoder.forward(features_map)

    predictions, loss_s, storage = decoder_structural.forward(encoded_features_map, structural_tokens, train = True)

    ### PROCESSING STORAGE ###
    list1 = []
    list2 = []
    list3 = []

    for example_num, example_triggers in enumerate(triggers):

        cc_tk = cells_content_tokens[example_num]

        for cell_num, example_trigger in enumerate(example_triggers):

            if example_trigger != 0:
                list1.append(encoded_features_map[example_num])

                list2.append(storage[example_trigger, 0, example_num, :])

                list3.append(cc_tk[cell_num])

    new_encoded_features_map = torch.stack(list1)
    structural_hidden_state = torch.stack(list2).unsqueeze(0)
    new_cells_content_tokens = torch.stack(list3)

    print(new_encoded_features_map.size(), structural_hidden_state.size(), cells_content_tokens.size())
    ###

    predictions, loss_cc = decoder_cell_content.forward(new_encoded_features_map, structural_hidden_state, new_cells_content_tokens)

    loss = loss_s + loss_cc

    print(loss)

    # Back propagation
    decoder_cell_content_optimizer.zero_grad()
    decoder_structural_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    loss.backward()

    # Update weights
    decoder_cell_content_optimizer.step()
    decoder_structural_optimizer.step()
    encoder_optimizer.step()

t1_stop = perf_counter()

print('time: ', t1_stop-t1_start)

storage_features_maps.close()
storage_structural_tokens.close()
storage_triggers.close()
storage_cells_content_tokens.close()

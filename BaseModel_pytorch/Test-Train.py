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
#print(structural_integer2token)
#quit()
suffix = '00'

storage_path = Utils.create_abs_path('Dataset/dataset_' + suffix + '.hdf5')
storage = h5py.File(storage_path, "r")

features_maps = storage['features maps']
structural_tokens = storage['structural tokens']
triggers = storage['triggers']
cells_content_tokens = storage['cells content tokens']

# batch of examples
# TODO: you have overridden the storage by renaming!
features_map = features_maps[0:10].astype(np.float32)
features_map = torch.from_numpy(features_map)

structural_tokens = structural_tokens[0:10].astype(np.int64)
structural_tokens = torch.from_numpy(structural_tokens)

triggers = triggers[0:10]

cells_content_tokens = cells_content_tokens[0:10].astype(np.int64)
cells_content_tokens = torch.from_numpy(cells_content_tokens)

#### validation set import ####
features_maps_val = storage['features maps']
structural_tokens_val = storage['structural tokens']
triggers_val = storage['triggers']
cells_content_tokens_val = storage['cells content tokens']

# batch of examples
# TODO: you have overridden the storage by renaming!
features_map_val = features_maps_val[0:10].astype(np.float32)
features_map_val = torch.from_numpy(features_map_val)

structural_tokens_val = structural_tokens_val[0:10].astype(np.int64)
structural_tokens_val = torch.from_numpy(structural_tokens_val)

triggers_val = triggers_val[0:10]

cells_content_tokens_val = cells_content_tokens_val[0:10].astype(np.int64)
cells_content_tokens_val = torch.from_numpy(cells_content_tokens_val)

####

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

t1_start = perf_counter()

epochs = 5

# Loss(total) = LAMBDA * Loss(structure) + (1-LAMBDA) * Loss(cell)
LAMBDA = 0.5




for epoch in range(epochs):
    #todo: add for loop for batches

    ##### training #####
    encoded_features_map = encoder.forward(features_map)

    predictions, loss_s, storage_hidden = decoder_structural.forward(encoded_features_map, structural_tokens)

    ### PROCESSING STORAGE ###
    list1 = []
    list2 = []
    list3 = []

    for example_num, example_triggers in enumerate(triggers):

        cc_tk = cells_content_tokens[example_num]

        for cell_num, example_trigger in enumerate(example_triggers):

            if example_trigger != 0:
                list1.append(encoded_features_map[example_num])

                list2.append(storage_hidden[example_trigger, 0, example_num, :])

                list3.append(cc_tk[cell_num])


    new_encoded_features_map = torch.stack(list1)
    structural_hidden_state = torch.stack(list2).unsqueeze(0)
    new_cells_content_tokens = torch.stack(list3)

#    predictions_cell, loss_cc = decoder_cell_content.forward(new_encoded_features_map, structural_hidden_state, new_cells_content_tokens)

#    loss = LAMBDA * loss_s + (1.0-LAMBDA) * loss_cc

#    print(loss)

    # Back propagation
#    decoder_cell_content_optimizer.zero_grad()#
#    decoder_structural_optimizer.zero_grad()
#    encoder_optimizer.zero_grad()
#    loss.backward()

    # Update weights
#    decoder_cell_content_optimizer.step()#
#    decoder_structural_optimizer.step()
#    encoder_optimizer.step()

    ##### validation #####
    # batch loop for validation set (only one batch, because batches are not implemented)
    encoded_features_map_val = encoder.forward(features_map_val) # change features_map to feature_map_val

    predictions_val, loss_s_val, storage_hidden_val = decoder_structural.predict(encoded_features_map_val, structural_target = structural_tokens_val )
    break
    ### PROCESSING STORAGE ###
    list1 = []
    list2 = []
    list3 = []

    for example_num, example_triggers in enumerate(triggers):

        cc_tk = cells_content_tokens[example_num]

        for cell_num, example_trigger in enumerate(example_triggers):

            if example_trigger != 0:
                list1.append(encoded_features_map[example_num])

                list2.append(storage_hidden[example_trigger, 0, example_num, :])

                list3.append(cc_tk[cell_num])


    new_encoded_features_map = torch.stack(list1)
    structural_hidden_state = torch.stack(list2).unsqueeze(0)
    new_cells_content_tokens = torch.stack(list3)

    predictions_cell, loss_cc = decoder_cell_content.forward(new_encoded_features_map, structural_hidden_state, new_cells_content_tokens)
    ####### validation end ########

t1_stop = perf_counter()

print('time: ', t1_stop-t1_start)

storage.close()

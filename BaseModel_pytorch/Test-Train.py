import Utils
from Encoder import Encoder
from DecoderStructural import DecoderStructural
from DecoderCellContent import DecoderCellContent
import h5py
import torch
import numpy as np
from time import perf_counter
from BatchingMechanism import BatchingMechanism
from TrainStep import train_step
from Model import Model

structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()
cell_content_token2integer, cell_content_integer2token = Utils.load_cell_content_vocabularies()

# instantiate the batching object
batching = BatchingMechanism(dataset_split='train', number_examples=125, batch_size=8, storage_size=1000)
batching_val = BatchingMechanism(dataset_split='train', number_examples=125, batch_size=8, storage_size=1000)

# initialise the object
# here the object works out how many storages and how many examples from every storage are needed
batching.initialise()
batching_val.initialise()

# import batch to calculate size of feature maps used in preprocessing
# this size is required to initialize the encoder.
batch= batching.build_batches(randomise=True)[0]
features_map, _, _, _ = batching.get_batch(batch)
features_map_size = features_map.size()[-1]

# then reinitialize so we haven't used up batch
batching.initialise()

# initialize encoder
encoder_size = 144 # this will make the feature maps square
encoder = Encoder(features_map_size, encoder_size)
encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()))

# set up the decoder for structural tokens
structural_embedding_size = 16
structural_hidden_size = 256
structural_attention_size = 256

decoder_structural = DecoderStructural(structural_token2integer, structural_embedding_size, encoder_size, structural_hidden_size, structural_attention_size)
decoder_structural_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_structural.parameters(), lr = 0.001))

# set up the decoder for cell content tokens
cell_content_embedding_size = 80
cell_content_hidden_size = 512
cell_content_attention_size = 256

decoder_cell_content = DecoderCellContent(cell_content_token2integer, cell_content_embedding_size, encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size)
decoder_cell_content_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_cell_content.parameters(), lr = 0.001))

# initialize model class
model = Model(encoder, encoder_optimizer, decoder_structural,decoder_structural_optimizer,decoder_cell_content,decoder_cell_content_optimizer,structural_token2integer, structural_integer2token , cell_content_token2integer, cell_content_integer2token )

t1_start = perf_counter()

# set number of epochs
epochs = 10

# make list of lambdas to use in training
# We set LAMBDA = 0 for the first five epochs to pretrain the structural decoder.
lambdas = [1 for _ in range(13)] + [0.5 for _ in range(5)] + [0.5 for _ in range(12)]
lrs = [0.001 for _ in range(10)] + [0.0001 for n in range(3)] + [0.001 for _ in range(10) + [0.0001 for _ in range(2)]

for epoch in range(epochs):
    # create random batches of examples
    # these "batches" are the just information needed to retrieve the actual tensors
    # batch = (storage number, [list of indices within the storage])
    batches = batching.build_batches(randomise=True)

    LAMBDA = lambdas[epoch]
    # set learning rate manually for each epoch
    lr = lrs[epoch]
    for g in decoder_structural_optimizer.param_groups:
        g['lr'] = lr
    for g in decoder_cell_content_optimizer.param_groups:
        g['lr'] = lr
    for g in encoder_optimizer.param_groups:
        g['lr'] = lr

    # batch looping for training
    for batch in batches:

        # call 'get_batch' to actually load the tensors from file
        features_maps, structural_tokens, triggers, cells_content_tokens = batching.get_batch(batch)

        # send to training function for forward pass, backpropagation and weight updates
        predictions, loss_s, predictions_cell, loss_cc, loss = train_step(features_maps, structural_tokens, triggers, cells_content_tokens, model,LAMBDA=LAMBDA)

    #create batches for validation set
    batches_val= batching_val.build_batches(randomise=False)

    # batch looping for validation
    for batch in batches_val:
        # call 'get_batch' to actually load the tensors from file
        features_maps_val, structural_tokens_val, triggers_val, cells_content_tokens_val = batching_val.get_batch(batch)

        encoded_features_map_val = encoder.forward(features_map_val)
        predictions_val, loss_s_val, storage_hidden_val, pred_triggers = decoder_structural.predict(encoded_features_map_val, structural_target = structural_tokens_val )

        ### PROCESSING STORAGE ###
        print("epoch", epoch)
        print("training loss", loss_s)
        print("validation loss", loss_s_val)

        # merge input for predicted and ground truth

        ### PROCESSING STORAGE ###
        list1 = []
        list2 = []
        list3 = []

        for example_num, example_triggers in enumerate(pred_triggers):
            print(example_num, example_trigger)
            # find true predicted tokens for predicted cell
            ##### this is where I am at ##### reverting to implementing batching.

        for example_num, example_triggers in enumerate(triggers_val):

            cc_tk = cells_content_tokens[example_num]

            for cell_num, example_trigger in enumerate(example_triggers):

                if example_trigger != 0:
                    list1.append(encoded_features_map[example_num])

                    list2.append(storage_hidden[example_trigger, 0, example_num, :])

                    list3.append(cc_tk[cell_num])

#        new_encoded_features_map = torch.stack(list1)
#        structural_hidden_state = torch.stack(list2).unsqueeze(0)
        new_cells_content_tokens = torch.stack(list3)

        predictions_cell, loss_cc_val = decoder_cell_content.predict(encoded_features_map, storage_hidden_val,cell_content_target =new_cells_content_tokens  )
        ####### validation end ########

#t1_stop = perf_counter()

print('time: ', t1_stop-t1_start)

storage.close()

import Utils
from Encoder import Encoder
from DecoderStructural import DecoderStructural
from DecoderCellContent import DecoderCellContent
import h5py
import torch
import numpy as np
from time import perf_counter, time
from BatchingMechanism import BatchingMechanism
from TrainStep import train_step
from ValStep import val_step
from Model import Model
from CheckPoint import CheckPoint

# set relative path to load data from
Utils.DatasetPath.set_relative_path("../../Dataset")

structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()
cell_content_token2integer, cell_content_integer2token = Utils.load_cell_content_vocabularies()

# instantiate the batching object
number_examples =30
number_examples_val = 30 # not implemented
batching = BatchingMechanism(dataset_split='train', number_examples=number_examples, batch_size=10, storage_size=1000)
batching_val = BatchingMechanism(dataset_split='train', number_examples=number_examples_val, batch_size=10, storage_size=1000)

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
decoder_structural_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_structural.parameters()))

# set up the decoder for cell content tokens
cell_content_embedding_size = 80
cell_content_hidden_size = 512
cell_content_attention_size = 256

decoder_cell_content = DecoderCellContent(cell_content_token2integer, cell_content_embedding_size, encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size)
decoder_cell_content_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_cell_content.parameters()))

# initialize model class
model = Model(encoder, encoder_optimizer, decoder_structural,decoder_structural_optimizer,decoder_cell_content,decoder_cell_content_optimizer,structural_token2integer, structural_integer2token , cell_content_token2integer, cell_content_integer2token )


# set number of epochs
epochs = 25

# make list of lambdas to use in training
# this is the same strategy as Zhong et al.
lambdas =  [1, 1 ] + [0.5 for _ in range(8)] + [1 for _ in range(3)]+ [0.5 for _ in range(10)] + [0.5 for _ in range(2)]
lrs = [0.001 for _ in range(10)] + [0.0001 for _ in range(3)] + [0.001 for _ in range(10)] + [0.0001 for _ in range(2)]

assert epochs == len(lambdas) == len(lrs), "number of epoch, learning rates and lambdas are inconsistent"

# construct checkpoint
checkpoint = CheckPoint("BaselineModel_checkpoints")




for epoch in range(epochs):
    t1_start = perf_counter()

    # reset total loss across epoch
    total_loss_s = 0
    total_loss_cc = 0
    total_loss = 0
    total_loss_s_val = 0
    total_loss_cc_val = 0

    # create random batches of examples
    # these "batches" are the just information needed to retrieve the actual tensors
    # batch = (storage number, [list of indices within the storage])
    batches = batching.build_batches(randomise=True)

    LAMBDA = lambdas[epoch]

    # update learning rates manually for each epoch
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
        total_loss_s+=loss_s
        total_loss+= loss
        if loss_cc:
            total_loss_cc+=loss_cc
    total_loss_s/=len(batches)

    checkpoint.save_checkpoint(epoch, encoder, decoder_structural, decoder_cell_content,
                              encoder_optimizer, decoder_structural_optimizer, decoder_cell_content_optimizer, total_loss, total_loss_s, total_loss_cc)
    checkpoint.archive_checkpoint()

    #batch loop for validation
    with torch.no_grad():
        batches_val= batching_val.build_batches(randomise=False)

        #batch looping for validation
        for batch in batches_val:
            # call 'get_batch' to actually load the tensors from file
            features_maps_val, structural_tokens_val, triggers_val, cells_content_tokens_val = batching_val.get_batch(batch)
            predictions_val, loss_s_val, predictions_cell_val, loss_cc_val, loss_val = val_step(features_maps_val, structural_tokens_val,triggers_val,cells_content_tokens_val, model, LAMBDA  )

            total_loss_s_val+= loss_s_val

        total_loss_s_val/=len(batches)

        print("Truth")
        print([structural_integer2token[p.item()] for p in structural_tokens_val[0][:10]])
        print("Prediction")
        print([structural_integer2token[p.item()] for p in predictions_val[0][:10]])
    ######################

    t1_stop = perf_counter()
    print("----------------------")
    print('epoch: %d \tLAMBDA: %.2f\tlr:%.5f\ttime: %.2f'%(epoch,LAMBDA, lr, t1_stop-t1_start))
    print('Total loss: %.5f'%total_loss)
    print('Struct. decod. loss: %.5f'%total_loss_s)
    print("Cell dec. loss:", total_loss_cc)
    print('Validation struct. decod. loss: %.5f'%loss_s_val)
    print('time for 100k examples:' , "%.2f hours"%((t1_stop-t1_start)/number_examples*100000/3600))

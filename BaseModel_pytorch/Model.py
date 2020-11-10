import Utils

from Encoder import Encoder
from DecoderStructural import DecoderStructural
from DecoderCellContent import DecoderCellContent

from BatchingMechanism import BatchingMechanism
from CheckPoint import CheckPoint
from TrainStep import train_step
from ValStep import val_step

import torch
from time import perf_counter, time


class Model:
    """Combined class for encoder, structural decoder and cell decoder."""

    def __init__(self, relative_path, model_tag, in_channels=512, out_channels=16,encoder_size =12, structural_embedding_size=16, structural_hidden_size=256, structural_attention_size=256, cell_content_embedding_size=80, cell_content_hidden_size=512, cell_content_attention_size=256):

        # set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors

        # set model tag to be used to identify checkpoint files
        self.model_tag = model_tag

        # set up path of dataset
        Utils.DatasetPath.set_relative_path(relative_path)

        # load dictionaries
        structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()
        cell_content_token2integer, cell_content_integer2token = Utils.load_cell_content_vocabularies()
        self.structural_token2integer = structural_token2integer
        self.structural_integer2token = structural_integer2token
        self.cell_content_token2integer = cell_content_token2integer
        self.cell_content_integer2token = cell_content_integer2token

        # initialize encoder
        encoder = Encoder(in_channels, out_channels)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()))
        self.encoder = encoder.to(self.device)
        self.encoder_optimizer = encoder_optimizer

        # set up the decoder for structural tokens
        decoder_structural = DecoderStructural(structural_token2integer, structural_embedding_size, encoder_size, structural_hidden_size, structural_attention_size)
        decoder_structural_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_structural.parameters()))
        self.decoder_structural = decoder_structural.to(self.device)
        self.decoder_structural_optimizer = decoder_structural_optimizer

        # set up the decoder for cell content tokens
        decoder_cell_content = DecoderCellContent(cell_content_token2integer, cell_content_embedding_size, encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size)
        decoder_cell_content_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder_cell_content.parameters()))

        self.decoder_cell_content = decoder_cell_content.to(self.device)
        self.decoder_cell_content_optimizer = decoder_cell_content_optimizer

    def set_eval(self):
        '''Change to evaluation state.'''
        self.decoder_structural = self.decoder_structural.eval()
        self.decoder_cell_content = self.decoder_cell_content.eval()
        self.encoder = self.encoder.eval()

    def set_train(self):
        ''' Change to training state'''
        self.decoder_structural = self.decoder_structural.train()
        self.decoder_cell_content = self.decoder_cell_content.train()
        self.encoder = self.encoder.train()

    def train(self, drive=None, checkpoint_temp_id=None, epochs=1, lambdas=[1], lrs=[0.001], number_examples=100, number_examples_val=100, batch_size=10, storage_size=1000,val = None ):

        assert epochs == len(lambdas) == len(lrs), "number of epoch, learning rates and lambdas are inconsistent"

        # instantiate the batching object
        batching = BatchingMechanism(dataset_split='train', number_examples=number_examples, batch_size=batch_size, storage_size=storage_size)


        # initialise the object
        # here the object works out how many storages and how many examples from every storage are needed
        batching.initialise()

        if val:
            batching_val = BatchingMechanism(dataset_split='val', number_examples=number_examples_val, batch_size=10, storage_size=storage_size)
            batching_val.initialise()

        # instantiate checkpoint
        checkpoint = CheckPoint(self.model_tag, drive=drive, checkpoint_temp_id=checkpoint_temp_id)

        # then reinitialize so we haven't used up batch
        batching.initialise()

        for epoch in range(epochs):
            print(epoch)
            # change model to training
            self.set_train()

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
            for g in self.decoder_structural_optimizer.param_groups:
                g['lr'] = lr
            for g in self.decoder_cell_content_optimizer.param_groups:
                g['lr'] = lr
            for g in self.encoder_optimizer.param_groups:
                g['lr'] = lr

            # batch looping for training
            for batch in batches:

                # call 'get_batch' to actually load the tensors from file
                features_maps, structural_tokens, triggers, cells_content_tokens = batching.get_batch(batch)
                # test greedy
                structural_tokens = structural_tokens[:, 0:2]
                triggers = triggers[:,0:2]
                #####
                # send to training function for forward pass, backpropagation and weight updates
                predictions, loss_s, predictions_cell, loss_cc, loss = train_step(features_maps, structural_tokens, triggers, cells_content_tokens, self, LAMBDA=LAMBDA)
                total_loss_s += loss_s
                total_loss += loss
                if loss_cc:
                    total_loss_cc+=loss_cc

            total_loss_s /= len(batches)

            checkpoint.save_checkpoint(epoch, self.encoder, self.decoder_structural, self.decoder_cell_content,
                                      self.encoder_optimizer, self.decoder_structural_optimizer, self.decoder_cell_content_optimizer, total_loss, total_loss_s, total_loss_cc)

            checkpoint.archive_checkpoint()

            if drive:
                checkpoint.copy_checkpoint()


            #batch loop for validation
            if val:
                # change state of encoder and decoders to .eval
                self.set_eval()

                batches_val = batching_val.build_batches(randomise=False)

                #batch looping for validation
                for batch in batches_val:
                    # call 'get_batch' to actually load the tensors from file
                    features_maps_val, structural_tokens_val, triggers_val, cells_content_tokens_val = batching_val.get_batch(batch)
                    predictions_val, loss_s_val, predictions_cell_val, loss_cc_val, loss_val = val_step(features_maps_val, structural_tokens_val,triggers_val, cells_content_tokens_val, self, LAMBDA )

                    total_loss_s_val+= loss_s_val
                    if loss_cc_val:
                        total_loss_cc_val += loss_cc_val
                total_loss_s_val/=len(batches)
                total_loss_cc_val/=len(batches)
                print("-- structural decoder:---")
                print("Truth")
                print([self.structural_integer2token[p.item()] for p in structural_tokens_val[0]])
                print("Prediction")
                print([self.structural_integer2token[p.item()] for p in predictions_val[0]])
                if abs(LAMBDA-1.0) > 0.01:
                    print("-- cell decoder:---")
                    print("Truth")
                    print([self.cell_content_integer2token[p.item()] for p in cells_content_tokens_val[0][0]])
                    print("Prediction")
                    print([self.cell_content_integer2token[p.item()] for p in predictions_cell_val[0][0]])
            ######################

            t1_stop = perf_counter()
            print("----------------------")
            print('epoch: %d \tLAMBDA: %.2f\tlr:%.5f\ttime: %.2f'%(epoch,LAMBDA, lr, t1_stop-t1_start))
            print('Total loss: %.5f'%total_loss)
            print('Struct. decod. loss: %.5f'%total_loss_s)
            print("Cell dec. loss:", total_loss_cc)

#            print('Validation struct. decod. loss: %.5f'%total_loss_s_val)
#            if loss_cc_val:
#                print('Validation cell decoder. loss: %.5f'%loss_cc_val)
            print('time for 100k examples:' , "%.2f hours"%((t1_stop-t1_start)/number_examples*100000/3600))

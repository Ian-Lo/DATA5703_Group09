import Utils

from EncoderStructural import EncoderStructural
from EncoderCellContent import EncoderCellContent
from DecoderStructural import DecoderStructural
from DecoderCellContent import DecoderCellContent

from BatchingMechanism import BatchingMechanism
from CheckPoint import CheckPoint
from TrainStep import train_step
from ValStep import val_step

import torch
from time import perf_counter, time
import numpy as np

from FixedEncoder import FixedEncoder
import PIL


class Model:
    """Combined class for encoder, structural decoder and cell decoder."""

    def __init__(self,
                 relative_path,
                 model_tag,
                 in_channels=512,
                 out_channels_structural=16,
                 out_channels_cell_content=16,
                 structural_embedding_size=16,
                 structural_hidden_size=256,
                 structural_attention_size=256,
                 cell_content_embedding_size=80,
                 cell_content_hidden_size=512,
                 cell_content_attention_size=256,
                 maxT_structure = 2000):

        # set device
        # sets device for model and PyTorch tensors
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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
        # initialize structural_encoder
        encoder_structural = EncoderStructural(
            in_channels, out_channels_structural)
        encoder_structural_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, encoder_structural.parameters()))
        self.encoder_structural = encoder_structural.to(self.device)
        self.encoder_structural_optimizer = encoder_structural_optimizer

        # initialize cell_encoder
        encoder_cell_content = EncoderCellContent(
            in_channels, out_channels_cell_content)
        encoder_cell_content_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, encoder_cell_content.parameters()))
        self.encoder_cell_content = encoder_cell_content.to(self.device)
        self.encoder_cell_content_optimizer = encoder_cell_content_optimizer

        # set up the decoder for structural tokens
        decoder_structural = DecoderStructural(structural_token2integer, structural_embedding_size,
                                               out_channels_structural, structural_hidden_size, structural_attention_size)
        decoder_structural_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, decoder_structural.parameters()))
        self.decoder_structural = decoder_structural.to(self.device)
        self.decoder_structural_optimizer = decoder_structural_optimizer

        # set up the decoder for cell content tokens
        decoder_cell_content = DecoderCellContent(cell_content_token2integer, cell_content_embedding_size,
                                                  out_channels_cell_content, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size)
        decoder_cell_content_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, decoder_cell_content.parameters()))
        self.decoder_cell_content = decoder_cell_content.to(self.device)
        self.decoder_cell_content_optimizer = decoder_cell_content_optimizer

    def load_checkpoint(self, file_path="checkpoint.pth.tar"):
        loader = CheckPoint.load_checkpoint(file_path)
        self.encoder_structural.load_state_dict(loader['encoder_structural'])
        self.encoder_cell_content.load_state_dict(
            loader['encoder_cell_content'])
        self.decoder_structural.load_state_dict(loader['decoder_structural'])
        self.decoder_cell_content.load_state_dict(
            loader['decoder_cell_content'])
        self.encoder_structural_optimizer.load_state_dict(
            loader['encoder_structural_optimizer'])
        self.encoder_cell_content_optimizer.load_state_dict(
            loader['encoder_cell_content_optimizer'])
        self.decoder_structural_optimizer.load_state_dict(
            loader['decoder_structural_optimizer'])
        self.decoder_cell_content_optimizer.load_state_dict(
            loader['decoder_cell_content_optimizer'])

    def set_eval(self):
        '''Change to evaluation state.'''
        self.decoder_structural = self.decoder_structural.eval()
        self.decoder_cell_content = self.decoder_cell_content.eval()
        self.encoder_structural = self.encoder_structural.eval()
        self.encoder_cell_content = self.encoder_cell_content.eval()

    def set_train(self):
        ''' Change to training state'''
        self.decoder_structural = self.decoder_structural.train()
        self.decoder_cell_content = self.decoder_cell_content.train()
        self.encoder_structural = self.encoder_structural.train()
        self.encoder_cell_content = self.encoder_cell_content.train()

    def predict(self, file_path, maxT=2000):
        ''' Only works for a single example.'''
        self.set_eval()
        # instantiate the fixed CNN encoder
        # with ResNet-18, the features map will be (features_map_size * features_map_size, 512)
        features_map_size = 12
        fixedEncoder = FixedEncoder('ResNet18', features_map_size)

        # open image
        image = PIL.Image.open(file_path)
        # preprocess image
        preprocessed_images = [fixedEncoder.preprocess(image)]
        preprocessed_image = torch.stack(preprocessed_images)

        # run through ResNet-18
        features_map = fixedEncoder.encode(preprocessed_image)
        features_map_float32 = features_map.astype(np.float32)
        features_map_tensor = torch.from_numpy(features_map_float32)

        # permute axes of features map in the same way as during training
        features_map_tensor = features_map_tensor.permute(0, 2, 1)

        # reshape to correct dimensions
        features_map_input = torch.reshape(
            features_map_tensor, (1, 512, features_map_size, features_map_size))

        # pass through encoders
        encoded_structural_features_map = self.encoder_structural.forward(
            features_map_input)
        predictions, storage, pred_triggers, structure_attention_weights = self.decoder_structural.predict(
            encoded_structural_features_map, structural_target=None, store_attention=True, maxT = maxT)
        encoded_cell_content_features_map = self.encoder_cell_content.forward(
            features_map_input)
        predictions_cell, cell_attention_weights = self.decoder_cell_content.predict(
            encoded_cell_content_features_map, storage, cell_content_target=None, store_attention=True)

        predicted_struc_tokens = [
            self.structural_integer2token[p.item()] for p in predictions[0]]

        predicted_cell_tokens = []
        for n, l in enumerate(predictions_cell[0]): #
            predicted_cell_tokens.append([])
            for cell_pred in l:
                predicted_cell_tokens[n].append(self.cell_content_integer2token[cell_pred.item()])

        return predicted_struc_tokens, predicted_cell_tokens, structure_attention_weights, cell_attention_weights

    def train(self,
            gauth=None,
            checkpoint_temp_id=None,
            epochs=1,
            lambdas=[1],
            lrs=[0.001],
            number_examples=100,
            number_examples_val=100,
            batch_size=10,
            batch_size_val=10,
            storage_size=1000,
            val=None,
            maxT_val = 2000,
            alpha_c_struc = 0.0,
            alpha_c_cell_content = 0.0):

        assert epochs == len(lambdas) == len(
            lrs) == len(val), "number of epoch, learning rates, lambdas and val are inconsistent"

        # instantiate the batching object
        batching = BatchingMechanism(
            dataset_split='train', number_examples=number_examples, batch_size=batch_size, storage_size=storage_size)

        # initialise the object
        # here the object works out how many storages and how many examples from every storage are needed
        batching.initialise()

        if val:
            batching_val = BatchingMechanism(
                dataset_split='dev', number_examples=number_examples_val, batch_size=batch_size_val, storage_size=storage_size)
            batching_val.initialise()

        # instantiate checkpoint
        checkpoint = CheckPoint(
            self.model_tag, gauth=gauth, checkpoint_temp_id=checkpoint_temp_id)

        # then reinitialize so we haven't used up batch
        batching.initialise()
        losses_s = []
        losses_s_val = []
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
            for g in self.encoder_structural_optimizer.param_groups:
                g['lr'] = lr
            for g in self.encoder_cell_content_optimizer.param_groups:
                g['lr'] = lr

            # batch looping for training
            for batch in batches:
                # call 'get_batch' to actually load the tensors from file
                features_maps, structural_tokens, triggers, cells_content_tokens = batching.get_batch(
                    batch)
                # test greedy
                structural_tokens = structural_tokens  # [:, 0:50]
                triggers = triggers  # [:,0:50]
#                assert 0
                #####
                # send to training function for forward pass, backpropagation and weight updates
                predictions, loss_s, predictions_cell, loss_cc, loss = train_step(
                    features_maps, structural_tokens, triggers, cells_content_tokens, self, LAMBDA=LAMBDA, alpha_c_struc=alpha_c_struc, alpha_c_cell_content = alpha_c_cell_content)

                # apply logsoftmax
                log_p = torch.nn.LogSoftmax(dim=2)(predictions)

                # greedy decoder to check prediction WITH teacher forcing
                _, predict_id = torch.max(log_p, dim=2)
                if abs(1.0-LAMBDA)>0.001:
                    log_p_cell = torch.nn.LogSoftmax(dim=2)(predictions_cell)
                    _, predict_id_cell = torch.max(log_p_cell, dim=2)

                total_loss_s += loss_s
                total_loss += loss
                if loss_cc:
                    total_loss_cc += loss_cc

#           total_loss_s /= len(batches)
            print("Ground truth, structural:")
            print([self.structural_integer2token[p.item()]
                   for p in structural_tokens[0].detach().numpy()])
            print("Prediction WITH teacher forcing (1 example):")
            print([self.structural_integer2token[p.item()]
                   for p in predict_id[:, 0].detach().numpy()])
            print("Accuracy WITH teacher forcing (1 example):")
            print(np.sum(structural_tokens[0].detach().numpy() == predict_id.detach(
            ).numpy()[:, 0])/structural_tokens[0].detach().numpy().shape[0])
#            for name, param in self.decoder_cell_content.named_parameters():
#                if param.requires_grad:
#                    print(name, param.data)
#            print(self.decoder_cell_content.cell_content_attention.attention_encoded_features_map.bias)

            if val and abs(LAMBDA-1.0)>0.001:
                print("Ground truth, cells:")
                print([self.cell_content_integer2token[p.item()]
                       for p in cells_content_tokens[0][1].detach().numpy()])
                print("Prediction WITH teacher forcing (1 example):")
                print([self.cell_content_integer2token[p.item()]
                       for p in predict_id_cell[:, 1].detach().numpy()])
                print("Accuracy WITH teacher forcing (1 example):")
                print(np.sum(structural_tokens[0].detach().numpy() == predict_id.detach(#
                ).numpy()[:, 0])/structural_tokens[0].detach().numpy().shape[0])


            checkpoint.save_checkpoint(epoch, self.encoder_structural, self.encoder_cell_content, self.decoder_structural, self.decoder_cell_content,
                                       self.encoder_structural_optimizer, self.encoder_cell_content_optimizer, self.decoder_structural_optimizer, self.decoder_cell_content_optimizer, total_loss, total_loss_s, total_loss_cc)

            checkpoint.archive_checkpoint()

            if gauth:
                checkpoint.copy_checkpoint()

            # batch loop for validation
            if val:
                if val[epoch]:
                    with torch.no_grad():
                        # change state of encoders and decoders to .eval
                        self.set_eval()

                        batches_val = batching_val.build_batches(randomise=False)

                        # batch looping for validation
                        for batch in batches_val:
                            # call 'get_batch' to actually load the tensors from file
                            features_maps_val, structural_tokens_val, triggers_val, cells_content_tokens_val = batching_val.get_batch(
                                batch)
                            predictions_val, loss_s_val, predictions_cell_val, loss_cc_val, loss_val = val_step(
                                features_maps_val, structural_tokens_val, triggers_val, cells_content_tokens_val, self, LAMBDA, maxT_val = maxT_val)
                            total_loss_s_val += loss_s_val
                            if loss_cc_val:
                                total_loss_cc_val += loss_cc_val
                        #total_loss_s_val /= len(batches)
                        #total_loss_cc_val /= len(batches)
                        print("-------------Validation loss:---------------")
                        print("-- structural decoder:---")
                        print("Truth (1 example)")
                        print([self.structural_integer2token[p.item()]
                               for p in structural_tokens_val[0]])
                        print("Prediction (1 example)")
                        print([self.structural_integer2token[p.item()]
                               for p in predictions_val[0]])
                        if abs(LAMBDA-1.0) > 0.01:
                            print("-- cell decoder:---")
                            print("Truth")
                            print([self.cell_content_integer2token[p.item()]
                                   for p in cells_content_tokens_val[0][0]])
                            print("Prediction")
                            print([self.cell_content_integer2token[p.item()]
                                   for p in predictions_cell_val[0][0]])
                    ######################
            losses_s.append(total_loss_s)
            losses_s_val.append(total_loss_s_val)
            t1_stop = perf_counter()
            print("----------------------")
            print('epoch: %d \tLAMBDA: %.2f\tlr:%.5f\ttime: %.2f' %
                  (epoch, LAMBDA, lr, t1_stop-t1_start))
            print('Total loss: %.5f' % total_loss)
            print('Struct. decod. loss: %.5f' % total_loss_s)
            print("Cell dec. loss:", total_loss_cc)
            if val:
                print('Validation struct. decod. loss: %.5f'%total_loss_s_val)
#            if loss_cc_val:
#                print('Validation cell decoder. loss: %.5f'%loss_cc_val)
            print('time for 100k examples:', "%.2f hours" %
                  ((t1_stop-t1_start)/number_examples*100000/3600))
        return losses_s, losses_s_val

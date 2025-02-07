from StructuralAttention import StructuralAttention
import torch
import numpy as np




class DecoderStructural(torch.nn.Module):

    def __init__(self, structural_token2integer, embedding_size, encoder_size, structural_hidden_size, structural_attention_size):

        super(DecoderStructural, self).__init__()

        # the attention mechanism
        self.structural_attention = StructuralAttention(encoder_size, structural_hidden_size, structural_attention_size)

        # the vocabulary embedding
        self.structural_token2integer = structural_token2integer
        self.vocabulary_size = len(structural_token2integer)
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(self.vocabulary_size, self.embedding_size)

        # the GRU
        self.input_size = encoder_size + embedding_size
        self.hidden_size = structural_hidden_size
        self.gru = torch.nn.GRU(self.input_size, self.hidden_size)

        # linear transformation to get vocabulary scores
        self.fc = torch.nn.Linear(self.hidden_size, self.vocabulary_size)

        # the loss criterion
        self.loss_criterion = torch.nn.CrossEntropyLoss()

        # softmax function for inference
        self.LogSoftmax = torch.nn.LogSoftmax(dim=1)

        # the device we are running on
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def initialise(self, batch_size):

        # initialise structural input

        init_structural_input = torch.from_numpy(np.repeat(self.structural_token2integer['<start>'], batch_size))
        init_structural_input = init_structural_input.type(torch.int64).to(self.device)

        # initialise structural hidden state
        init_structural_hidden_state = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        init_structural_hidden_state = torch.from_numpy(init_structural_hidden_state)
        init_structural_hidden_state = init_structural_hidden_state.unsqueeze(0).to(self.device)

        return init_structural_input, init_structural_hidden_state

    def timestep(self, encoded_features_map, structural_input, structural_hidden_state):

        # compute the context vector
        # context_vector: (batch_size, encoder_size)
        context_vector, attention_weights = self.structural_attention.forward(encoded_features_map, structural_hidden_state)
        # embed the input
        # embedded_structural_input: (batch size, embedding_size)
        embedded_structural_input = self.embedding(structural_input)

        # concatenate
        concatenated_input = torch.cat([context_vector, embedded_structural_input], 1)
        # the GRU requires an initial extra dimensions (num GRU layers = 1)
        # concatenated_input: (1, batch_size, encoder_size + embedding_size)
        concatenated_input = concatenated_input.unsqueeze(0)

        # perform GRU step
        # hidden_state: (1, batch_size, hidden_size)
        output, structural_hidden_state = self.gru(concatenated_input, structural_hidden_state)
        # undo the unsqueeze
        # output: (batch_size, hidden_size)
        output = torch.squeeze(output, 0)

        # linearly transform to get vocabulary scores
        # output: (batch_size, vocabulary_size)
        prediction = self.fc(output)

        return prediction, structural_hidden_state, attention_weights

    def forward(self, encoded_features_map, structural_target, alpha_c_struc = 0.05):

        batch_size = encoded_features_map.shape[0]
        first_nonzero = (structural_target == 0).sum(dim=1)

        # find lengths without padding
        caption_lengths = structural_target.shape[1]*torch.ones(batch_size).long()-first_nonzero
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending = True)
        # sort
        encoded_features_map = encoded_features_map[sort_ind]
        structural_target = structural_target[sort_ind]

        decode_lengths = (caption_lengths).tolist()

        # prepare to collect all predictions
        num_timesteps = structural_target.size()[-1]

#        batch_size = encoded_features_map.shape[0]
        predictions = torch.zeros((num_timesteps, batch_size, self.vocabulary_size), dtype=torch.float).to(self.device)
#        predictions = np.zeros((num_timesteps, batch_size, self.vocabulary_size), dtype=np.float32)
#        predictions = torch.from_numpy(predictions).to(self.device)

        # TODO: implement more efficiently
        storage = torch.zeros((num_timesteps, 1, batch_size, self.hidden_size) , dtype = torch.float).to(self.device)
#        storage = np.zeros((num_timesteps, 1, batch_size, self.hidden_size), dtype=np.float32)
#        storage = torch.from_numpy(storage).to(self.device)

        # define the size of feature map
        feature_sizes = encoded_features_map.shape[1]

        # initialisation
        structural_input, structural_hidden_state = self.initialise(batch_size)
        attention_weights = torch.zeros(num_timesteps, batch_size, feature_sizes).to(self.device)


        loss = 0

        # run the timesteps
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths]) #

            prediction, structural_hidden_state, attention_weight = self.timestep(encoded_features_map[:batch_size_t], structural_input[:batch_size_t], structural_hidden_state[:, :batch_size_t, :])

            # stores the predictions
            predictions[t, :batch_size_t, :] = prediction

            # stores the attention weights
            attention_weights[t,:batch_size_t,:] = attention_weight

            # teacher forcing
            structural_input = structural_target[:batch_size_t, t]

            loss += self.loss_criterion(prediction, structural_input)/batch_size_t


            # TODO: implement more efficiently
            storage[t, :, :batch_size_t] = structural_hidden_state


        # reorder back to original positions
        storage = storage[:,:, sort_ind]
        predictions = predictions[:, sort_ind, :]

        # normalize by the number of timesteps and examples to allow comparison
        loss = loss/num_timesteps

        regularisation_term = alpha_c_struc * torch.mean(((1 - attention_weights.sum(dim=0)) ** 2))
        loss += regularisation_term


        return predictions, loss, storage

    def calc_loss_struc(self,targets, prediction_probs):
        """ Calculate loss for predictions from targets.
        targets: tensor of shape num_examples, max_length. Contains "true" token indices.
        prediction_probs: list of lists of tensors. Outer list: num_examples. Inner list:
            triggers. Tensors of shape num_tokens.
        """

        batch_size = targets.size()[0]

        loss =0

        for example_idx in range(batch_size):
            # get all target tokens
            target = targets[example_idx]

            # remove trailing zeros (='<pad>') to only get the actual tokens
            unpadded_target = target[target.nonzero()].squeeze(1)
            unpadded_target_size = unpadded_target.size()[0]

            # stack all probability predictions
            prediction_prob = torch.stack(prediction_probs[example_idx])
            num_predictions, num_probs = prediction_prob.size()

            # if unpadded_target_size < num_predictions:

            # pad the target tokens to reach the the number of predictions
            padded_target = torch.zeros(num_predictions, dtype=torch.int64)
            padded_target[0:min(unpadded_target_size, num_predictions)] = unpadded_target[0:min(unpadded_target_size, num_predictions)]

            # the tensors have compatible lengths
            compatible_target = padded_target
            compatible_prediction_prob = prediction_prob

            # else:
            #     # pad the probability predictions to reach the number of target tokens
            #     padded_prediction_prob = torch.zeros((unpadded_target_size, num_probs))
            #     padded_prediction_prob[0:num_predictions, :] = prediction_prob
            #
            #     compatible_target = unpadded_target
            #     compatible_prediction_prob = padded_prediction_prob
            #
#                print('pad prob', compatible_target.size(), compatible_prediction_prob.size())

            loss+= self.loss_criterion(compatible_prediction_prob, compatible_target)/unpadded_target_size

        return loss/batch_size

    def predict(self, encoded_features_map, structural_target = None, maxT = 2000, store_attention=False):
        ''' For use on validation set and test set.
        encoded_features_map: torch.tensor of shape (num_examples, encoder_size, encoder_size)
        structural target: None or tensor of shape (timesteps, batch_size)
            Targets for prediction. If None: loss function is not calculated
            and returned.
        maxT: Integer. The maximum number of timesteps that is attempted to
            find <end> if structural target is not supplied.
        store_attention: Boolean. Flag to activate the storage of attention weights.
        '''

        batch_size = encoded_features_map.shape[0]

        # create list to hold predictions since we sometimes don't know the size
        predictions = [ [] for n in range(batch_size)]
        prediction_props = [ [] for n in range(batch_size) ]

        # create list for timesteps when td tokens are called:
        pred_triggers = [ [] for n in range(batch_size)]

        # create list to store hidden state
        storage = [ [] for n in range(batch_size)]

        # create a list to store the attention weights
        attention_storage = [ [] for n in range(batch_size)]

        # initialisation
        predict_id, structural_hidden_state = self.initialise(batch_size)

        loss = 0

        # define tensor to contain batch indices to run through timestep.
        batch_indices_to_keep = torch.tensor(range(batch_size), dtype = torch.long)

        #indices to keep within for loop
        indices_to_keep = torch.tensor(range(batch_size), dtype = torch.long)
#        print("outside")
#        print(structural_input.shape)
        #run the timesteps
        for t in range(maxT):

            # slice out only those in continue_decoder
            encoded_features_map_in = encoded_features_map[batch_indices_to_keep,:,:]
            structural_input_in = predict_id[indices_to_keep]

            structural_hidden_state_in = structural_hidden_state[:, indices_to_keep, :]
            # run through rnn
            prediction, structural_hidden_state, attention_weights = self.timestep(encoded_features_map_in, structural_input_in, structural_hidden_state_in)

            # apply logsoftmax
            log_p = self.LogSoftmax(prediction)

            # greedy decoder:
            _, predict_id = torch.max(log_p, dim = 1 )

#            print("inside")#
#            print(structural_input.shape)


            # list to contain indices to remove from batch_indices_to_keep
            removes = []
            keeps = []

            # loop through predictions
            for n, id in enumerate(predict_id):
                batch_index = batch_indices_to_keep[n]

                # if stop:
                if id in [self.structural_token2integer["<end>"], self.structural_token2integer["<pad>"] ]:
                    #store to later remove element from continue_decoder
                    removes.append(batch_index)
                    # do not save prediction or hidden state
                    predictions[batch_index].append(id)
                    prediction_props[batch_index].append(prediction[n,:])

                    if store_attention:
                        attention_storage[batch_index].append(attention_weights[n, :])

                    continue
                #if not stop
                else:
                    keeps.append(n)
                    # get correct index
#                    print("index", "n")
#                    print(index, n)
                    # save prediction
                    predictions[batch_index].append(id)
                    prediction_props[batch_index].append(prediction[n,:])

                    if store_attention:
                        attention_storage[batch_index].append(attention_weights[n, :])

                # if <td> or >:
                if id in [self.structural_token2integer["<td>"], self.structural_token2integer[">"]]:
                    # keep hidden state
                    storage[batch_index].append(structural_hidden_state[:,n, :])
                    pred_triggers[batch_index].append(t)

            # break condition for inference
            if len(keeps)==0:
                break

            #update indices to keep track
            indices_to_keep = torch.tensor(keeps, dtype =torch.long)
            for element in removes[::-1]:
                batch_indices_to_keep = batch_indices_to_keep[batch_indices_to_keep!=element]

        if structural_target is not None:
            loss = self.calc_loss_struc(structural_target, prediction_props )
            return predictions, loss, storage, pred_triggers

        elif store_attention:
            return predictions, storage, pred_triggers, attention_storage

        else:
            return predictions, storage, pred_triggers

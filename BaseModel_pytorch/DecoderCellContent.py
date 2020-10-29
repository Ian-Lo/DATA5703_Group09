from CellContentAttention import CellContentlAttention
import torch
import numpy as np


class DecoderCellContent(torch.nn.Module):

    def __init__(self, cell_content_token2integer, embedding_size, encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size):

        super(DecoderCellContent, self).__init__()

        # the attention mechanism
        self.cell_content_attention = CellContentlAttention(encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size)

        # the vocabulary embedding
        self.cell_content_token2integer = cell_content_token2integer
        self.vocabulary_size = len(cell_content_token2integer)
        self.embedding_size = embedding_size
        self.embedding = torch.nn.Embedding(self.vocabulary_size, self.embedding_size)

        # the GRU
        self.input_size = encoder_size + embedding_size
        self.hidden_size = cell_content_hidden_size
        self.gru = torch.nn.GRU(self.input_size, self.hidden_size)

        # linear transformation to get vocabulary scores
        self.fc = torch.nn.Linear(self.hidden_size, self.vocabulary_size)

        # the loss criterion
        self.loss_criterion = torch.nn.CrossEntropyLoss()

    def initialise(self, batch_size):

        # initialise structural input
        init_cell_content_input = np.repeat(self.cell_content_token2integer['<start>'], batch_size).astype(np.int64)
        init_cell_content_input = torch.from_numpy(init_cell_content_input)

        # initialise structural hidden state
        init_cell_content_hidden_state = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        init_cell_content_hidden_state = torch.from_numpy(init_cell_content_hidden_state)
        init_cell_content_hidden_state = init_cell_content_hidden_state.unsqueeze(0)

        return init_cell_content_input, init_cell_content_hidden_state

    def timestep(self, encoded_features_map, structural_hidden_state, cell_content_input, cell_content_hidden_state):

        # compute the context vector
        # context_vector: (batch_size, encoder_size)
        context_vector = self.cell_content_attention.forward(encoded_features_map, structural_hidden_state, cell_content_hidden_state)

        # embed the input
        # embedded_structural_input: (batch size, embedding_size)
        embedded_structural_input = self.embedding(cell_content_input)

        # concatenate
        concatenated_input = torch.cat([context_vector, embedded_structural_input], 1)
        # the GRU requires an initial extra dimensions (num GRU layers = 1)
        # concatenated_input: (1, batch_size, encoder_size + embedding_size)
        concatenated_input = concatenated_input.unsqueeze(0)

        # perform GRU step
        # hidden_state: (1, batch_size, hidden_size)
        output, cell_content_hidden_state = self.gru(concatenated_input, cell_content_hidden_state)
        # undo the unsqueeze
        # output: (batch_size, hidden_size)
        output = torch.squeeze(output, 0)

        # linearly transform to get vocabulary scores
        # output: (batch_size, vocabulary_size)
        prediction = self.fc(output)

        return prediction, cell_content_hidden_state

    def forward(self, encoded_features_map, structural_hidden_state, cell_content_target):

        # prepare to collect all predictions
        num_timesteps = cell_content_target.size()[-1]
        batch_size = encoded_features_map.shape[0]
        predictions = np.zeros((num_timesteps, batch_size, self.vocabulary_size), dtype=np.float32)
        predictions = torch.from_numpy(predictions)

        # initialisation
        cell_content_input, cell_content_hidden_state = self.initialise(batch_size)

        loss = 0

        # run the timesteps
        for t in range(0, num_timesteps):
            # Luca: shouldn't this be cell_content_hidden_state instead of init_cell_content_hidden_state
            prediction, init_cell_content_hidden_state = self.timestep(encoded_features_map, structural_hidden_state, cell_content_input, cell_content_hidden_state)

            # stores the predictions
            predictions[t] = prediction

            # teacher forcing
            cell_content_input = cell_content_target[:, t]

            # compute loss
            loss += self.loss_criterion(prediction, cell_content_input)

        # normalize
        loss = loss/num_timesteps/batch_size

        return predictions, loss

    def calc_loss_cell(self,targets, prediction_probs):
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

            if unpadded_target_size < num_predictions:

                # pad the target tokens to reach the the number of predictions
                padded_target = torch.zeros(num_predictions, dtype=torch.int64)
                padded_target[0:unpadded_target_size] = unpadded_target

                # the tensors have compatible lengths
                compatible_target = padded_target
                compatible_prediction_prob = prediction_prob

            else:

                # pad the probability predictions to reach the number of target tokens
                padded_prediction_prob = torch.zeros((unpadded_target_size, num_probs))
                padded_prediction_prob[0:num_predictions, :] = prediction_prob

                compatible_target = unpadded_target
                compatible_prediction_prob = padded_prediction_prob

#                print('pad prob', compatible_target.size(), compatible_prediction_prob.size())

            loss+= self.loss_criterion(compatible_prediction_prob, compatible_target)/num_predictions
        return loss

    def predict(self, encoded_features_map, structural_hidden_state, cell_content_target=None, maxT = 2000):
        ''' For use on validation set and test set.
        encoded_features_map: tensor of shape (num_examples,encoder_size,encoder_size)
        structural_hidden_state: list of list of tensors (num_examples, num_struc_token, hidden_dim)
        cell_content_target: tensor of shape (num_examples,  max_struc_token_pred , max_cell_tokens_true)
        maxT: integer, maximum number of time steps
        '''

        print("encoded_features_map")
        print(encoded_features_map.shape)
        print("structural_hidden_state")
        print(len(structural_hidden_state))
        print("cell_content_target")
        print(cell_content_target.shape)

        batch_size =encoded_features_map.shape[0] #number of triggers for each example. Will vary from image to image

        # create list to hold predictions since we sometimes don't know the size
        predictions = [ [ []  ] for n in range(batch_size)  ]
        prediction_propbs = [ [ [] ] for n in range(batch_size) ]


        loss = 0

        # loop over images/example
        for batch_index in range(batch_size):

            # define tensor to contain outer indices to run through timestep.
            outer_indices_to_keep = torch.tensor([n for n in range(len(structural_hidden_state[batch_index])) if len(structural_hidden_state[batch_index])!=0] , dtype = torch.long)

            # indices to keep within for loop
            indices_to_keep = torch.tensor(range(batch_size), dtype = torch.long)

            # initialisation
            cell_content_input, cell_content_hidden_state = self.initialise(outer_indices_to_keep.shape[0])

            encoded_features_map_example = encoded_features_map[batch_index].repeat(outer_indices_to_keep.shape[0],1,1 )
            strucural_hidden_state_example = torch.stack(structural_hidden_state[batch_index])

            # loop over timesteps
            for t in range(max_T):

                # keep only those examples that have not predicted and <end> token
                encoded_features_map_in = encoded_features_map_example[outer_indices_to_keep, :, :]
                structural_hidden_states_in = structural_hidden_state_example[outer_indices_to_keep]

                # keep only those examples that have not predicted and <end> token
                cell_content_input_in = cell_content_input[indices_to_keep]
                cell_content_hidden_state_in = cell_content_hidden_state[:, indices_to_keep, :]

                # run through rnn
                prediction, cell_content_hidden_state = self.timestep(encoded_features_map_in, structural_hidden_state_in, cell_content_input_in, cell_content_hidden_state_in)

                # apply logsoftmax
                log_p = self.LogSoftmax(prediction)

                # greedy decoder:
                _, predict_id = torch.max(log_p, dim = 1 )

                removes = []
                keeps = []

                # loop through predictions
                for n, id in enumerate(predict_id):
                    outer_index = outer_indices_to_keep[n]

                    # if stop:
                    if id in [self.cell_content_token2integer["<end>"],self.cell_content_token2integer["<pad>"]]:
                        #remove element from outer_index_to_keep
                        removes.append(outer_index)
                        # do not save prediction or hidden state
                        predictions[batch_index][outer_index].append(id)
                        prediction_props[batch_index][outer_index].append(prediction[n,:])
                        continue
                    #if not stop
                    else:
                        keeps.append(n)
                        # get correct index
                        predictions[batch_index][outer_index].append(id)
                        prediction_props[batch_index][outer_index].append(prediction[n,:])

                if len(keeps)==0:
                    break

                #update indices to keep track
                indices_to_keep = torch.tensor(keeps, dtype =torch.long)
                for element in removes[::-1]:
                    outer_indices_to_keep = batch_indices_to_keep[outer_indices_to_keep!=element]

        if cell_content_target  is not None:
            loss_batch = 0
            for batch_index in range(batch_size):
                loss_batch += self.calc_loss_cell(cell_content_target[batch_index], prediction_props[batch_index] )
            return predictions, loss, storage, pred_triggers

        else:
            return predictions

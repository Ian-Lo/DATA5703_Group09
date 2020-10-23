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

    def predict(self, encoded_features_map, structural_hidden_state, cell_content_target=None, maxT = 500):

        ''' For use on validation set and test set.
        encoded_features_map: tensor of shape (num_examples,encoder_size,encoder_size)
        structural_hidden_state: list of list of tensors (num_examples, num_struc_token, hidden_dim)
        cell_content_target:
        maxT: integer, maximum number of time steps
        '''

        print("encoded_features_map")
        print(encoded_features_map.shape)
        print("structural_hidden_state")
        print(len(structural_hidden_state))
        print("cell_content_target")
        print(cell_content_target.shape)

        batch_size = encoded_features_map.shape[0]

        # create list to hold predictions since we sometimes don't know the size
        predictions = [ [] for n in range(batch_size)]
        prediction_propbs = [ [] for n in range(batch_size) ]

        # initialisation
        cell_content_input, cell_content_hidden_state = self.initialise(batch_size)

        loss = 0


        # define tensor to contain batch indices to run through timestep.

        batch_indices_to_keep = torch.tensor([n for n in range(batch_size) if len(structural_hidden_state[n])!=0] , dtype = torch.long)

        # indices to keep within for loop
        indices_to_keep = torch.tensor(range(batch_size), dtype = torch.long)

        #update indices to remove empty lists
        preremove = [n for n in range(batch_size) if len(structural_hidden_state)==0]
        for element in preremove[::-1]:
            batch_indices_to_keep = batch_indices_to_keep[batch_indices_to_keep!=element]


        # run the timesteps
        for t in range(maxT):
            # break condition for inference
            if batch_indices_to_keep.numel()==0:
                break

            # slice out only elements that are required
            encoded_features_map_in = encoded_features_map[batch_indices_to_keep, :, :]
            cell_content_input_in = cell_content_input[batch_indices_to_keep]
            # get hidden state
            hidden_states_t =torch.stack([ h[t] for n, h in enumerate(structural_hidden_state) if n in batch_indices_to_keep])
            hidden_states_t.reshape(batch_indices_to_keep.numel(), -1, )
            cell_content_hidden_state_in = cell_content_hidden_state[:, continue_decoder, :]

            # Anders: figure out how to collape the hidden states of this timestep
            # so it can be processed through rnn.
            print("structural_hidden_state")
            print(structural_hidden_state[t])
            structural_hidden_state_in = structural_hidden_state[:, continue_decoder, :]

            # run through rnn
            prediction, cell_content_hidden_state = self.timestep(encoded_features_map_in, structural_hidden_state_in, cell_content_input_in, cell_content_hidden_state_in)

            # apply logsoftmax
            log_p = self.LogSoftmax(prediction)

            # greedy decoder:
            _, predict_id = torch.max(log_p, dim = 1 )

            # calculate loss when possible
            if cell_content_target is not None:
                # what if length different?
                truth = structural_target[continue_decoder, t]
                loss += self.loss_criterion(prediction, truth)/continue_decoder.shape[0] # normalize

            # loop through predictions
            for n, id in enumerate(predict_id):
                # if stop:
                if id in [self.cell_content_token2integer["<end>"]]:
                    #remove element from continue_decoder
                    continue_decoder = continue_decoder[continue_decoder!=n]
                    # do not save prediction or hidden state
                    continue
                #if not stop
                else:
                    # get correct index
                    index = continue_decoder[n]
                    # save prediction
                    predictions[index].append(id)
                    prediction_propbs[index].append(prediction[n,:])

        if 0:  # we need to get this way of calculating loss to work.
            if structural_target is not None:
                # this is where the new calculation of loss goes
                collapsed_predictions = [ torch.stack(l) for l in prediction_propbs ]
                padded_prediction_probs = torch.nn.utils.rnn.pad_sequence(collapsed_predictions, batch_first=True, padding_value=0)
                # insert Luca's function to calculate loss
    #            loss = XXXXX
                loss = loss/t

        if cell_content_target is not None:
            loss = self.calc_loss_struc(cell_content_target, prediction_props )
            return predictions, loss

        else:
            return predictions

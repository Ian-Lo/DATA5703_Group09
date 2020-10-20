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

    def initialise(self, batch_size):

        # initialise structural input
        init_structural_input = torch.from_numpy(np.repeat(self.structural_token2integer['<start>'], batch_size))

        # initialise structural hidden state
        init_structural_hidden_state = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        init_structural_hidden_state = torch.from_numpy(init_structural_hidden_state)
        init_structural_hidden_state = init_structural_hidden_state.unsqueeze(0)

        return init_structural_input, init_structural_hidden_state

    def timestep(self, encoded_features_map, structural_input, structural_hidden_state):

        # compute the context vector
        # context_vector: (batch_size, encoder_size)
        context_vector = self.structural_attention.forward(encoded_features_map, structural_hidden_state)

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

        return prediction, structural_hidden_state

    def forward(self, encoded_features_map, structural_target):

        # prepare to collect all predictions
        num_timesteps = structural_target.size()[-1]
        batch_size = encoded_features_map.shape[0]
        predictions = np.zeros((num_timesteps, batch_size, self.vocabulary_size), dtype=np.float32)
        predictions = torch.from_numpy(predictions)

        # TODO: implement more efficiently
        storage = np.zeros((num_timesteps, 1, batch_size, self.hidden_size), dtype=np.float32)
        storage = torch.from_numpy(storage)

        # initialisation
        structural_input, structural_hidden_state = self.initialise(batch_size)

        loss = 0

        # run the timesteps
        for t in range(0, num_timesteps):

            prediction, structural_hidden_state = self.timestep(encoded_features_map, structural_input, structural_hidden_state)

            # stores the predictions
            predictions[t] = prediction

            # teacher forcing
            structural_input = structural_target[:, t]

            loss += self.loss_criterion(prediction, structural_input)

            # TODO: implement more efficiently
            storage[t] = structural_hidden_state

        # normalize by the number of timesteps and examples to allow comparison
        loss = loss/num_timesteps/batch_size

        return predictions, loss, storage

    def predict(self, encoded_features_map, structural_target = None, maxT = 1000):
        ''' For use on validation set and test set.
        structural target: None or tensor of shape (timesteps, batch_size)
            Targets for prediction. If None: loss function is not calculated
            and returned.
        maxT: Integer. The maximum number of timesteps that is attempted to
            find <end> if structural target is not supplied.  '''

        batch_size = encoded_features_map.shape[0]

        # create list to hold predictions since we sometimes don't know the size
        predictions = [ [] for n in range(batch_size)]
        prediction_propbs = [ [] for n in range(batch_size) ]
        # create list to td tokens:

        pred_triggers = [ [] for n in range(batch_size)]

        # create list to store hidden state
        storage = [ [] for n in range(batch_size)]

        # initialisation
        structural_input, structural_hidden_state = self.initialise(batch_size)

        loss = 0

        # set maximum number of structural tokens
        if structural_target is not None:
            maxT = structural_target.shape[1]
        else:
            maxT = 1000

        # define tensor to contain batch indices run through timestep.
        continue_decoder = torch.tensor(range(batch_size))

        for t in range(maxT):

            # slice out only those in continue_decoder
            encoded_features_map_in = encoded_features_map[continue_decoder,:,:]
            structural_input_in = structural_input[continue_decoder]
            structural_hidden_state_in = structural_hidden_state[:, continue_decoder, :]

            # run through rnn
            prediction, structural_hidden_state = self.timestep(encoded_features_map_in, structural_input_in, structural_hidden_state_in)

            # apply logsoftmax
            log_p = self.LogSoftmax(prediction)

            # greedy decoder:
            _, predict_id = torch.max(log_p, dim = 1 )

            # calculate loss when possible
            if structural_target is not None:
                # what if length of different?
                truth = structural_target[continue_decoder, t]
                loss += self.loss_criterion(prediction, truth)/continue_decoder.shape[0] # normalize

            # loop through predictions
            for n, id in enumerate(predict_id):
                # if stop:
                if id in [self.structural_token2integer["<end>"]]:
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
                # if <td> or >:
                if id in [self.structural_token2integer["<td>"], self.structural_token2integer[">"]]:
                    # keep hidden state
                    storage[index].append(structural_hidden_state[:,n, :])
                    pred_triggers[index].append(t)


        if structural_target is not None:
            collapsed_predictions = [ torch.stack(l) for l in prediction_propbs ]
            padded_prediction_probs = torch.nn.utils.rnn.pad_sequence(collapsed_predictions, batch_first=True, padding_value=0)
            # insert Luca's function to calculate loss
            loss = loss/t

            return predictions, loss, storage, pred_triggers
        else:
            return predictions, storage, pred_triggers

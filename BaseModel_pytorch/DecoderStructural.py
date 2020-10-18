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
        self.softmax = torch.nn.Softmax(dim=1)

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

        return predictions, loss, storage

    def predict(self, encoded_features_map, structural_target = False):
            ### make list of lists and append to lists

            batch_size = encoded_features_map.shape[0]

            num_timesteps = structural_target.size()[-1]
            predictions = np.zeros((num_timesteps, batch_size, self.vocabulary_size), dtype=np.float32)
            predictions = torch.from_numpy(predictions)

            # TODO: implement more efficiently
            storage = np.zeros((num_timesteps, 1, batch_size, self.hidden_size), dtype=np.float32)
            storage = torch.from_numpy(storage)

            # initialisation
            structural_input, structural_hidden_state = self.initialise(batch_size)

            loss = 0

            # run the timesteps
            for t in range(1000):

#            for t in range(0, num_timesteps):

                prediction, structural_hidden_state = self.timestep(encoded_features_map, structural_input, structural_hidden_state)

                # apply softmax
                output = self.softmax(prediction)

                # greedy decoder

                #self.softmax = nn.LogSoftmax(dim=1)

                # stores the predictions
                predictions[t] = prediction

                # teacher forcing
                structural_input = structural_target[:, t]
                loss += self.loss_criterion(prediction, structural_input)
                print("teacher")

                # TODO: implement more efficiently
                storage[t] = structural_hidden_state

            return predictions, loss, storage

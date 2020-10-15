import torch


class Encoder(torch.nn.Module):

    def __init__(self, features_map_size, encoder_size):

        super(Encoder, self).__init__()

        self.fc = torch.nn.Linear(features_map_size, encoder_size)
        self.relu = torch.nn.ReLU()

    def forward(self, features_map):

        # encoded_features_map: (batch_size, n*n, encoder_size)
        encoded_features_map = self.relu(self.fc(features_map))

        return encoded_features_map


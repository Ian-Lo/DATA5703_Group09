import torch


class CellContentlAttention(torch.nn.Module):

    def __init__(self, encoder_size, structural_hidden_size, cell_content_hidden_size, cell_content_attention_size):

        super(CellContentlAttention, self).__init__()

        # attention layers
        self.attention_encoded_features_map = torch.nn.Linear(encoder_size, cell_content_attention_size)
        self.attention_structural_hidden_state = torch.nn.Linear(structural_hidden_size, cell_content_attention_size)
        self.attention_cell_content_hidden_state = torch.nn.Linear(cell_content_hidden_size, cell_content_attention_size)
        self.attention_combined = torch.nn.Linear(cell_content_attention_size, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, encoded_features_map, structural_hidden_state, cell_content_hidden_state):

        # compute attention on encoded features map
        # attention_encoded_features_map: (batch_size, n*n, cell_content_attention_size)
        attention_encoded_features_map = self.attention_encoded_features_map(encoded_features_map)

        # compute attention on structural hidden state
        # we remove the first dimension that is only needed by the GRU
        structural_hidden_state = torch.squeeze(structural_hidden_state, 0)
        attention_structural_hidden_state = self.attention_structural_hidden_state(structural_hidden_state)
        # we add an extra dimension so to be able to sum up the three tensor
        # attention_structural_hidden: (batch_size, 1, cell_content_attention_size)
        attention_structural_hidden_state = attention_structural_hidden_state.unsqueeze(1)

        # compute attention on cell content hidden state
        # we remove the first dimension that is only needed by the GRU
        cell_content_hidden_state = torch.squeeze(cell_content_hidden_state, 0)
        attention_cell_content_hidden_state = self.attention_cell_content_hidden_state(cell_content_hidden_state)
        # we add an extra dimension so to be able to sum up the three tensor
        # attention_cell_content_hidden: (batch_size, 1, cell_content_attention_size)
        attention_cell_content_hidden_state = attention_cell_content_hidden_state.unsqueeze(1)

        # combine the attentions
        attention_combined = self.attention_combined(self.relu(attention_encoded_features_map + attention_structural_hidden_state + attention_cell_content_hidden_state))
        # we remove last dimension
        # attention_combined: (batch_size, n*n)
        attention_combined = torch.squeeze(attention_combined, 2)

        # compute the attention weights
        attention_weights = self.softmax(attention_combined)
        # we restore last dimension
        # attention_weights: (batch_size, n*n, 1)
        attention_weights = attention_weights.unsqueeze(2)

        # context_vector: (batch_size, encoder_size)
        context_vector = (encoded_features_map * attention_weights).sum(dim=1)

        return context_vector
    
import torch


class EncoderStructural(torch.nn.Module):

    def __init__(self, in_channels, out_channels_structural, conv = False):
        """conv: use convolutional layer, else linear layer """

        super(EncoderStructural, self).__init__()

        self.conv = conv
        if self.conv:
            self.conv1x1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels_structural, kernel_size=1)
        else:
            self.fc = torch.nn.Linear(in_channels, out_channels_structural)


    def forward(self, features_map):

        if self.conv:
            # reducing the number of channels
            features_map = self.conv1x1(features_map)
            features_map = features_map.permute(0, 2, 3, 1)
        else:
        # swap axes
            features_map = features_map.permute(0, 2, 3, 1)
            features_map = self.fc(features_map)

        # stacking the layers vertically
        dims = features_map.size()
        num_examples = dims[0]
        num_layers = dims[1]
        layer_dim1 = dims[2]
        layer_dim2 = dims[3]
        structural_features_map = torch.reshape(features_map, (num_examples, num_layers * layer_dim1, layer_dim2))

        return structural_features_map

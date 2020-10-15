import Utils
from Encoder import Encoder
from DecoderStructural import DecoderStructural
import h5py
import torch
import numpy as np

structural_token2integer, structural_integer2token = Utils.load_structural_vocabularies()

suffix = '00'

storage_features_maps_path = Utils.create_abs_path('Dataset/features_maps_' + suffix + '.hdf5')
storage_features_maps = h5py.File(storage_features_maps_path, "r")

storage_structural_tokens_path = Utils.create_abs_path('Dataset/structural_tokens_' + suffix + '.hdf5')
storage_structural_tokens = h5py.File(storage_structural_tokens_path, "r")

features_maps = storage_features_maps['data']
structural_tokens = storage_structural_tokens['data']

# batch of examples
features_map = features_maps[0:10].astype(np.float32)
features_map = torch.from_numpy(features_map)

structural_tokens = structural_tokens[0:10].astype(np.int64)
structural_tokens = torch.from_numpy(structural_tokens)

print('fm', features_map.size(), features_map.dtype)
print('st', structural_tokens.shape, structural_tokens.dtype)

###---###

# set up the encoder for the features maps
features_map_size = features_map.size()[-1]
encoder_size = 200

encoder = Encoder(features_map_size, encoder_size)

# encode
encoded_features_map = encoder.forward(features_map)
print('e_fm', encoded_features_map.size(), encoded_features_map.dtype)

###---###

# set up decoder
embedding_size = 15
encoder_size = encoded_features_map.shape[-1]
structural_hidden_size = 100
structural_attention_size = 50

decoder_structural = DecoderStructural(structural_token2integer, embedding_size, encoder_size, structural_hidden_size, structural_attention_size)

# run timestep

predictions, loss = decoder_structural.forward(encoded_features_map, structural_tokens)

print(predictions.size(), predictions.dtype)
print(structural_tokens.size(), structural_tokens.dtype)
print(loss)

storage_features_maps.close()
storage_structural_tokens.close()
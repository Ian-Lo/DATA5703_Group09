# the relative path of the folder containing the dataset
relative_path = "../Dataset/"

# model_tag is the name of the folder that the checkpoints folders will be saved in
model_tag = "baseline_min_struc"

# tunable parameters
out_channels = 128 # number of channels
structural_hidden_size = 128 # dimensions of hidden layer in structural decoder
structural_attention_size = 128 # dimensions of context vector in structural decoder
cell_content_hidden_size = 256 # dimensions of hidden layer in cell decoder
cell_content_attention_size = 128 # dimensions of ontext vector in structural decoder

# fixed parameters
in_channels = 512 # fixed in output from resnet, do not change
encoder_size = out_channels # fixed in output from resnet, do not change
structural_embedding_size = 16 # determined from preprocessing, do not change
cell_content_embedding_size = 80 # determined from preprocessing, do not change

# import model
from Model import Model

# instantiate model
model = Model(relative_path, model_tag, in_channels = in_channels, out_channels = out_channels, encoder_size = encoder_size, structural_embedding_size=structural_embedding_size, structural_hidden_size=structural_hidden_size, structural_attention_size=structural_attention_size, cell_content_embedding_size=cell_content_embedding_size, cell_content_hidden_size=cell_content_hidden_size, cell_content_attention_size=cell_content_attention_size)

# reload latest checkpoint
model.load_checkpoint("../../checkpoint.pth.tar")

image_path = '/Users/andersborges/Documents/Capstone/code/table.png'

model.predict(image_path)

from Model import Model

model_tag = "matrix_plot"
file1_path = "/Users/andersborges/Documents/Capstone/PMC3681597_004_00.png"
relative_path = "../../Dataset"

# tunable parameters
out_channels_structural = 64 # number of channels
out_channels_cell_content = 64 # number of channels
structural_hidden_size = 128 # dimensions of hidden layer in structural decoder
structural_attention_size = 128 # dimensions of context vector in structural decoder
cell_content_hidden_size = 256 # dimensions of hidden layer in cell decoder
cell_content_attention_size = 128 # dimensions of ontext vector in structural decoder

# fixed parameters
in_channels = 512 # fixed in output from resnet, do not change
structural_embedding_size = 16 # determined from preprocessing, do not change
cell_content_embedding_size = 80 # determined from preprocessing, do not change

# import model
from Model import Model

# instantiate model
model = Model(relative_path,
        model_tag,
        in_channels = in_channels,
        out_channels_structural = out_channels_structural,
        out_channels_cell_content = out_channels_cell_content,
        structural_embedding_size=structural_embedding_size,
        structural_hidden_size=structural_hidden_size,
        structural_attention_size=structural_attention_size,
        cell_content_embedding_size=cell_content_embedding_size,
        cell_content_hidden_size=cell_content_hidden_size,
        cell_content_attention_size=cell_content_attention_size)

# reload latest checkpoint
model.load_checkpoint(file_path = "checkpoint_004.pth.tar")
predicted_struc_tokens, predictions_cell, structure_attention_weights , cell_attention_weights = model.predict(file1_path)

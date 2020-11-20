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
model.load_checkpoint(file_path = "checkpoint1.pth.tar")

image_path = '/Users/andersborges/Documents/Capstone/code/PMC1797029_008_00.png'

import sys
def build_html_structure(structure_information):
    ''' Build the structure skeleton of the HTML code
        Add the structural <thead> and the structural <tbody> sections to a fixed <html> header
    '''

    html_structure = '''<html>
                       <head>
                       <meta charset="UTF-8">
                       <style>
                       table, th, td {
                         border: 1px solid black;
                         font-size: 10px;
                       }
                       </style>
                       </head>
                       <body>
                       <table frame="hsides" rules="groups" width="100%%">
                         %s
                       </table>
                       </body>
                       </html>''' % ''.join(structure_information['tokens'])

    return html_structure


predicted_struc_tokens, predictions_cell, structure_attention_weights , cell_attention_weights = model.predict(image_path)
# print("structure_attention_weights")
# print(structure_attention_weights[0][0].size())
# print("cell_attention_weights")
# print(cell_attention_weights[0][0][0].size())
# print(predictions)

pred_dict = {}
pred_dict['tokens']=predicted_struc_tokens
pred_html_struc = build_html_structure(pred_dict)
f = open("pred.html","w" )
f.write(pred_html_struc)
f.close()
quit()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import skimage.transform

# load image
image = Image.open(image_path)
image = image.resize([32 * 12, 32 * 12], Image.LANCZOS)

structure_attention_weights = structure_attention_weights[0]

#  structural tokens
structural_tks = ['<start>'] + predicted_struc_tokens

num_subplots = min(len(structure_attention_weights), 25)

rows = num_subplots // 5
cols = min(num_subplots, 5)
fig, axes = plt.subplots(rows, cols)

for t in range(num_subplots):

    row = t // 5
    col = t % 5

    # to obtain structure tokens in every time step
    alphas = structure_attention_weights[t]
    alphas = alphas.detach().numpy()[:,0].reshape(12,12)

    axes[row, col].text(0, 1, '%s' % (structural_tks[t]), color='black', backgroundcolor='white', fontsize=7)
    axes[row, col].imshow(image)

    alphas = skimage.transform.pyramid_expand(alphas, upscale=32, sigma=8)

    axes[row, col].imshow(image)

    if t == 0:
        axes[row, col].imshow(alphas, alpha=0, cmap=cm.Greys_r)
    else:
        axes[row, col].imshow(alphas, alpha=0.8, cmap=cm.Greys_r)

    axes[row, col].axis('off')

plt.show()

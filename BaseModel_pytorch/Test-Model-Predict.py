relative_path = "."

# model_tag is the name of the folder that the checkpoints folders will be saved in

model_tag = "django_predict"

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
model.load_checkpoint("../../trained_struc_cell_dec.pth.tar")

image_path = '/Users/andersborges/Documents/Capstone/code/PMC1797029_008_00.png'

import sys


predicted_struc_tokens, predictions_cell, structure_attention_weights , cell_attention_weights = model.predict(image_path)


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
                       </html>''' % ''.join(structure_information)

    return html_structure


def fill_html_structure(html_structure, cells_information):
    ''' Fill the structure skeleton of the HTML code with the cells content
        Every cell description is stored in a separate "token" field
        An initial assessment is performed to check that the cells content
        is compatible with the HTML structure skeleton
    '''

    import re
    from bs4 import BeautifulSoup as bs

    # initial compatibility assessment
    cell_nodes = list(re.finditer(r'(<td[^<>]*>)(</td>)', html_structure))
    assert len(cell_nodes) == len(cells_information), 'Number of cells defined in tags does not match the length of cells'

    # create a list with each cell content compacted into a single string
    cells = [''.join(cell) for cell in cells_information]

    # sequentially fill the HTML structure with the cells content at the appropriate spots
    offset = 0
    html_string = html_structure
    for n, cell in zip(cell_nodes, cells):
        html_string = html_string[:n.end(1) + offset] + cell + html_string[n.start(2) + offset:]
        offset += len(cell)
    print(html_string)
#    soup = bs(html_string, features="lxml")
#    html_string = soup.prettify(formatter='minimal')
    return html_string

# print("structure_attention_weights")
# print(structure_attention_weights[0][0].size())
# print("cell_attention_weights")
# print(cell_attention_weights[0][0][0].size())
# print(predictions)

print(len(predicted_struc_tokens))
print(len(predictions_cell[0]))
html_struc = build_html_structure(predicted_struc_tokens)
html_cell = fill_html_structure(html_struc, predictions_cell)
print(html_cell)
#, predictions_cell,

f = open("pred.html","w" )
f.write(html_cell)
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

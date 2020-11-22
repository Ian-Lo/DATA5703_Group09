import os
import datetime
import random

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from FeatureVector.settings import MEDIA_ROOT, CAPTION_MODEL_ROOT


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import skimage.transform


import sys
sys.path.append('/Users/andersborges/Documents/Capstone/code/DATA5703_Group09/BaseModel_pytorch/')


def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + \
        str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name


def _get_caption(img_path):
    shell = "python -W ignore \
    main/caption.py \
    --img={} \
    --model='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' \
    --word_map='WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' \
    --beam_size=5".format(img_path)
    stream = os.popen(shell)

    return stream.read()

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
#    soup = bs(html_string, features="lxml")
#    html_string = soup.prettify(formatter='minimal')
    return html_string



def index(request):
    if request.POST:

        file1_path, file1_name = handle_uploaded_file(request.FILES['file1'])

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
        predicted_struc_tokens, predictions_cell, structure_attention_weights , cell_attention_weights = model.predict(file1_path)

        html_struc = build_html_structure(predicted_struc_tokens)
        html_out = fill_html_structure(html_struc, predictions_cell)

        # add attention plot

        # load image
        image = Image.open(file1_path)
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
            alphas = alphas.detach().numpy().reshape(12,12)

            axes[row, col].text(0, 1, '%s' % (structural_tks[t]), color='black', backgroundcolor='white', fontsize=7)
            axes[row, col].imshow(image)

            alphas = skimage.transform.pyramid_expand(alphas, upscale=32, sigma=8)

            axes[row, col].imshow(image)

            if t == 0:
                axes[row, col].imshow(alphas, alpha=0, cmap=cm.Greys_r)
            else:
                axes[row, col].imshow(alphas, alpha=0.8, cmap=cm.Greys_r)

            axes[row, col].axis('off')

        attention_file_path  = file1_path.replace(".png", "_attention.png")
        attention_file_name = attention_file_path.split("/media")[1]
        print( "attention_file_name", attention_file_name)
        plt.savefig(attention_file_path)


        return render(request, "index.html", {
            "post": True,
            "img1src": file1_name,
            "img2src": attention_file_name,
            "table_tags": html_out
        })

    return render(request, "index.html", {'post': False})


class Img2Vec:
    def __init__(self, cuda=True, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(
            model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.normalize(self.to_tensor(
            self.scaler(img))).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)

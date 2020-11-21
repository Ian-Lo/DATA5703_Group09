import os, datetime, random

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from FeatureVector.settings import MEDIA_ROOT

import sys
sys.path.append('/Users/andersborges/Documents/Capstone/code/DATA5703_Group09/BaseModel_pytorch/')


def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name

def index(request):
    if request.POST:
        # the relative path of the folder containing the dataset
        relative_path = "../../Dataset"

        # model_tag is the name of the folder that the checkpoints folders will be saved in

        model_tag = "baseline_cell"

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

        # get path of selected image
        image_path, file1_name = handle_uploaded_file(request.FILES['file1'])
        print(image_path)

        predictions, predictions_cell = model.predict(image_path)

        print(predictions)
        quit()



        print('\nCosine similarity: {0:.2f}\n'.format(float(cos_sim)))
        return render(request, "index.html", {"cos_sim": 'Score: {0:.2f}'.format(float(cos_sim)),
                                              "post": True,
                                              "img1src": file1_name,
                                              })
    return render(request, "index.html", {'post': False})


class Img2Vec:
    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

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
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

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

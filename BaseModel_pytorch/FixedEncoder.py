import torch
import torchvision as tv
import numpy as np


class FixedEncoder(torch.nn.Module):
    """
    Pretrained CNN Encoder
    """

    # constructor
    def __init__(self, cnn_model_name, features_map_size):

        super(FixedEncoder, self).__init__()

        # set up the image transformations required
        # to generate the input for the model
        if cnn_model_name in ['ResNet18']:
            # all ResNet reduce the initial image size by a factor of 32
            # so we resize the images to 32 times the size of the features map
            preprocess = tv.transforms.Compose([tv.transforms.Resize((32 * features_map_size,
                                                                      32 * features_map_size)),
                                                tv.transforms.ToTensor(),
                                                tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])
        else:
            preprocess = None

        # import the layers of the pretrained CNN model
        # drop the layers only used for classification
        # as we are just interested in the final maps
        if cnn_model_name == 'ResNet18':
            cnn_model = tv.models.resnet18(pretrained=True)
            layers = list(cnn_model.children())[:-2]
        else:
            layers = None

        self.preprocess = preprocess
        self.encoder = torch.nn.Sequential(*layers)

    # preprocess the image
    def preprocess_image(self, image):

        # transform the images to fulfil the input requirement of the CNN
        input_image = self.preprocess(image)

        return input_image

    # generate the features map
    def encode(self, input_batch):

        # forward propagate
        with torch.no_grad():
            features_maps = self.encoder(input_batch)

        # reshape (n, l, size, size) -> (n, l, size * size)
        # where size is the features map size
        features_maps = torch.reshape(features_maps, (features_maps.shape[0], features_maps.shape[1], -1))

        # convert to array and type cast to reduce storage requirements
        features_maps = features_maps.numpy().astype(np.float16)

        # transpose (n, l, features_map_size * features_map_size) ->
        #           (n, features_map_size * features_map_size, l)
        # we do this to prepare for the second encoder which will reduce the magnitude of l
        # the value of l is solely determined by the type of ResNet
        # e.g. for ResNet-18, l = 512
        features_maps = np.swapaxes(features_maps, 1, 2)

        return features_maps

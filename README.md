

# INSTRUCTIONS ON HOW TO RUN THE MODEL

* An example of how to run the model is provided in this Colab notebook:
https://colab.research.google.com/drive/1y9kbMLHX3UC3LEZhzauRrJxemZAKC3f_?usp=sharing

* This script will set up folders, clone the repository and download the HDF5 files. It then allows running the training of the model.

# REQUIREMENTS

* Python 3.8
  * numpy
  * pandas
  * matplotlib
  * jsonlines
  * pillow (PIL)
  * h5py
  * pytorch
  * torchvision
  * tdqm

# THE MODEL: ENCODER-DECODER

This is the structure of the model:

* PRETRAINED ENCODER

* ENCODER

* STRUCTURAL ATTENTION MECHANISM + STRUCTURAL DECODER

* CELL CONTENT ATTENTION MECHANISM + CELL CONTENT DECODER

* (OPTIONAL) DATA EXPLORATION AND SUBSET CREATION (PREPROCESSING)

The dataset for this project is PubTabNet. It is only required if performing data exploration and preprocessing.
PubTabNet: this dataset can be downloaded from https://developer.ibm.com/technologies/artificial-intelligence/data/pubtabnet/
Untar the file in a convenient location and leave its folder structure intact.
Under the 'PubTabNet' folder, create a folder named 'Dataset'.
The pretrained encoder for this model is a pretrained CNN network (ResNet18). Because this section of the model is frozen, we can take all images, pass them through the ResNet18 and consider the output tensors as the real input to the model. Most of the preprocessing consists in transforming each image in its features map and storing it away for further usage later on.

# HDF5 FILES   

10k examples were randomly selected among the 500k images in the PubTabNet training set to form the validation set for this model.
10k examples were randomly selected among the 500k images in the PubTabNet training set to form the test set for this model.
100k examples were selected among the 500k images in the PubTabNet training set to form the training set for this model. 
In order to reduce the training load, these 100k images all fulfil some criteria (<200 structural tokens, <30 cells, <100 cell content tokens).
All these examples were preprocessed and the result of the preprocessing stored in HDF5 files for easiness of retrival.
Images were passed through a ResNet18 and the final tensor was stored in the HDF5 files.
Structural tokens and cell content tokens were converted into numbers by means of vocabularies and the final tensors were stored in the HDF5 files.
The input of the model are these HDF5 files not the original PubTabNet files.

To reproduce the preprocessing step run the following files, in this order:

* Dataset-Exploration.py (this, among other things, will create the vocabularies of structural and cell content tokens)

* Dataset-Visualisation.py

* Analysis_Test_Dev.py

* Subset-Criteria-Analysis.py

* Subset_Creation_dev.py

* Subset_Creation_test.py

* Subset-Creation.py

# BATCHINGMECHANISM AND STORAGE CLASS

The BatchingMechanism class contains the infrustructure to gather batches from the HDF5 files. It leverage the Storage class that wraps the low level communication with the HDF5 files. 

# ENCODERSTRUCTURAL AND ENCODERCELLCONTENT CLASS

The encoder class contains both a 1x1 convolutional layer and a fully connected layer that takes in the features maps tensors contained in the HDF5 files and produces transformed features maps. The function of this layer is to adapt the features maps produced by the ResNet18 to out task. There is the option of using the 1x1 convolutional layer or the fully connected layer to perform this action. 

# STRUCTURALATTENTION CLASS

This StructuralAttention class contains the attention mechanism for the structural decoder.

# DECODERSTRUCTURAL CLASS

The DecoderStructural class contains the decoder for the structural tokens.

# CELLCONTENTATTENTION CLASS

The CellContentAttention class contains the attention mechanism for the cell content tokens.

# DECODERCELLCONTENT CLASS

The DecoderCellContent class contains the decoder for the structural tokens.

# CHECKPOINT CLASS

The CheckPoint class contains functionality to save and retrieve checkpoints. A checkpoint contains information about the weights/biases of the various fully connected and convolutional layers that define the model.

# UTILS CLASS

The Utils class contains functionality to set up file paths.

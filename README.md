REQUIREMENTS

Python 3.8

numpy
pandas
matplotlib
jsonlines
pillow (PIL)
h5py
pytorch
torchvision
tdqm

INSTRUCTIONS ON HOW TO RUN THE MODEL

Run the script run.sh (in the DevOps folder) in a bash terminal.

Then edit the file Test_Model.py: in the first line, edit the folder string to point it to the folder where the Dataset folder is on 
the local machine.

Finally, run: python Test_Model.py from the directory BaseModel_pytorch

THE MODEL: ENCODER-DECODER

This is the structure of the model:

PRETRAINED ENCODER

ENCODER

STRUCTURAL ATTENTION MECHANISM + STRUCTURAL DECODER

CELL CONTENT ATTENTION MECHANISM + CELL CONTENT DECODER

(OPTIONAL) DATA EXPLORATION AND SUBSET CREATION (PREPROCESSING)

The dataset for this project is PubTabNet. It is only required if performing data exploration and preprocessing.
PubTabNet: this dataset can be downloaded from https://developer.ibm.com/technologies/artificial-intelligence/data/pubtabnet/
Untar the file in a convenient location and leave its folder structure intact.
Under the 'PubTabNet' folder, create a folder named 'Dataset'.
The pretrained encoder for this model is a pretrained CNN network (ResNet18). Because this section of the model is frozen,
we can take all images, pass them through the ResNet18 and consider the output tensors as the real input to the model. 
Most of the preprocessing consists in transforming each image in its features map and storing it away for further usage later on.

HDF5 FILES   

10k examples were randomly selected among the 500k images in the PubTabNet training set to form the validation set for this model.
10k examples were randomly selected among the 500k images in the PubTabNet training set to form the test set for this model.
100k examples were selected among the 500k images in the PubTabNet training set to form the training set for this model. 
In order to reduce the training load, these 100k images all fulfil some criteria (<200 structural tokens, <30 cells, <100 cell content tokens).
All these examples were preprocessed and the result of the preprocessing stored in HDF5 files for easiness of retrival.
Images were passed through a ResNet18 and the final tensor was stored in the HDF5 files.
Structural tokens and cell content tokens were converted into numbers by means of vocabularies and the final tensors were stored in the HDF5 files.
The input of the model are these HDF5 files not the original PubTabNet files.

BATCHINGMECHANISM AND STORAGE CLASS

The BatchingMechanism class contains the infrustructure to gather batches from the HDF5 files. It leverage the Storage class that wraps 
the low level communication with the HDF5 files. 

ENCODER CLASS

The encoder class contains a 1x1 convolutional layer that takes in the features maps tensors contained in the HDF5 files and produces 
transformed features maps. The function of this layer is to adapt the features maps produced by the ResNet18 to out task.

STRUCTURALATTENTION CLASS

This StructuralAttention class contains the attention mechanism for the struictural decoder.

DECODERSTRUCTURAL CLASS

The DecoderStructural class contains the decoder for the structural tokens.

CELLCONTENTATTENTION CLASS

The CellContentAttention class contains the attention mechanism for the cell content tokens.

DECODERCELLCONTENT CLASS

The DecoderCellContent class contains the decoder for the structural tokens.

CHECKPOINT CLASS

The CheckPoint class contains functionality to save and retrieve checkpoints. A checkpoint contains information about the weights/biases
of the various fully connected and convolutional layers that define the model.

UTILS CLASS

The Utils class contains functionality to set up file paths.

# the relative path of the folder containing the dataset
relative_path = "../../Dataset"

# model_tag is the name of the folder that the checkpoints folders will be saved in
model_tag = "baseline_try"

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

# 50 epochs with batch size 10 to get a decent length

# set number of epochs
epochs = 10
#epochs = 25

# make list of lambdas to use for each epoch in training
lambdas = [1 for n in range(epochs) ] # LAMBDA = 1 turns OFF cell decoder
#lambdas = [1.0]*25 + 25*[1, 1, 0.5, 0.5]# for n in range(epochs)] # LAMBDA = 1 turns OFF cell decoder
# if you want to run WITH cell decoder, you can uncomment the line below, remember to change epochs to 25
#lambdas = [1 for _ in range(30)] + [0.5 for _ in range(70)]#+ [0.5 for _ in range(10)] + [0.5 for _ in range(2)]

# make list of learning rate to use for each epoch in training
lrs = [0.001 for _ in range(epochs)] #+ [0.001]*25
#lrs =[0.001 for n in range(20)]+ [0.0001 for _ in range(30)] + [0.00001 for _ in range(50)]# + [0.001 for _ in range(10)] + [0.0001 for _ in range(2)]
#if you want to run WITH cell decoder, you can uncomment the line below, rembember to change epochs to 25
#lrs = [0.001 for _ in range(10)] + [0.0001 for _ in range(3)] + [0.001 for _ in range(10)] + [0.0001 for _ in range(2)]

# Number of examples to include in the training set
number_examples=1000

# Number of examples to include in validation set
number_examples_val=3000 # not used if val==None

# size of batches
batch_size=10
batch_size_val = 10

# number of examples in each preprocessed file
storage_size=1000 # fixed, do not change

# whether to calculate the validation loss
f = 0
val = f*[False]+(epochs-f)*[True]#, False, True, True]

maxT_val = 200

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

#model.load_checkpoint(file_path="checkpoint.pth.tar")

# train model

loss_s, loss_s_val = model.train(epochs=epochs,
            lambdas=lambdas,
            lrs=lrs,
            number_examples=number_examples,
            number_examples_val=number_examples_val,
            batch_size=batch_size,
            batch_size_val = batch_size_val,
            storage_size=storage_size,
            val = val,
            maxT_val = maxT_val)

# checkout branch test-link2

# the relative path of the folder containing the dataset
relative_path = "../../Dataset"

# model_tag is the name of the folder that the checkpoints folders will be saved in

model_tag = "test"

# tunable parameters
out_channels_structural = 64 # number of channels
out_channels_cell_content =64 # number of channels
structural_hidden_size = 128 # dimensions of hidden layer in structural decoder
structural_attention_size = 128 # dimensions of context vector in structural decoder
cell_content_hidden_size = 256 # dimensions of hidden layer in cell decoder
cell_content_attention_size = 128 # dimensions of ontext vector in structural decoder

# fixed parameters
in_channels = 512 # fixed in output from resnet, do not change
structural_embedding_size = 16 # determined from preprocessing, do not change
cell_content_embedding_size = 80 # determined from preprocessing, do not change

# set number of epochs
epochs =100
# make list of lambdas to use for each epoch in training
lambdas = epochs*[0.0]#+10*[0.5]
# make list of learning rate to use for each epoch in training
lrs = epochs * [0.0001] #+ [0.001]*25

# whether to calculate the validation loss
f = epochs
val = f*[False]#+(epochs-f)*[True]#, False, True, True]
# whether to neglect structural loss and only backpropagate the structural decoder
test_link = epochs*[True]

# Number of examples to include in the training set
number_examples=1

# Number of examples to include in validation set
number_examples_val=1 # not used if val==None

# size of batches
batch_size=10
batch_size_val = 10

# number of examples in each preprocessed file
storage_size=1000 # fixed, do not change


maxT_val = 100

alpha_c_struc = 0.0
alpha_c_cell_content = 0.0

structural_encoder_conv = False
cell_content_encoder_conv = False


from Model import Model
from matplotlib import pylab as plt

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
                cell_content_attention_size=cell_content_attention_size,
                structural_encoder_conv = False)

loss,loss_s, loss_cc, loss_val, loss_s_val, loss_cc_val = model.train(epochs=epochs,
            lambdas=lambdas,
            lrs=lrs,
            number_examples=number_examples,
            number_examples_val=number_examples_val,
            batch_size=batch_size,
            batch_size_val = batch_size_val,
            storage_size=storage_size,
            val = val,
            maxT_val = maxT_val,
            alpha_c_struc = alpha_c_struc,
            alpha_c_cell_content = alpha_c_cell_content,
            test_link = test_link)

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})

fig, ax1 = plt.subplots(figsize=(4,2))
ax1.title.set_text("Test structural-/cell decoder link")

color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Cell decoder loss', color=color)
ax1.plot(range(1, len(loss_cc)+1), loss_cc, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Structural decoder loss', color=color)  # we already handled the x-label with ax1
ax2.plot(range(1, len(loss_s)+1), loss_s, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig("Figures/Test-Link.png")

model_tag = "baseline_min"

relative_path = "../../Dataset/"

# tunable parameters
out_channels = 64
structural_hidden_size=256
structural_attention_size=64
cell_content_hidden_size=64
cell_content_attention_size=64

# fixed parameters
in_channels=512
encoder_size =12
structural_embedding_size=16
cell_content_embedding_size=80

# set number of epochs
epochs = 100
#epochs = 1

# make list of lambdas to use in training
# this is the same strategy as Zhong et al.
lambdas = [1 for n in range(100)]# [1 for _ in range(10)] + [1 for _ in range(3)]+ [0.5 for _ in range(10)] + [0.5 for _ in range(2)]
lrs =[0.01 for n in range(100)]# [0.001 for _ in range(10)] + [0.0001 for _ in range(3)] + [0.001 for _ in range(10)] + [0.0001 for _ in range(2)]

#lambdas =  [1 for _ in range(10)] + [1 for _ in range(3)]+ [0.5 for _ in range(10)] + [0.5 for _ in range(2)]
#lrs = [0.01 for _ in range(10)] + [0.0001 for _ in range(3)] + [0.001 for _ in range(10)] + [0.0001 for _ in range(2)]


assert epochs == len(lambdas) == len(lrs), "number of epoch, learning rates and lambdas are inconsistent"

number_examples=1
number_examples_val=1 # not used if val==None
batch_size=1
storage_size=1000
val = False


from Model import Model

model = Model(relative_path, model_tag, in_channels = in_channels, out_channels = out_channels, encoder_size = encoder_size, structural_embedding_size=structural_embedding_size, structural_hidden_size=structural_hidden_size, structural_attention_size=structural_attention_size, cell_content_embedding_size=cell_content_embedding_size, cell_content_hidden_size=cell_content_hidden_size, cell_content_attention_size=cell_content_attention_size)

model.train(epochs=epochs, lambdas=lambdas, lrs=lrs, number_examples=number_examples, number_examples_val=number_examples_val, batch_size=batch_size, storage_size=storage_size,val = val)

from Model import Model
#fjkalsdfjaslkj
# initialize model class
model = Model("../../Dataset/" , "test_run", encoder_size = 12, structural_embedding_size = 16, structural_hidden_size = 256, structural_attention_size = 256, cell_content_embedding_size = 80, cell_content_hidden_size = 512, cell_content_attention_size = 256 )

# set number of epochs
epochs = 25

# make list of lambdas to use in training
# this is the same strategy as Zhong et al.
lambdas =  [1 for _ in range(10)] + [1 for _ in range(3)]+ [0.5 for _ in range(10)] + [0.5 for _ in range(2)]
lrs = [0.001 for _ in range(10)] + [0.0001 for _ in range(3)] + [0.001 for _ in range(10)] + [0.0001 for _ in range(2)]

model.train(epochs = epochs, lambdas = lambdas, lrs = lrs)

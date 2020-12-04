import torch


class StructuralAttention(torch.nn.Module):

    def __init__(self, encoder_size, structural_hidden_size, structural_attention_size):

        super(StructuralAttention, self).__init__()

        # attention layers
        self.attention_encoded_features_map = torch.nn.Linear(encoder_size, structural_attention_size)
        self.attention_structural_hidden_state = torch.nn.Linear(structural_hidden_size, structural_attention_size)
        self.attention_combined = torch.nn.Linear(structural_attention_size, 1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        #self.sigmoid = torch.nn.Sigmoid()


    def forward(self, encoded_features_map, structural_hidden_state):

        # compute attention on encoded features map
        # attention_encoded_features_map: (batch_size, n*n, structural_attention_size)
        attention_encoded_features_map = self.attention_encoded_features_map(encoded_features_map)

        # compute attention on structural hidden state
        # we remove the first dimension that is only needed by the GRU
        structural_hidden_state = torch.squeeze(structural_hidden_state, 0)
        attention_structural_hidden_state = self.attention_structural_hidden_state(structural_hidden_state)
        # we add an extra dimension so to be able to sum up the two tensor
        # attention_structural_hidden: (batch_size, 1, structural_attention_size)
#        print(attention_structural_hidden_state.shape)
#         quit()

        attention_structural_hidden_state = attention_structural_hidden_state.unsqueeze(1)
        # print("attention_structural_hidden_state[0]")
        # print(torch.min(attention_structural_hidden_state[0]), torch.max(attention_structural_hidden_state[0]))
        # print("attention_encoded_features_map[0]")
        # print(torch.min(attention_encoded_features_map[0]), torch.max(attention_encoded_features_map[0]))

#         from matplotlib import pylab as plt
#         import numpy as np
#         fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (50, 10))
#         ax0 = axes.flat[0]
#         ax1 = axes.flat[1]
#         print(encoded_features_map[0,:,:].detach().numpy().shape)
#         print(structural_hidden_state.detach().numpy().shape)
#         print(attention_encoded_features_map[0,:,:].detach().numpy().shape)
#         print(np.repeat(attention_structural_hidden_state[0,:,:].detach().numpy(), 144, axis = 0).shape)
#         print( (attention_encoded_features_map + attention_structural_hidden_state).shape)
#
#         ax0.matshow(encoded_features_map[0,:,:].detach().numpy())
#         ax1.matshow(structural_hidden_state.detach().numpy())
#         plt.savefig('Figures/attention_input.png')
#
#         fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (6,1.7))
#         ax0 = axes.flat[0]
#         ax1 = axes.flat[1]
#         ax2 = axes.flat[2]
#         print("attention_encoded_features_map")
#         print(attention_encoded_features_map[0,:,:].detach().numpy().min(), attention_encoded_features_map[0,:,:].detach().numpy().max())
#         print("attention_structural_hidden_state")
#         print(attention_structural_hidden_state[0,:,:].detach().numpy().min(), attention_structural_hidden_state[0,:,:].detach().numpy().max())
#         min_ = min(attention_encoded_features_map[0,:,:].detach().numpy().min(),attention_structural_hidden_state[0,:,:].detach().numpy().min() )
#         max_ = max(attention_encoded_features_map[0,:,:].detach().numpy().max(),attention_structural_hidden_state[0,:,:].detach().numpy().max() )
#         im0 = ax0.matshow(attention_encoded_features_map[0,:,:].detach().numpy(),vmin = min_, vmax = max_)
#         ax0.set_xticks([])
#         ax0.set_yticks([])
#         im1 = ax1.matshow(np.repeat(attention_structural_hidden_state[0,:,:].detach().numpy(), 144, axis = 0), vmin = min_, vmax = max_)
#         ax1.set_xticks([])
#         ax1.set_yticks([])
#         im2 = ax2.matshow( (attention_encoded_features_map + attention_structural_hidden_state).detach().numpy()[0,:,:], vmin  = min_, vmax = max_)
#         ax2.set_xticks([])
#         ax2.set_yticks([])
#         fig.subplots_adjust(bottom = 0.4)
# #      cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#         cbar_ax = fig.add_axes([0.15, 0.25, 0.7, 0.07])
#         fig.colorbar(im2, cax=cbar_ax, orientation = 'horizontal')
# #        fig.tight_layout()
#         plt.savefig('Figures/attention_intoReLu.png')
# #        quit()

        #plt.close()

        # print(structural_hidden_state.shape)
        # plt.matshow(structural_hidden_state.detach().numpy())
        # plt.colorbar()
        # plt.savefig("structural_hidden_state.png")
        # plt.close()
        #
        # # combine the attentions
        # print(attention_encoded_features_map.shape)
        # print(attention_structural_hidden_state.shape)
        # from matplotlib import pylab as plt
        # plt.matshow(attention_encoded_features_map[0,:,:].detach().numpy())
        # plt.colorbar()
        # plt.savefig('attention_encoded_features_map.png')
        # plt.close()
        #
        # plt.matshow(attention_structural_hidden_state[:,0,:].detach().numpy())
        # plt.colorbar()
        # plt.savefig('attention_structural_hidden_state.png')
        # plt.close()


        attention_combined = self.attention_combined(self.relu(attention_encoded_features_map + attention_structural_hidden_state))
        # we remove last dimension
        # attention_combined: (batch_size, n*n)
        attention_combined = torch.squeeze(attention_combined, 2)

        # compute the attention weights
        attention_weights = self.softmax(attention_combined)
#        print("attention_weigths")
#        print(attention_weights.detach().numpy().min(), attention_weights.detach().numpy().max(), attention_weights.detach().numpy().mean())
        # we restore last dimension
        # attention_weights: (batch_size, n*n, 1)
        attention_weights = attention_weights.unsqueeze(2)

        # context_vector: (batch_size, encoder_size)
        context_vector = (encoded_features_map * attention_weights).sum(dim=1)


        attention_weights = torch.squeeze(attention_weights, 2)

        return context_vector, attention_weights

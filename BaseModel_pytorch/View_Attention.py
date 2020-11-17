import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import skimage.transform
import numpy as np

# load image
#image = Image.open(image_path)
#image = image.resize([32 * 12, 32 * 12], Image.LANCZOS)

structure_attention_weights_all = np.loadtxt('testA.csv', delimiter=',')

#  structural tokens
structural_tks = ['<pad>']*10

num_subplots = 10 #min(len(structure_attention_weights_all), 25)

rows = 3
cols = 4

for j in range(15):

    structure_attention_weights = structure_attention_weights_all[0+9*j:10+9*j]
    print(structure_attention_weights.shape)

    fig, axes = plt.subplots(rows, cols)

    for t in range(num_subplots):
        print(t)
        row = t // 4
        col = t % 4

        # to obtain structure tokens in every time step
        alphas = structure_attention_weights[t]
        alphas = alphas.reshape(12, 12)

        axes[row, col].text(0, 1, '%s' % (structural_tks[t]), color='black', backgroundcolor='white', fontsize=7)

        alphas = skimage.transform.pyramid_expand(alphas, upscale=32, sigma=8)

        #if t == 0:
        #    axes[row, col].imshow(alphas, alpha=0, cmap=cm.Greys_r)
        #else:
        axes[row, col].imshow(alphas, alpha=0.8, cmap=cm.Greys_r)

        axes[row, col].axis('off')

    plt.show()

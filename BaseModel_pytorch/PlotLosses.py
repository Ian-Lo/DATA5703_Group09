from matplotlib import pylab as plt
import numpy as np
from glob import glob as glob


f = open("DataForPlotting/train_baseline_num_examples_val_maxT200.txt", "r")
ls = f.readlines()
epochs = []
losses = []
val_losses = []

for l in ls:
    if "epoch" in l:
        epochs.append(int(l.split()[1]))
    if "Struct. decod. loss:" in l:
        losses.append(float(l.split()[-1]))
    if "Validation struct. decod. loss:" in l:
        val_losses.append(float(l.split()[-1]))
f.close()

plt.plot(epochs, losses, label = 'training loss')
plt.plot(epochs, val_losses, label = "validation loss")
plt.ylabel("Loss")
plt.xlabel("epochs")
plt.title("10k samples, lr=0.001, lambda = 1.0")
plt.legend()
plt.savefig("Figures/train_val_loss.png")

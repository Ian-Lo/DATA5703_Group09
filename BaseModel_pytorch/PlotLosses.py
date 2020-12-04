from matplotlib import pylab as plt
import numpy as np
from glob import glob as glob


import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8})


# plot training and validation loss for trained model
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

fig, ax1 = plt.subplots(figsize=(4,2))
ax1.title.set_text("10k samples, lr=0.001, lambda = 1.0")
ax1.plot(epochs, losses, label = 'Training')
ax1.plot(epochs, val_losses, label = "Validation")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Epochs")
ax1.legend()
fig.tight_layout()
plt.savefig("Figures/train_val_loss.png")

# plot structural loss and validation loss for pretrained model


# plot training and validation loss for trained model
f = open("DataForPlotting/train_baseline_cell_decoder.txt", "r")
ls = f.readlines()
epochs = []
losses_total = []
losses_s = []
losses_cc = []
for l in ls:
    if "epoch" in l:
        epochs.append(int(l.split()[1]))
    if "Total loss:" in l:
        losses_total.append(float(l.split()[-1]))
    if "Struct. decod. loss:" in l:
        losses_s.append(float(l.split()[-1]))
    if "Cell dec. loss:" in l:
        losses_cc.append(float(l.split("(")[1].split(",")[0]))

f.close()

fig, ax1 = plt.subplots(figsize=(4,2))
ax1.title.set_text("10k samples, lr=0.001, lambda = 0.5")
ax1.plot(epochs, losses_total, label = 'Total loss')
ax1.plot(epochs, losses_s, label = "Structural loss")
ax1.plot(epochs, losses_cc, label = "Cell content loss")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Epochs")
ax1.legend(loc = 1)
fig.tight_layout()
plt.savefig("Figures/train_cell.png")

# plot structural loss and validation loss for pretrained model

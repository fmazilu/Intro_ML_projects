import matplotlib.pylab as plt
from data_loader import *

# Histogram for each attribute
fig, axs = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle("Histogram for each attribute", fontsize=14)
for i, col in enumerate(["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]):
    axs[int((i-(i%3))/3),i%3].hist(df[col], color=(0.2, i*0.1, 0.4))
    axs[int((i-(i%3))/3),i%3].set_xlabel(col)        
    axs[int((i-(i%3))/3),i%3].set_ylabel("Counts")
plt.subplots_adjust(hspace=0.6)
# plt.show()

# Box plot for each attribute
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Box-plot for each attribute", fontsize=14)
for i, col in enumerate(["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]):
        boxplot = df.boxplot(column=col, ax=axs[int((i-(i%3))/3),i%3])
# plt.show()

# Box plot for each attribute for each class
# Class "float processed glass"
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Box-plot for each attribute for float glass", fontsize=14)
for i, col in enumerate(["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]):
        boxplot = df.where(df["type"] == 1).boxplot(column=col, ax=axs[int((i-(i%3))/3),i%3])
# plt.show()

# Class "non-float processed glass"
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Box-plot for each attribute for non-float glass", fontsize=14)
for i, col in enumerate(["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]):
        boxplot = df.where(df["type"] == 2).boxplot(column=col, ax=axs[int((i-(i%3))/3),i%3])
plt.show()

# A matrix of scatter plots of each combination of two attributes against each other (emphasis each glass type)
fig, ax = plt.subplots(4, 4, figsize=(16, 16))
for i, col_1 in enumerate(["Na", "Mg", "Si", "Ca"]):
        for j, col_2 in enumerate(["Na", "Mg", "Si", "Ca"]):
                ax[i,j].set_xlabel(col_1)
                ax[i,j].set_ylabel(col_2)
                ax[i,j].scatter(df.where(df["type"] == 1)[col_2],  df.where(df["type"] == 1)[col_1], c="limegreen",s=50)
                ax[i,j].scatter(df.where(df["type"] == 2)[col_2],  df.where(df["type"] == 2)[col_1], c="darkorange", s=50)
                ax[i,j].legend(classNames)
plt.subplots_adjust(hspace=0.5)
plt.show()
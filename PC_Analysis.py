import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show, legend, title, grid
from scipy.linalg import svd

filename = 'dataset/US_Accidents_small.csv'
df = pd.read_csv(filename)

attributeNames = np.asarray(df.columns)
attributeNames = attributeNames[2:12]
print(attributeNames)

# will do a bit of encoding such as Right = 1 and Left = 0, Day = 1 & Night = 0
# and also encoding of Weather_Condition
df['Side'] = df['Side'].replace({'R': 1, 'L': 0})
df = df.replace({'Day': 1, 'Night': 0})

# Seeing how many unique values are in the Weather_condition column
weather_cond = df['Weather_Condition'].unique()
k = len(weather_cond)
weather_cond_int = np.arange(k) + 1
# Encoding weather_condition
cond = df['Weather_Condition']
cond = cond.replace(weather_cond, weather_cond_int)
# print(cond)
df = df.assign(Weather_Condition=cond)
# print(df.to_string())

# X,y-format
data = np.array(df.values)
# dropping the index
data = data[:, 1:]
# print(data.shape)

# If the modelling problem of interest was a classification problem where
# we wanted to classify the origin attribute, we could now identify obtain
# the data in the X,y-format as so:
y_c = data[:, 0].copy()
print(y_c)
X_c = data[:, 1:].copy()
# Dropping columns that have boolean values
X_c = X_c[:, :10]
X_c = np.array(X_c, dtype=np.float64)
# print(X_c.shape)

# Compute values of N, M and C.
N = len(y_c)
M = X_c.shape[1]
print(N, M)
C = np.max(y_c)
print(C)

# Subtract mean value from data
X_tilde = X_c - np.ones((N, 1))*X_c.mean(0)

# Not sure if standardizing the data is the best course of action
# print(np.std(X_tilde, 0))
X_tilde = X_tilde * (1 / np.std(X_tilde, 0))

# PCA by computing SVD of X_tilde
U, S, Vh = svd(X_tilde, full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

# Project the centered data onto principal component space
Z = X_tilde @ V

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

threshold = 0.9

# Plot variance explained
figure()
plot(range(1, len(rho)+1), rho, 'x-')
plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
plot([1, len(rho)], [threshold, threshold], 'k--')
title('Variance explained by principal components')
xlabel('Principal component')
ylabel('Variance explained')
legend(['Individual', 'Cumulative', 'Threshold'])
grid()

# Plotting the first three PCs and how the extract information from the original features
figure()
pcs = [0, 1, 2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r', 'g', 'b']
bw = .2
r = np.arange(1, M+1)
for i in pcs:
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('US Accidents: PCA Component Coefficients')

# Indices of the principal components to be plotted
i = 0
j = 1
k = 2

# Much more principal components are needed to be plotted to separate the classes
f = figure()
title('US Accidents data: PCA')
# Z = array(Z)
for c in range(C + 1):
    # select indices belonging to class c:
    class_mask = y_c == c
    plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
legend([0, 1, 2, 3, 4])
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Plot PCA of the data
f = figure().add_subplot(projection='3d')
title('US Accidents data: PCA')
# Z = array(Z)
for c in range(C + 1):
    # select indices belonging to class c:
    class_mask = y_c == c
    plot(Z[class_mask, i], Z[class_mask, j], Z[class_mask, k], 'o', alpha=.5)
legend(['0', '1', '2', '3', '4'])
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))
f.set_zlabel(f'PC{k+1}')

# Output result to screen
show()
# As it can be seen the data is not separable using just 2 PCs

# TODO: Drop the columns that have boolean values

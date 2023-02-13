from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

data = np.array([[0]*10]*10)
for i in range(5):
    data[2*i][2*i] = 1

cmap = colors.ListedColormap(['Blue', 'red'])
plt.figure(figsize=(6, 6))
plt.pcolor(data, cmap=cmap, edgecolors='k', linewidths=3)
plt.show()

print(data[::-1])
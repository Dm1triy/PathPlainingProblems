import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

data = np.array([[0]*10]*10)
indx1 = np.array([[1, 3], [2, 8], [5, 4]])
indx1 = np.array(np.split(indx1, 2, axis=1))
indx1 = indx1.reshape(indx1.shape[0], indx1.shape[1])
print(indx1)

indx = [[1, 2, 5], [3, 8, 4]]
for i in range(5):
    data[2*i][2*i] = 1
data[indx[0], indx[1]] = 1
print(data)

fig, ax = plt.subplots(figsize=(6, 6))
cmap = colors.ListedColormap(['white', 'blue'])
ax.pcolor(data, cmap=cmap, snap=True)
ax.grid()
plt.show()
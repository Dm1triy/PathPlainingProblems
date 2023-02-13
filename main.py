import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

data = np.array([[0]*10]*10)
for i in range(5):
    data[2*i][2*i] = 1

print(data)

cmap = colors.ListedColormap(['white', 'blue'])
bounds = [0, 1, 2]
norm = colors.BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(10, 10))

# ax.set_xlim([0, 10])  # 100 cells * 0.05 cell_size = 10 meters
# ax.set_ylim([0, 10])

ax.imshow(data, cmap=cmap, norm=norm)
ax.grid()
ax.set_xticks(np.arange(0, 10, 1))
ax.set_yticks(np.arange(0, 10, 1))

plt.show()
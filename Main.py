from Math import find_temp
import numpy as np
import matplotlib.pyplot as plt

accuracy = 0.01
gridSize = int(float(2) / accuracy)
solution = np.zeros((gridSize, gridSize))
y = -1 + accuracy / 2

for i in range(gridSize):
    x = -1 + accuracy / 2
    for j in range(gridSize):
        solution[i][j] = find_temp(x, y)
        x += accuracy
    y += accuracy

for i in range(int(gridSize / 2), gridSize):
    for j in range(int(gridSize / 2), gridSize):
        solution[i][j] = 0

    plt.imshow(solution, cmap='magma', vmin=0, vmax=2, origin='lower', aspect='equal', extent=(-1, 1, -1, 1))
    plt.colorbar()
    plt.show()

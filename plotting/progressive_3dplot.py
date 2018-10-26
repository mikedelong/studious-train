# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


def update_graph(num):
    t0 = num % count
    graph = ax.scatter(xs[:t0], ys[:t0], zs[:t0], marker='o', color='blue')
    return graph


count = 100
xs = range(count)
ys = [2 * x for x in xs]
zs = np.random.rand(count)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(left=-1, right=count + 1)
ax.set_ylim3d(bottom=-1, top=2 * count + 1)
ax.set_zlim3d(bottom=0, top=1)

animation = FuncAnimation(fig, update_graph, interval=40, blit=False, repeat=False)
plt.show()

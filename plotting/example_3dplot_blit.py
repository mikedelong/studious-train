import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

a = np.random.rand(2000, 3) * 10
t = np.array([np.ones(100) * i for i in range(20)]).flatten()
df = pd.DataFrame({'time': t, 'x': a[:, 0], 'y': a[:, 1], 'z': a[:, 2]})


def update_graph(num):
    data = df[df['time'] == num]
    graph.set_data(data.x, data.y)
    graph.set_3d_properties(data.z)
    title.set_text('3D test, time={}'.format(num))
    return title, graph,


fig = plt.figure()
ax = Axes3D(fig)
title = ax.set_title('3D test')

data = df[df['time'] == 0]
graph, = ax.plot(data.x, data.y, data.z, linestyle='', marker='o')
ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=40, blit=False)
plt.show()

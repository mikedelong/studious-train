import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = np.random.rand(2000, 3) * 10
t = np.array([np.ones(100) * i for i in range(20)]).flatten()
df = pd.DataFrame({'time': t, 'x': a[:, 0], 'y': a[:, 1], 'z': a[:, 2]})


def update_graph(num):
    data = df[df['time'] == num]
    graph._offset3d = (data.x, data.y, data.z)
    title.set_text('3D test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D test')

data = df[df['time'] == 0]
graph = ax.scatter(data.x, data.y, data.z)
ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, interval=40, blit=False)
plt.show()

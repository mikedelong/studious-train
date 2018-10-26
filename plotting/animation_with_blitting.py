import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class AnimatedScatter(object):
    def __init__(self, numpoints=50):
        self.numpoints = numpoints
        self.stream = self.data_stream()

        self.fig, self.ax = plt.subplots()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        x, y, s, c = next(self.stream)
        self.scat = self.ax.scatter(x, y, c=c, s=s, animated=True)
        self.ax.axis([-10, 10, -10, 10])
        return self.scat,

    def data_stream(self):
        data = np.random.random((4, self.numpoints))
        xy = data[:2, :]
        s, c = data[2:, :]
        xy -= -0.5
        xy *= 10
        while True:
            xy += 0.03 * (np.random.random((2, self.numpoints)) - 0.5)
            s += 0.05 * (np.random.random(self.numpoints) - 0.5)
            c += 0.02 * (np.random.random(self.numpoints) - 0.5)
            yield data

    def update(self, i):
        data = next(self.stream)
        self.scat.set_offsets(data[:2, :])
        self.scat._sizes = 300 * abs(data[2]) ** 1.5 + 100
        self.scat.set_array(data[3])
        return self.scat,

    def show(self):
        plt.show()


if __name__ == '__main__':
    a = AnimatedScatter()
    a.show()

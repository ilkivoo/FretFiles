import pyabf.plot
import numpy as np
import matplotlib.pyplot as plt
from numpy import median


# итоговый размер окна = 2 * window_size + 1
def median_filter(x, y, window_size):
    new_y = []
    y = np.array(y)
    for i in range(len(y)):
        window = []
        if i < window_size:
            window += (y[0] for _ in range(window_size - i))
        left = max(0, i - window_size)
        right = min(len(y), i + window_size + 1)
        window += y[left: right].tolist()
        if len(y) < i + window_size + 1:
            window += (y[len(y) - 1] for _ in range(window_size + i + 1 - len(y)))
        new_y.append(median(window))
    return x, new_y


# итоговый размер окна = 2 * window_size + 1
def moving_average_filter(x, y, window_size):
    new_y = []
    y = np.array(y)
    for i in range(len(y)):
        left = max(0, i - window_size)
        right = min(len(y), i + window_size + 1)
        window = y[left: right].tolist()
        new_y.append(np.average(window))
    return x, new_y


def exponential_smoothing(x, y, alpha):
    result = [y[0]]
    for n in range(1, len(y)):
        result.append(alpha * y[n] + (1 - alpha) * result[n - 1])
    return x, result


def double_exponential_smoothing(x, y, alpha, beta):
    result = [y[0]]
    level, trend = 0, 0
    for n in range(1, len(y)):
        if n == 1:
            level, trend = y[0], y[1] - y[0]
        if n >= len(y):
            value = result[-1]
        else:
            value = y[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return x, result


def draw(x, y, title):
    fig, ax = plt.subplots(1, 1)
    plt.title(title)
    ax.plot(x, y)
    plt.show()


if __name__ == "__main__":
    abf = pyabf.ABF("resources/data/18405003_cut.abf")
    x_mf, y_mf = median_filter(abf.sweepX, abf.sweepY, 100)
    x_maf, y_maf = moving_average_filter(abf.sweepX, abf.sweepY, 100)
    x_es, y_es = exponential_smoothing(abf.sweepX, abf.sweepY, 0.01)
    x_des, y_des = double_exponential_smoothing(abf.sweepX, abf.sweepY, 0.02, 0.02)
    draw(x_mf, y_mf, "median filter")
    draw(x_maf, y_maf, "moving average filter")
    draw(x_es, y_es, "exponential smoothing")
    print(f'{len(x_des)}  {len(y_des)}')
    draw(x_des, y_des, "double exponential smoothing")
    draw(abf.sweepX, abf.sweepY, 'raw')

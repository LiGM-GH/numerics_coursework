import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt

l = 1.0

h_t = 0.1
h_x = 0.1
h_y = 0.1
a = 2
b = 2
n = 10

fps = int(1 / h_t)
t_steps = int(pi * 2 * fps)


def f(x, y, t):
    global l
    return pi * pi / l / l * sin(pi * x / l) * sin(pi * y / l) * sin(pi * t / l)


def psi(x, y):
    global l
    return pi / l * sin(pi * x / l) * sin(pi * y / l)


def phi(x, y):
    return 0


C1 = h_x**2 * h_y**2
C2 = -(h_t**2) * h_y**2
C3 = -(h_t**2) * h_x**2
C4 = h_t**2 * h_x**2 * h_y**2
C0 = -2 * (C1 + C2 + C3)


def method_step(prev_matrix, curr_matrix, f, k):
    next_matrix = np.zeros(prev_matrix.shape)
    for i in range(1, next_matrix.shape[0] - 1):
        for j in range(1, next_matrix.shape[1] - 1):
            next_matrix[i][j] = -prev_matrix[i][j] - 1.0 / C1 * (
                C0 * curr_matrix[i][j]
                + C2 * (curr_matrix[i + 1][j] + curr_matrix[i - 1][j])
                + C3 * (curr_matrix[i][j + 1] + curr_matrix[i][j - 1])
                - C4 * f(j * h_x, i * h_y, (k - 1) * h_t)
            )
    return next_matrix


def show_plot(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_zlim(-1, 1)
    x, y = np.meshgrid(
        np.linspace(0, a, matrix.shape[0], endpoint=True),
        np.linspace(0, b, matrix.shape[1], endpoint=True),
    )
    ax.plot_surface(x, y, matrix, cmap="inferno", rstride=1, cstride=1)
    plt.show()
    return 0


x = np.linspace(0, a, n + 1, endpoint=True)
y = np.linspace(0, b, n + 1, endpoint=True)
x, y = np.meshgrid(x, y)
h_x = a / n
h_y = b / n
matrix = np.zeros((t_steps, n + 1, n + 1))

for i in range(1, n):
    for j in range(1, n):
        matrix[0][i][j] = phi(j * h_x, i * h_y)
        matrix[1][i][j] = h_t * psi(j * h_x, i * h_y) + matrix[0][i][j]

print(matrix[0])
print()

print(matrix[1])
print()

for k in range(2, t_steps):
    matrix[k] = method_step(matrix[k - 2], matrix[k - 1], f, k)
    # show_plot(matrix[k])

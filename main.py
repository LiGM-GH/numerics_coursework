import numpy as np
from numpy import sin, pi

l = 1.0


def y(t):
    global l
    return sin(pi * t / l)


def f(x, y, t):
    global l
    return pi * pi / l / l * sin(pi * x / l) * sin(pi * t / l) * sin(pi * y / l)


def phi(x, y):
    return 0


def psi(x, y):
    global l
    return pi / l * sin(pi * x / l) * sin(pi * y / l)


x_steps = 10
y_steps = 10
t_steps = 10

a = 1.0
b = 1.0
T = pi * 2.0

x_end = a
x_start = 0.0
y_end = b
y_start = 0.0
t_end = T
t_start = 0.0

h_x = (x_end - x_start) / x_steps
h_y = (y_end - y_start) / y_steps
h_t = 0.1

print(h_x, h_y, h_t)

C1 = h_x**2 * h_y**2
C2 = -(h_y**2) * h_t**2
C3 = -(h_x**2) * h_t**2
C4 = h_x**2 * h_y**2 * h_t**2
C0 = -2 * (C1 + C2 + C3)


def next_layer(k: int, matrix):
    for i in range(1, x_steps):
        for j in range(1, y_steps):
            matrix[i, j, k + 1] = -matrix[i, j, k - 1] - 1 / C1 * (
                C0 * matrix[i, j, k]
                + C2 * (matrix[i + 1, j, k] + matrix[i - 1, j, k])
                + C3 * (matrix[i, j + 1, k] + matrix[i, j - 1, k])
                - C4 * f(j * h_x, i * h_y, k * h_t)
            )


matrix = np.zeros((x_steps + 1, y_steps + 1, t_steps + 1))

for i in range(1, x_steps):
    for j in range(1, y_steps):
        matrix[i, j, 0] = phi(h_x * i, h_y * j)
        matrix[i, j, 1] = matrix[i, j, 0] + psi(h_x * i, h_y * j) * h_t

print(matrix[:, :, 0])
print()

print(matrix[:, :, 1])
print()

for i in range(1, t_steps):
    next_layer(i, matrix)
    print(matrix[:, :, i + 1])
print(matrix[:, :, t_steps])

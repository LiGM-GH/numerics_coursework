import numpy as np
from numpy import sin, pi

l = 1.0

def y(t):
    global l
    return sin(pi * t / l)

def f(x, y, t):
    global l
    return pi * pi / l / l *  sin(pi * x / l) * sin(pi * t / l)

def phi(x, y):
    return 0

def psi(x, y):
    global l
    return np.sin(np.pi * x / l) * np.sin(pi * y / l)

x_steps = 10
y_steps = 10
t_steps = 10

a = 1.0
b = 1.0
T = 5.0

x_end = a
x_start = 0.0
y_end = b
y_start = 0.0
t_end = T
t_start = 0.0

h_x = (x_end - x_start) / x_steps
h_y = (y_end - y_start) / y_steps
h_t = (t_end - t_start) / t_steps

matrix = np.full((x_steps + 1, y_steps + 1), 0.0)
next_matrix = np.full((x_steps + 1, y_steps + 1), 0.0)

for i in range(0, x_steps + 1):
    for j in range(0, y_steps + 1):
        matrix[i, j] = phi(h_x * i, h_y * j)
        next_matrix[i, j] = matrix[i, j] + psi(h_x * i, h_y * j) * h_t
print(next_matrix)

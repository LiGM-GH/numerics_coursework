import numpy as np
from numpy import sin, pi
import matplotlib.pyplot as plt


def f(x, y, t):
    return -sin(t)


def phi(x, y):
    return 0


def psi(x, y):
    return 1


def method_step(u0, u1, f, k):
    C1 = hx**2 * hy**2
    C2 = -(ht**2) * hy**2
    C3 = -(ht**2) * hx**2
    C4 = ht**2 * hx**2 * hy**2
    C0 = -2 * (C1 + C2 + C3)
    u2 = np.zeros(u0.shape)
    for i in range(1, u2.shape[0] - 1):
        for j in range(1, u2.shape[1] - 1):
            u2[i][j] = (
                -u0[i][j]
                - (
                    C0 * u1[i][j]
                    + C2 * (u1[i + 1][j] + u1[i - 1][j])
                    + C3 * (u1[i][j + 1] + u1[i][j - 1])
                    - C4 * f(j * hx, i * hy, (k - 1) * ht)
                )
                / C1
            )
    return u2


def show_plot(u_numeric):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_zlim(-1, 1)
    x, y = np.meshgrid(
        np.linspace(0, a, u_numeric.shape[0], endpoint=True),
        np.linspace(0, b, u_numeric.shape[1], endpoint=True),
    )
    ax.plot_surface(x, y, u_numeric, cmap="inferno", rstride=1, cstride=1)
    plt.show()
    return 0


ht = 0.1
hx = 0.1
hy = 0.1
a = 2
b = 2
n = 10

fps = int(1 / ht)
frn = int(pi * 2 * fps)

x = np.linspace(0, a, n + 1, endpoint=True)
y = np.linspace(0, b, n + 1, endpoint=True)
x, y = np.meshgrid(x, y)
hx = a / n
hy = b / n
u0 = np.zeros((frn, n + 1, n + 1))
for i in range(1, n):
    for j in range(1, n):
        u0[0][i][j] = phi(j * hx, i * hy)
        u0[1][i][j] = ht * psi(j * hx, i * hy) + u0[0][i][j]

for k in range(2, frn):
    u0[k] = method_step(u0[k - 2], u0[k - 1], f, k)
    show_plot(u0[k])

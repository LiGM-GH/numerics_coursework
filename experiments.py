import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from typing import Callable


def method_step(*, prev_matrix, curr_matrix, f, k, deltas_matrix, hx, hy, ht, real_y):
    C1 = hx**2 * hy**2
    C2 = -(ht**2) * hy**2
    C3 = -(ht**2) * hx**2
    C4 = ht**2 * hx**2 * hy**2
    C0 = -2 * (C1 + C2 + C3)
    next_matrix = np.zeros(prev_matrix.shape)
    for i in range(1, next_matrix.shape[0] - 1):
        for j in range(1, next_matrix.shape[1] - 1):
            next_matrix[i][j] = (
                -prev_matrix[i][j]
                - (
                    C0 * curr_matrix[i][j]
                    + C2 * (curr_matrix[i + 1][j] + curr_matrix[i - 1][j])
                    + C3 * (curr_matrix[i][j + 1] + curr_matrix[i][j - 1])
                    - C4 * f(j * hx, i * hy, (k - 1) * ht)
                )
                / C1
            )

            deltas_matrix[i][j] = abs(
                next_matrix[i][j] - real_y(j * hx, i * hy, (k - 1) * ht)
            )
    return next_matrix


def show_deltas(*, deltas_to_show, a, b):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # ax.set_zlim(-1, 1)
    x, y = np.meshgrid(
        np.linspace(0, a, deltas_to_show.shape[0], endpoint=True),
        np.linspace(0, b, deltas_to_show.shape[1], endpoint=True),
    )
    ax.plot_surface(x, y, deltas_to_show, cmap="inferno", rstride=1, cstride=1)
    plt.show()
    return 0


def show_plot(*, matrix_to_show, i, a, b, ht):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title(ht * i)
    ax.set_zlim(-1, 1)
    x, y = np.meshgrid(
        np.linspace(0, a, matrix_to_show.shape[0], endpoint=True),
        np.linspace(0, b, matrix_to_show.shape[1], endpoint=True),
    )
    ax.plot_surface(x, y, matrix_to_show, cmap="inferno", rstride=1, cstride=1)
    plt.show()
    return 0


def main(
    *,
    a: float,
    b: float,
    ht: float,
    n: int,
    real_y: Callable[[float, float, float], float],
    f: Callable[[float, float, float], float],
    phi: Callable[[float, float], float],
    psi: Callable[[float, float], float],
):
    fps = int(1 / ht)
    t_steps = int(pi * 2 * fps)

    x = np.linspace(0, a, n + 1, endpoint=True)
    y = np.linspace(0, b, n + 1, endpoint=True)
    x, y = np.meshgrid(x, y)
    hx = a / n
    hy = b / n
    matrix = np.zeros((t_steps, n + 1, n + 1))
    deltas = np.zeros((t_steps, n + 1, n + 1))
    for i in range(1, n):
        for j in range(1, n):
            matrix[0][i][j] = phi(j * hx, i * hy)
            matrix[1][i][j] = ht * psi(j * hx, i * hy) + matrix[0][i][j]
    # show_plot(matrix_to_show=matrix[0], ht=0, a=a, b=b, i=0)
    # show_plot(matrix_to_show=matrix[1], ht=ht, a=a, b=b, i=1)

    for k in range(2, t_steps):
        matrix[k] = method_step(
            prev_matrix=matrix[k - 2],
            curr_matrix=matrix[k - 1],
            f=f,
            k=k,
            deltas_matrix=deltas[k],
            hx=hx,
            hy=hy,
            ht=ht,
            real_y=real_y,
        )
        # show_plot(matrix_to_show=matrix[k], i=k, ht=ht, a=a, b=b)

    # for k in range(2, t_steps):
    #     show_deltas(deltas[k])

    # for i, matr in enumerate(deltas):
    #     print()
    #     print(f"{i}=====================================")
    #     for j, line in enumerate(matr):
    #         print(f"{j}:\t{line}")
    return matrix, deltas

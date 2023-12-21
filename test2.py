from experiments import main
from numpy import pi, sin
import numpy as np
from glob import glob


def phi(_x: float, _y: float):
    return 0


def psi(x: float, y: float):
    return abs(x - a / 2) * abs(y - a / 2) - a / 2


def f(x: float, y: float, t: float):
    return -abs(x - a / 2) * abs(y - a / 2) * sin(t)


def real_y(x: float, y: float, t: float):
    return abs(x - a / 2) * abs(y - a / 2) * sin(t)


if __name__ == "__main__":
    ht = 0.001

    a = 2
    b = 2
    n = 10

    filename=f"./test2_diff_with_ht_{ht}_n_{n}.npy"
    matrix_update = False
    file_present = glob(filename)

    if matrix_update or not file_present:
        matrix, matrix1 = main(
            ht=ht, a=a, b=a, n=n, f=f, phi=phi, psi=psi, real_y=real_y
        )
        np.save(filename, matrix1)
    else:
        matrix1 = np.load(filename)
    # print(matrix1)
    print(np.max(matrix1[:]))

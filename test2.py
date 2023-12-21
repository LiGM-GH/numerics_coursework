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
    matrix_update = False
    prev_update = False
    prev_filename = f"./test2_diff_with_ht_{ht * 2}_n_{n / 2}.npy"
    prev_matrix_filename = f"./test2_diff_with_ht_{ht * 2}_n_{n / 2}_matrix.npy"
    filename = f"./test2_diff_with_ht_{ht}_n_{n}.npy"
    matrix_filename = f"./test2_diff_with_ht_{ht}_n_{n}_matrix.npy"
    file_present = glob(filename) and glob(matrix_filename)
    prev_present = glob(prev_filename) and glob(prev_matrix_filename)

    if matrix_update or not file_present:
        matrix, matrix1 = main(
            ht=ht, a=2, b=2, n=n, f=f, phi=phi, psi=psi, real_y=real_y
        )
        np.save(filename, matrix1)
        np.save(matrix_filename, matrix)
    else:
        matrix1 = np.load(filename)
        matrix = np.load(matrix_filename)

    if prev_update or not prev_present:
        prev_matrix, prev_matrix1 = main(
            ht=ht * 2,
            a=2,
            b=2,
            n=int(n/2),
            f=f,
            phi=phi,
            psi=psi,
            real_y=real_y,
        )
        np.save(prev_filename, prev_matrix1)
        np.save(prev_matrix_filename, prev_matrix)
    else:
        prev_matrix1 = np.load(prev_filename)
        prev_matrix =  np.load(prev_matrix_filename)
    # print(matrix1)

    print(f"Runge: {np.max(matrix[:-1:2, ::2, ::2] - prev_matrix)}")

    print(np.max(matrix1[:]))

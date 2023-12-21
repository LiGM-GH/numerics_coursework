from experiments import main
from numpy import pi, sin
import numpy as np
from glob import glob

l = 1.0


def real_y(x, y, t):
    return sin(x * pi / l) * sin(y * pi / l) * sin(t * pi / l)


def f(x, y, t):
    return pi * pi / l / l * sin(pi * x / l) * sin(pi * y / l) * sin(pi * t / l)


def phi(x, y):
    return 0


def psi(x, y):
    return pi / l * sin(pi * x / l) * sin(pi * y / l)


if __name__ == "__main__":
    ht = 0.001
    n = 20
    matrix_update = False
    prev_update = False
    prev_filename = f"./test1_diff_with_ht_{ht * 2}_n_{n / 2}.npy"
    prev_matrix_filename = f"./test1_diff_with_ht_{ht * 2}_n_{n / 2}_matrix.npy"
    filename = f"./test1_diff_with_ht_{ht}_n_{n}.npy"
    matrix_filename = f"./test1_diff_with_ht_{ht}_n_{n}_matrix.npy"
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

from experiments import main
from numpy import pi, sin

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
    hx = 0.001
    hy = 0.001
    a = 2
    b = 2
    n = 10
    main(a=a, b=b, ht=ht, hx=hx, hy=hy, n=n, f=f, phi=phi, psi=psi, real_y=real_y)

from experiments import main
from numpy import pi, sin


def phi(_x: float, _y: float):
    return 0


def psi(x: float, y: float):
    return abs(x - a / 2) * abs(y - a / 2) - a / 2


def f(x: float, y: float, t: float):
    return -abs(x - a / 2) * abs(y - a / 2) * sin(t)


def real_y(x: float, y: float, t: float):
    return abs(x - a / 2) * abs(y - a / 2) * sin(t)


if __name__ == "__main__":
    ht = 0.01
    hx = 0.01
    hy = 0.01

    a = 2
    b = 2
    n = 10
    main(a=a, b=b, ht=ht, hx=hx, hy=hy, n=n, real_y=real_y, f=f, phi=phi, psi=psi)

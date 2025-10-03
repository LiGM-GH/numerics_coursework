# This is the README file

## The Problem
### Начально-краевая задача для  двумерного уравнения колебаний

2.1. Первая краевая задача для уравнения с постоянными коэффициентами. Схема с весами. Моделирование нестационарных процессов в зависимости от правой части уравнения.
$$d²u / dt² = Δu + f(x, y, t), 0 < x < a, 0 < y < b, t > 0$$
$$u(x, y, 0) = φ(x, y), du/dt = ψ(x, y), 0 < x < a, 0 < y < b$$
$$u|Г = 0, t >= 0$$

Let test problem be the following:
```math
f(x, y, t) = sin(x * π / l)*sin(y * π / l)*sin(t * π / l)
ψ(x, y) = 0
φ = sin(x)*sin(y)
```

Let another test problem be the following:
```math
u(x, y, t) = (|x-a/2|*|y-a/2|) * sin(t)
φ = 0
ψ = -|x-a/2|*|y-a/2|
f = -|x-a/2|*|y-a/2|sin(t)
a = b
```

# How to run
```
$ ./run.sh
```

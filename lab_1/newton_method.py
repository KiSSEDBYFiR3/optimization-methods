import math
import time
from matplotlib import pyplot as plt
import numpy as np

def f(x1, x2):
    return x1 + 2 * x2 + 4 * np.sqrt(1 + x1 ** 2 + x2 ** 2) - 4

def grad_f(x1, x2):
    df_dx1 = 1 + 4 * x1 / math.sqrt(1 + x1 ** 2 + x2 ** 2)
    df_dx2 = 2 + 4 * x2 / math.sqrt(1 + x1 ** 2 + x2 ** 2)
    return (df_dx1, df_dx2)

def hess_f(x1, x2):
    d2f_dx1dx1 = 4 * (2 * x1 ** 2 + x2 ** 2 + 1) / (1 + x1 ** 2 + x2 ** 2) ** (3/2)
    d2f_dx1dx2 = 4 * x1 * x2 / (1 + x1 ** 2 + x2 ** 2) ** (3/2)
    d2f_dx2dx2 = 4 * (x1 ** 2 + 2 * x2 ** 2 + 1) / (1 + x1 ** 2 + x2 ** 2) ** (3/2)
    return [[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx1dx2, d2f_dx2dx2]]

def newton_method(x0, eps):
    x_prev = x0
    iter_num = 0
    while True:
        df = grad_f(x_prev[0], x_prev[1])
        H = hess_f(x_prev[0], x_prev[1])
        d = [-df[0], -df[1]]
        S = np.linalg.solve(H, d)
        x_next = (x_prev[0] + S[0], x_prev[1] + S[1])
        iter_num += 1
        if abs(f(x_next[0], x_next[1]) - f(x_prev[0], x_prev[1])) < eps:
            break
        x_prev = x_next
    return x_next, f(x_next[0], x_next[1]), iter_num

x0 = (-2, -3)
eps = 0.01

start_time = time.time()

x_min, f_min, iterations = newton_method(x0, eps)
result1 = newton_method(x0, eps)
end_time = time.time()

x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-3, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2,np.log10(Z+1), cmap="coolwarm", alpha=0.8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$f(x_1, x_2)$")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

result1 = newton_method(x0, eps)
print("Метод Ньютона: ", result1, "Время: ", end_time - start_time, "Число итераций: ", iterations)

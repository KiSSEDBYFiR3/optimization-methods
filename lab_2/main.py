import numpy as np
import matplotlib.pyplot as plt

# Считываем данные из файла
with open('var_12.dat') as f:
    data = f.read().splitlines()

t = np.array(list(map(float, data[0].split(','))))
y = np.array(list(map(float, data[1].split(','))))

# Инициализация переменных
x = np.array([0, 0, 0, 0])
epsilon = 0.001
J = np.zeros((len(t), len(x)))
F = np.zeros(len(t))

# Определение модельной функции
def phi(x, t):
    return x[3] * t ** 3 + x[2] * t ** 2 + x[1] * t + x[0]

# Определение целевой функции
def f(x, t, y):
    return phi(x, t) - y

# Вычисление матрицы Якоби
def compute_jacobian(x, t):
    jacobian = np.zeros((len(t), len(x)))
    for i in range(len(t)):
        jacobian[i][0] = 1
        jacobian[i][1] = t[i]
        jacobian[i][2] = t[i] ** 2
        jacobian[i][3] = t[i] ** 3
    return jacobian

# Вычисление оптимального значения x методом Гаусса-Ньютона
while True:
    F = f(x, t, y)
    J = compute_jacobian(x, t)
    delta = np.linalg.lstsq(J, -F, rcond=None)[0]
    x = x + delta
    if np.linalg.norm(delta) < epsilon:
        break

# Вывод оптимального значения x и числа обусловленности матрицы Якоби
print("Оптимальное значение x: ", x)
print("Число обусловленности матрицы Якоби: ", np.linalg.cond(J))

# Построение графика модельной функции
plt.plot(t, y, 'o', label='Data')
t_new = np.linspace(t[0], t[-1], 100)
plt.plot(t_new, phi([np.sqrt(2), -3, 3*np.sqrt(2), 1], t_new), '-', label='Model')
plt.legend()
plt.show()
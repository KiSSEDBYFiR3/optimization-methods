import math
import time
from matplotlib import pyplot as plt
import numpy as np


# функция для вычисления значения f(x1, x2)
def f(x1, x2):
    return x1 + 2 * x2 + 4 * math.sqrt(1 + x1**2 + x2**2) - 4

# заданный интервал
x1_min, x1_max = -2, 2
x2_min, x2_max = -3, 1

# точность
epsilon = 0.01

# начальное значение минимума
f_min = float('inf')
x1_min_opt, x2_min_opt = None, None

start_time = time.time()

# перебор по всем точкам в интервале с заданным шагом
for x1 in range(int((x1_max - x1_min) / epsilon)):
    for x2 in range(int((x2_max - x2_min) / epsilon)):
        x1_val = x1_min + x1 * epsilon
        x2_val = x2_min + x2 * epsilon
        f_val = f(x1_val, x2_val)
        if f_val < f_min:
            f_min = f_val
            x1_min_opt, x2_min_opt = x1_val, x2_val

end_time = time.time()



print(f"Минимум функции: f({x1_min_opt}, {x2_min_opt}) = {f_min}")
print("Время выполения: ", end_time-start_time, "seconds")
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([0.017, 0.02, 0.025, 0.085, 0.087, 0.111, 0.119, 0.171, 0.174, 0.21, 0.211, 0.233, 0.783, 0.999, 1.11, 1.29, 1.32, 1.35, 1.69, 1.74, 2.75, 3.02, 3.04, 3.34, 4.09, 4.28, 4.29, 4.58, 4.68, 4.83, 5.3, 5.45, 5.48, 5.53, 5.96])
y = np.array([0.154, 0.181, 0.23, 0.26, 0.296, 0.357, 0.299, 0.334, 0.363, 0.428, 0.366, 0.537, 1.47, 0.771, 0.531, 0.87, 1.15, 2.48, 1.44, 2.23, 1.84, 2.01, 3.59, 2.83, 3.58, 3.28, 3.4, 2.96, 5.1, 4.66, 3.88, 3.52, 4.15, 6.94, 2.4])

x = np.log(x0)
x_sum = np.sum(x)
y_log = np.log(y)
y_log_sum = np.sum(y_log)
x_squared = np.square(x)
x_squared_sum = np.sum(x_squared)
xlny = x * np.log(y)
xlny_sum = np.sum(xlny)

print("Soma de x:", x_sum)
print("lny:", y_log)
print("Soma de lny^2:", y_log_sum)
print("x^2:", x_squared)
print("Soma de x^2:", x_squared_sum)
print("x*ln(y):", xlny)
print("Soma de x*ln(y):", xlny_sum)

a = (len(x)*xlny_sum - x_sum*y_log_sum)/(len(x)*x_squared_sum-x_sum**2)
print("a:", a)

lnb = (x_squared_sum*y_log_sum - xlny_sum*x_sum)/(len(x)*x_squared_sum - x_sum**2)
print("ln(b):", lnb)

b = np.exp(lnb)
print("b:", b)

print(f'lny = {lnb} + {a} lnx')
print(f'y = {b} * x^{a})')

def erro(x):
    y_calculado = b * np.power(x0, a)
    error = (y - y_calculado)**2
    erro_total = np.sum(error)
    return y_calculado, error, erro_total

y_values, errors, erro_total = erro(x0)

print("Valores de x:", x0)
print("Valores de y:", y_values)
print("Erros associados linearizado:", erro_total)

a_newton = 0.752647
b_newton = 1.18045

def erro_newton(x):
    y_calculado_newton = b_newton * np.power(x0, a_newton)
    error_newton = (y - y_calculado_newton)**2
    erro_total_newton = np.sum(error_newton)
    return y_calculado_newton, error_newton, erro_total_newton

y_calculado_newton, errors_newton, erro_total_newton = erro_newton(x0)

print("Erros associados newton:", erro_total_newton)


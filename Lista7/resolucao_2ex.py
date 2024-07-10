import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as spi
import matplotlib.pyplot as plt
import math

# Define o domínio
domain = [-1, 1, -1, 1]

# Define a solução exata uex
def uex(x, y):
    return (x + 1) * (y + 1) * np.arctan(x - 1) * np.arctan(y - 1)

# Define o gradiente de uex
def graduex(x, y):
    return np.array([
        -((1 + x) * (1 + y) * math.atan(1 - y) / (1 + (1 - x)**2)) +
            (1 + y) * math.atan(1 - x) * math.atan(1 - y),
        -((1 + x) * (1 + y) * math.atan(1 - x)) / (1 + (1 - y)**2)+(1+x)*math.atan(1-x)*math.atan(1-y)
    ])

# Define a base polinomial
def Basis(order):
    if order == 1:
        return [lambda x,y: np.array([(1 + x) * (1 + y)])]

    elif order == 2:
        return [
    lambda x, y: np.array([(1 + x) * (1 + y)]),
    lambda x, y: np.array([(1 + x) * y * (1 + y)]),
    lambda x, y: np.array([x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([x * (1 + x) * y * (1 + y)])
]
    elif order == 3:
        return [
    lambda x, y: np.array([(1 + x) * (1 + y)]),
    lambda x, y: np.array([(1 + x) * y * (1 + y)]),
    lambda x, y: np.array([0.5 * (1 + x) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([x * (1 + x) * y * (1 + y)]),
    lambda x, y: np.array([0.5 * x * (1 + x) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([0.5 * (1 + x) * (-1 + 3 * x**2) * (1 + y)]),
    lambda x, y: np.array([0.5 * (1 + x) * (-1 + 3 * x**2) * y * (1 + y)]),
    lambda x, y: np.array([0.25 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-1 + 3 * y**2)])
]
    elif order == 4:
        return [
    lambda x, y: np.array([(1 + x) * (1 + y)]),
    lambda x, y: np.array([(1 + x) * y * (1 + y)]),
    lambda x, y: np.array([0.5 * (1 + x) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([0.5 * (1 + x) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([x * (1 + x) * y * (1 + y)]),
    lambda x, y: np.array([0.5 * x * (1 + x) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([0.5 * x * (1 + x) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([0.5 * (1 + x) * (-1 + 3 * x**2) * (1 + y)]),
    lambda x, y: np.array([0.5 * (1 + x) * (-1 + 3 * x**2) * y * (1 + y)]),
    lambda x, y: np.array([0.25 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([0.25 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([0.5 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y)]),
    lambda x, y: np.array([0.5 * (1 + x) * (-3 * x + 5 * x**3) * y * (1 + y)]),
    lambda x, y: np.array([0.25 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([0.25 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-3 * y + 5 * y**3)])
]
    elif order == 5:
        return [
    lambda x, y: np.array([(1 + x) * (1 + y)]),
    lambda x, y: np.array([(1 + x) * y * (1 + y)]),
    lambda x, y: np.array([1/2 * (1 + x) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/2 * (1 + x) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/8 * (1 + x) * (1 + y) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([x * (1 + x) * y * (1 + y)]),
    lambda x, y: np.array([1/2 * x * (1 + x) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/2 * x * (1 + x) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/8 * x * (1 + x) * (1 + y) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-1 + 3 * x**2) * (1 + y)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-1 + 3 * x**2) * y * (1 + y)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/16 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-3 * x + 5 * x**3) * y * (1 + y)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/16 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([1/8 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y)]),
    lambda x, y: np.array([1/8 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * y * (1 + y)]),
    lambda x, y: np.array([1/16 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/16 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/64 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (3 - 30 * y**2 + 35 * y**4)])
]
    
# Define o gradiente da base
def gradBasis(order):
    if order == 1:
        return [lambda x, y: np.array([1 + y, 1 + x])]
    elif order == 2:
        return [
    lambda x, y: np.array([1 + y, 1 + x]),
    lambda x, y: np.array([y * (1 + y), (1 + x) * y + (1 + x) * (1 + y)]),
    lambda x, y: np.array([x * (1 + y) + (1 + x) * (1 + y), x * (1 + x)]),
    lambda x, y: np.array([x * y * (1 + y) + (1 + x) * y * (1 + y), x * (1 + x) * y + x * (1 + x) * (1 + y)])
]
    elif order == 3:
        return [
    lambda x, y: np.array([1 + y, 1 + x]),
    lambda x, y: np.array([y * (1 + y), (1 + x) * y + (1 + x) * (1 + y)]),
    lambda x, y: np.array([1/2 * (1 + y) * (-1 + 3 * y**2), 3 * (1 + x) * y * (1 + y) + 1/2 * (1 + x) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([x * (1 + y) + (1 + x) * (1 + y), x * (1 + x)]),
    lambda x, y: np.array([x * y * (1 + y) + (1 + x) * y * (1 + y), x * (1 + x) * y + x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([1/2 * x * (1 + y) * (-1 + 3 * y**2) + 1/2 * (1 + x) * (1 + y) * (-1 + 3 * y**2), 3 * x * (1 + x) * y * (1 + y) + 1/2 * x * (1 + x) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([3 * x * (1 + x) * (1 + y) + 1/2 * (-1 + 3 * x**2) * (1 + y), 1/2 * (1 + x) * (-1 + 3 * x**2)]),
    lambda x, y: np.array([3 * x * (1 + x) * y * (1 + y) + 1/2 * (-1 + 3 * x**2) * y * (1 + y), 1/2 * (1 + x) * (-1 + 3 * x**2) * y + 1/2 * (1 + x) * (-1 + 3 * x**2) * (1 + y)]),
    lambda x, y: np.array([3/2 * x * (1 + x) * (1 + y) * (-1 + 3 * y**2) + 1/4 * (-1 + 3 * x**2) * (1 + y) * (-1 + 3 * y**2), 3/2 * (1 + x) * (-1 + 3 * x**2) * y * (1 + y) + 1/4 * (1 + x) * (-1 + 3 * x**2) * (-1 + 3 * y**2)])
]
    elif order == 4:
        return [
    lambda x, y: np.array([1 + y, 1 + x]),
    lambda x, y: np.array([y * (1 + y), (1 + x) * y + (1 + x) * (1 + y)]),
    lambda x, y: np.array([1/2 * (1 + y) * (-1 + 3 * y**2), 3 * (1 + x) * y * (1 + y) + 1/2 * (1 + x) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/2 * (1 + y) * (-3 * y + 5 * y**3), 1/2 * (1 + x) * (1 + y) * (-3 + 15 * y**2) + 1/2 * (1 + x) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([x * (1 + y) + (1 + x) * (1 + y), x * (1 + x)]),
    lambda x, y: np.array([x * y * (1 + y) + (1 + x) * y * (1 + y), x * (1 + x) * y + x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([1/2 * x * (1 + y) * (-1 + 3 * y**2) + 1/2 * (1 + x) * (1 + y) * (-1 + 3 * y**2), 3 * x * (1 + x) * y * (1 + y) + 1/2 * x * (1 + x) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/2 * x * (1 + y) * (-3 * y + 5 * y**3) + 1/2 * (1 + x) * (1 + y) * (-3 * y + 5 * y**3), 1/2 * x * (1 + x) * (1 + y) * (-3 + 15 * y**2) + 1/2 * x * (1 + x) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([3 * x * (1 + x) * (1 + y) + 1/2 * (-1 + 3 * x**2) * (1 + y), 1/2 * (1 + x) * (-1 + 3 * x**2)]),
    lambda x, y: np.array([3 * x * (1 + x) * y * (1 + y) + 1/2 * (-1 + 3 * x**2) * y * (1 + y), 1/2 * (1 + x) * (-1 + 3 * x**2) * y + 1/2 * (1 + x) * (-1 + 3 * x**2) * (1 + y)]),
    lambda x, y: np.array([3/2 * x * (1 + x) * (1 + y) * (-1 + 3 * y**2) + 1/4 * (-1 + 3 * x**2) * (1 + y) * (-1 + 3 * y**2), 3/2 * (1 + x) * (-1 + 3 * x**2) * y * (1 + y) + 1/4 * (1 + x) * (-1 + 3 * x**2) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([3/2 * x * (1 + x) * (1 + y) * (-3 * y + 5 * y**3) + 1/4 * (-1 + 3 * x**2) * (1 + y) * (-3 * y + 5 * y**3), 1/4 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-3 + 15 * y**2) + 1/4 * (1 + x) * (-1 + 3 * x**2) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-3 + 15 * x**2) * (1 + y) + 1/2 * (-3 * x + 5 * x**3) * (1 + y), 1/2 * (1 + x) * (-3 * x + 5 * x**3)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-3 + 15 * x**2) * y * (1 + y) + 1/2 * (-3 * x + 5 * x**3) * y * (1 + y), 1/2 * (1 + x) * (-3 * x + 5 * x**3) * y + 1/2 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-3 + 15 * x**2) * (1 + y) * (-1 + 3 * y**2) + 1/4 * (-3 * x + 5 * x**3) * (1 + y) * (-1 + 3 * y**2), 3/2 * (1 + x) * (-3 * x + 5 * x**3) * y * (1 + y) + 1/4 * (1 + x) * (-3 * x + 5 * x**3) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-3 + 15 * x**2) * (1 + y) * (-3 * y + 5 * y**3) + 1/4 * (-3 * x + 5 * x**3) * (1 + y) * (-3 * y + 5 * y**3), 1/4 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-3 + 15 * y**2) + 1/4 * (1 + x) * (-3 * x + 5 * x**3) * (-3 * y + 5 * y**3)])
]
    elif order == 5:
        return [
    lambda x, y: np.array([1 + y, 1 + x]),
    lambda x, y: np.array([y * (1 + y), (1 + x) * y + (1 + x) * (1 + y)]),
    lambda x, y: np.array([1/2 * (1 + y) * (-1 + 3 * y**2), 
                            3 * (1 + x) * y * (1 + y) + 1/2 * (1 + x) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/2 * (1 + y) * (-3 * y + 5 * y**3), 
                            1/2 * (1 + x) * (1 + y) * (-3 + 15 * y**2) + 1/2 * (1 + x) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/8 * (1 + y) * (3 - 30 * y**2 + 35 * y**4), 
                            1/8 * (1 + x) * (1 + y) * (-60 * y + 140 * y**3) + 1/8 * (1 + x) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([x * (1 + y) + (1 + x) * (1 + y), 
                            x * (1 + x)]),
    lambda x, y: np.array([x * y * (1 + y) + (1 + x) * y * (1 + y), 
                            x * (1 + x) * y + x * (1 + x) * (1 + y)]),
    lambda x, y: np.array([1/2 * x * (1 + y) * (-1 + 3 * y**2) + 1/2 * (1 + x) * (1 + y) * (-1 + 3 * y**2), 
                            3 * x * (1 + x) * y * (1 + y) + 1/2 * x * (1 + x) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/2 * x * (1 + y) * (-3 * y + 5 * y**3) + 1/2 * (1 + x) * (1 + y) * (-3 * y + 5 * y**3), 
                            1/2 * x * (1 + x) * (1 + y) * (-3 + 15 * y**2) + 1/2 * x * (1 + x) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/8 * x * (1 + y) * (3 - 30 * y**2 + 35 * y**4) + 1/8 * (1 + x) * (1 + y) * (3 - 30 * y**2 + 35 * y**4), 
                            1/8 * x * (1 + x) * (1 + y) * (-60 * y + 140 * y**3) + 1/8 * x * (1 + x) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([3 * x * (1 + x) * (1 + y) + 1/2 * (-1 + 3 * x**2) * (1 + y), 
                            1/2 * (1 + x) * (-1 + 3 * x**2)]),
    lambda x, y: np.array([3 * x * (1 + x) * y * (1 + y) + 1/2 * (-1 + 3 * x**2) * y * (1 + y), 
                            1/2 * (1 + x) * (-1 + 3 * x**2) * y + 1/2 * (1 + x) * (-1 + 3 * x**2) * (1 + y)]),
    lambda x, y: np.array([3/2 * x * (1 + x) * (1 + y) * (-1 + 3 * y**2) + 1/4 * (-1 + 3 * x**2) * (1 + y) * (-1 + 3 * y**2), 
                            3/2 * (1 + x) * (-1 + 3 * x**2) * y * (1 + y) + 1/4 * (1 + x) * (-1 + 3 * x**2) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([3/2 * x * (1 + x) * (1 + y) * (-3 * y + 5 * y**3) + 1/4 * (-1 + 3 * x**2) * (1 + y) * (-3 * y + 5 * y**3), 
                            1/4 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-3 + 15 * y**2) + 1/4 * (1 + x) * (-1 + 3 * x**2) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([3/8 * x * (1 + x) * (1 + y) * (3 - 30 * y**2 + 35 * y**4) + 1/16 * (-1 + 3 * x**2) * (1 + y) * (3 - 30 * y**2 + 35 * y**4), 
                            1/16 * (1 + x) * (-1 + 3 * x**2) * (1 + y) * (-60 * y + 140 * y**3) + 1/16 * (1 + x) * (-1 + 3 * x**2) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-3 + 15 * x**2) * (1 + y) + 1/2 * (-3 * x + 5 * x**3) * (1 + y), 
                            1/2 * (1 + x) * (-3 * x + 5 * x**3)]),
    lambda x, y: np.array([1/2 * (1 + x) * (-3 + 15 * x**2) * y * (1 + y) + 1/2 * (-3 * x + 5 * x**3) * y * (1 + y), 
                            1/2 * (1 + x) * (-3 * x + 5 * x**3) * y + 1/2 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-3 + 15 * x**2) * (1 + y) * (-1 + 3 * y**2) + 1/4 * (-3 * x + 5 * x**3) * (1 + y) * (-1 + 3 * y**2), 
                            3/2 * (1 + x) * (-3 * x + 5 * x**3) * y * (1 + y) + 1/4 * (1 + x) * (-3 * x + 5 * x**3) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/4 * (1 + x) * (-3 + 15 * x**2) * (1 + y) * (-3 * y + 5 * y**3) + 1/4 * (-3 * x + 5 * x**3) * (1 + y) * (-3 * y + 5 * y**3), 
                            1/4 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-3 + 15 * y**2) + 1/4 * (1 + x) * (-3 * x + 5 * x**3) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/16 * (1 + x) * (-3 + 15 * x**2) * (1 + y) * (3 - 30 * y**2 + 35 * y**4) + 1/16 * (-3 * x + 5 * x**3) * (1 + y) * (3 - 30 * y**2 + 35 * y**4), 
                            1/16 * (1 + x) * (-3 * x + 5 * x**3) * (1 + y) * (-60 * y + 140 * y**3) + 1/16 * (1 + x) * (-3 * x + 5 * x**3) * (3 - 30 * y**2 + 35 * y**4)]),
    lambda x, y: np.array([1/8 * (1 + x) * (-60 * x + 140 * x**3) * (1 + y) + 1/8 * (3 - 30 * x**2 + 35 * x**4) * (1 + y), 
                            1/8 * (1 + x) * (3 - 30 * x**2 + 35 * x**4)]),
    lambda x, y: np.array([1/8 * (1 + x) * (-60 * x + 140 * x**3) * y * (1 + y) + 1/8 * (3 - 30 * x**2 + 35 * x**4) * y * (1 + y), 
                            1/8 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * y + 1/8 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y)]),
    lambda x, y: np.array([1/16 * (1 + x) * (-60 * x + 140 * x**3) * (1 + y) * (-1 + 3 * y**2) + 1/16 * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (-1 + 3 * y**2), 
                            3/8 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * y * (1 + y) + 1/16 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (-1 + 3 * y**2)]),
    lambda x, y: np.array([1/16 * (1 + x) * (-60 * x + 140 * x**3) * (1 + y) * (-3 * y + 5 * y**3) + 1/16 * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (-3 * y + 5 * y**3), 
                            1/16 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (-3 + 15 * y**2) + 1/16 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (-3 * y + 5 * y**3)]),
    lambda x, y: np.array([1/64 * (1 + x) * (-60 * x + 140 * x**3) * (1 + y) * (3 - 30 * y**2 + 35 * y**4) + 1/64 * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (3 - 30 * y**2 + 35 * y**4), 
                            1/64 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (1 + y) * (-60 * y + 140 * y**3) + 1/64 * (1 + x) * (3 - 30 * x**2 + 35 * x**4) * (3 - 30 * y**2 + 35 * y**4)])
]

def Stiff(gradPhi, domain):
    n = len(gradPhi)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            integrand = lambda y, x: np.dot(gradPhi[i](x, y), gradPhi[j](x, y))
            K[i, j] = spi.dblquad(integrand, domain[0], domain[1], lambda x: domain[2], lambda x: domain[3])[0]
   
    return K

def Load(Phi, domain, uex):
    n = len(Phi)
    F = np.zeros(n)
    
    for i in range(n):
        integrand = lambda y, x: Phi[i](x, y) * ((2 * (1 + x) * np.arctan(1 - x)) / (1 + (1 - y)**2) + (2 * (1 + x) * (1 - y) * (1 + y) * np.arctan(1 - x)) / ((1 + (1 - y)**2)**2) + (2 * (1 + y) * np.arctan(1 - y)) / (1 + (1 - x)**2) + (2 * (1 - x) * (1 + x) * (1 + y) * np.arctan(1 - y)) / ((1 + (1 - x)**2)**2))
        
        sum_integrand = spi.dblquad(integrand, domain[0], domain[1], lambda x: domain[2], lambda x: domain[3])[0]
        #print(f'sum_integrand = {sum_integrand}')
        
        fright = spi.quad(lambda y: graduex(1,y)[0] * Phi[i](1, y), -1, 1)[0]
        #print(f'fright = {fright}')
        ftop = spi.quad(lambda x: graduex(x,1)[1] * Phi[i](x, 1), -1, 1)[0]
        #print(f'ftop = {ftop}')
        
        F[i] = sum_integrand + fright + ftop
    
    return F

order = 4
base_aprox = Basis(order)
gradbasis_aprox = gradBasis(order)

Kij = Stiff(gradbasis_aprox, domain)
Fi = Load(base_aprox, domain, uex)
alpha_i = np.linalg.solve(Kij, Fi)

# Solução aproximada uh
def uh(x, y):
    return sum(alpha_i[i] * base_aprox[i](x, y) for i in range(len(base_aprox)))

def err(x, y):
    integrand = lambda x, y: (uex(x, y) - uh(x, y))**2
    result, _ = spi.dblquad(integrand, domain[0], domain[1], lambda x: domain[2], lambda x: domain[3])
    return np.sqrt(result)

# Define the grid
x = np.linspace(domain[0], domain[1], 100)
y = np.linspace(domain[2], domain[3], 100)
X, Y = np.meshgrid(x, y)
erro = err(x, y)
print(f'erro = {erro}')

Z = uh(X, Y)[0]
Z2 = uex(X,Y)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap='plasma', edgecolor='none')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='orange', edgecolor='none')
ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, color='blue', edgecolor='none')

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Solução')
#ax.set_title('3D Plot of the Approximate Solution')
plt.tight_layout()
#plt.savefig('/home/ingrid/Relatorios_Numerico/Lista7/Figuras/uex_uh5.pdf', format='pdf')
#plt.show()

def formatar_valor(valor):
    if np.abs(valor) < 1e-2 or np.abs(valor) >= 1e+2:
        return f"{valor:.2e}".replace('e', ' \\cdot 10^{') + "}"
    else:
        return f"{valor:.2f}"

# Geração da matriz Kij em LaTeX
latex_Kij = "\\begin{equation*}\\begin{bmatrix}\n"
for i in range(Kij.shape[0]):
    latex_Kij += " & ".join([formatar_valor(Kij[i, j]) for j in range(Kij.shape[1])])
    latex_Kij += " \\\\\n"
latex_Kij += "\\end{bmatrix}"
latex_Kij += "\\end{equation*}"

# Geração do vetor Fi em LaTeX
latex_Fi = "\\begin{equation*}\\begin{Bmatrix}\n"
latex_Fi += " \\\\\n".join([formatar_valor(Fi[i]) for i in range(Fi.shape[0])])
latex_Fi += " \\\\\n\\end{Bmatrix}"
latex_Fi += "\\end{equation*}"

# Escreve os resultados em um arquivo LaTeX
with open('/home/ingrid/Numerical_Methods/Lista7/matrix_vector_latex.txt', 'w') as file:
    file.write("Matriz Kij:\n")
    file.write(latex_Kij)
    file.write("\n\nVetor Fi:\n")
    file.write(latex_Fi)




import numpy as np
import scipy.integrate as spi

# Define o domínio
domain = np.array([[-1, -1], [1, 1]])

# Define a solução exata uex
def uex(x, y):
    return (x + 1) * (y + 1) * (np.sin(x) * np.cos(y))

# Define o gradiente de uex
def graduex(x, y):
    return np.array([
        np.cos(x) * np.cos(y) * (y + 1)*(1 + x) + np.sin(x) * np.cos(y) * (y + 1),
        -np.sin(x) * np.sin(y) * (x + 1) * (1 + y) + np.sin(x) * np.cos(y) * (x + 1)
    ])

# Define a base polinomial
def Basis(x, y):
    return (x + 1) * (y + 1) * np.sin(x) * np.cos(y)

# Define o gradiente da base
def gradBasis(x, y):
    return np.array([
        (1+x)*(1+y) * np.cos(x) * np.cos(y) + (1+y) * np.cos(y) * np.sin(x),
        (x + 1) * np.cos(y) * np.sin(x) - (x + 1) * (1 + y) * np.sin(x) * np.sin(y)
    ])

# def Stiff(gradPhi, domain):
#     n = len(gradPhi)
#     K = np.zeros((n, n))
    
#     # Número de pontos para integração
#     num_points = 100
#     x_vals = np.linspace(domain[0, 0], domain[1, 0], num_points)
#     y_vals = np.linspace(domain[0, 1], domain[1, 1], num_points)
#     dx = x_vals[1] - x_vals[0]
#     dy = y_vals[1] - y_vals[0]
    
#     for i in range(n):
#         for j in range(n):
#             sum_integrand = 0.0
#             for x in x_vals:
#                 for y in y_vals:
#                     grad_i = gradPhi[i](x, y)
#                     grad_j = gradPhi[j](x, y)
#                     integrand = np.dot(grad_i, grad_j)
#                     sum_integrand += integrand * dx * dy
#             K[i, j] = sum_integrand
    
#     return K

# def Stiff(gradPhi, domain):
#     n = len(gradPhi)
#     K = np.zeros((n, n))
    
#     for i in range(n):
#         for j in range(n):
#             integrand = lambda x, y: np.dot(gradPhi[i](x, y), gradPhi[j](x, y))
#             K[i, j] = spi.dblquad(integrand, domain[0, 0], domain[1, 0], lambda x: domain[0, 1], lambda x: domain[1, 1])[0]
    
#     return K

def Stiff(gradPhi, domain):
    n = len(gradPhi)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            integrand = lambda x, y: np.dot(gradPhi[i](x, y), gradPhi[j](x, y))
            K[i, j] = spi.dblquad(integrand, domain[0, 0], domain[1, 0], lambda x: domain[0, 1], lambda x: domain[1, 1])[0]
    # print(domain[0, 0])
    # print(domain[1, 0])
    # print(domain[0, 1])
    # print(domain[1, 1])
    return K


# def Load(Phi, domain, uex):
#     n = len(Phi)
#     F = np.zeros(n)
    
#     # Definindo o número de pontos para integração
#     num_points = 100
#     x_vals = np.linspace(domain[0, 0], domain[1, 0], num_points)
#     y_vals = np.linspace(domain[0, 1], domain[1, 1], num_points)
#     dx = x_vals[1] - x_vals[0]
#     dy = y_vals[1] - y_vals[0]
    
#     for i in range(n):
#         sum_integrand = 0.0
#         for x in x_vals:
#             for y in y_vals:
#                 Phi_i = Phi[i](x, y)
#                 b = -2 * (1+y) * np.cos(x) * np.cos(y) + 2 * (1+x) * (1+y) * np.cos(y) * np.sin(x) + 2 * (1+x) * np.sin(x) * np.sin(y)
#                 integrand = Phi_i * b
#                 sum_integrand += integrand * dx * dy
        
#         # Integrando as condições de contorno
#         # rightBC = graduex(1, 0)  # Gradiente em x = 1
#         # topBC = graduex(0, 1)    # Gradiente em y = 1
#         #rightBC = (1 + x) * (1 + y) * np.cos(x) * np.cos(y) + (1 + y) * np.cos(y) * np.sin(x)
#         rightBC = (1 + 1) * (1 + 0) * np.cos(1) * np.cos(0) + (1 + 0) * np.cos(0) * np.sin(1)
#         print(f'rightBC = {rightBC}')
#         # topBC = (1 + x) * np.cos(y) * np.sin(x) - (1 + x) *(1+y)* np.sin(x) * np.sin(y)
#         topBC = (1 + 0) * np.cos(1) * np.sin(0) - (1 + 0) *(1+1)* np.sin(0) * np.sin(1)
#         print(f'topBC = {topBC}')
        
#         fright = 0.0
#         ftop = 0.0
        
#         for x in x_vals:
#             fright += np.dot(rightBC, Phi[i](x, 1)) * dx
#         for y in y_vals:
#             ftop += np.dot(topBC, Phi[i](1, y)) * dy
        
#         F[i] = sum_integrand + fright + ftop
    
#     return F

def Load(Phi, domain, uex):
    n = len(Phi)
    F = np.zeros(n)
    
    for i in range(n):
        integrand = lambda x, y: Phi[i](x, y) * (-2 * (1+y) * np.cos(x) * np.cos(y) + 2 * (1+x) * (1+y) * np.cos(y) * np.sin(x) + 2 * (1+x) * np.sin(x) * np.sin(y))
        sum_integrand = spi.dblquad(integrand, domain[0, 0], domain[1, 0], lambda x: domain[0, 1], lambda x: domain[1, 1])[0]
        
        rightBC = (graduex(1, 0))[0]
        print(f'rightBC = {rightBC}')
        topBC = (graduex(0, 1))[1]
        print(f'topBC = {topBC}')
        
        fright = spi.quad(lambda y: rightBC * Phi[i](1, y), -1, 1)[0]
        ftop = spi.quad(lambda x: topBC * Phi[i](x, 1), -1, 1)[0]
        
        F[i] = sum_integrand + fright + ftop
    
    return F

Basis = [Basis]
gradBasis = [gradBasis]

Kij = Stiff(gradBasis, domain)
Fi = Load(Basis, domain, uex)
alpha_i = np.linalg.solve(Kij, Fi)
print(f'Kij = {Kij}')
print(f'Fi = {Fi}')
print(f'alpha_i = {alpha_i}')

# Solução aproximada uh
def uh(x, y):
    return np.dot(alpha_i, Basis[0](x, y))

# Cálculo do erro - colocar a integral
# def err(x, y):
#     return np.sqrt((uex(x, y) - uh(x, y))**2)
def err(x, y):
    integrand = lambda x, y: (uex(x, y) - uh(x, y))**2
    result, _ = spi.dblquad(integrand, domain[0, 0], domain[1, 0], lambda x: domain[0, 1], lambda x: domain[1, 1])
    return np.sqrt(result)

# Teste
x = 0
y = 0
print("Solução aproximada uh(x, y):", uh(x, y))
print("Erro em (x, y):", err(x, y))

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jacfwd

# Sistema de Equações Não Lineares
def eq1(x1, x2, x3):
    return -x1 * jnp.cos(x2) - 1

def eq2(x1, x2, x3):
    return x1 * x2 + x3 - 2

def eq3(x1, x2, x3):
    return jnp.exp(-x3) * jnp.sin(x1 + x2) + x1**2 + x2**2 - 1

# Vetor das Funções
def F(X):
    x1, x2, x3 = X
    return jnp.array([eq1(x1, x2, x3), eq2(x1, x2, x3), eq3(x1, x2, x3)])

def newton_method(F, jacobian, X0, tol=1e-15, max_iter=100):
    X = np.array(X0, dtype=float)
    errors = []
    print("Newton Method")
    print("N° de Iterações.\t\tSolução\t\tErro")
    for i in range(max_iter):
        J = jacobian(X)
        FX = F(X)

        delx = np.linalg.solve(J, -FX)
        X += delx

        error = np.linalg.norm(F(X))
        errors.append(error)

        if error < 1e-2 or error >= 1e+2 or (1e-2 <= error < 1 and 'e' in f"{error:.4e}"):
            formatted_error = f"${error:.4e}".replace('e', ' \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
        else:
            formatted_error = f"{error:.4f}"
        print(f"{len(errors)}&{X}&{formatted_error}\\\\")

        if error < tol:
            break
        
    return np.array(X), errors

def broyden_method(F, jacobian, X0, tol=1e-20, max_iter=100):
    X = np.array(X0, dtype=float)
    J = jacobian(X)
    errors = []
    xlist = [X.copy()]
    
    print("Metodo de Broyden")
    print("N° de Iterações.\t\tSolução\t\tErro")

    for _ in range(max_iter):
        FX = F(X)
        delx = np.linalg.solve(J, -FX)
        X_new = X + delx

        error = np.linalg.norm(F(X_new))
        errors.append(error)
        xlist.append(X_new.copy())

        if error < tol:
            break

        delG = F(X_new) - FX

        J += np.outer((delG - J @ delx), delx) / np.dot(delx, delx)

        X = X_new

        if error < 1e-2 or error >= 1e+2 or (1e-2 <= error < 1 and 'e' in f"{error:.4e}"):
            formatted_error = f"${error:.4e}".replace('e', ' \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
        else:
            formatted_error = f"{error:.4f}"
        print(f"{len(errors)}&{X}&{formatted_error}\\\\")

    return np.array(xlist), errors

# Ponto inicial para ambos os métodos
x0 = np.array([0.1, 0.1, 0.1])
X0_newton = [0.1, 0.1, 0.1]

# Jacobiano usando JAX
jacobian_jax = jacfwd(F)

xlist_secante, diff_secante = broyden_method(F, jacobian_jax, x0)

try:
    sol_newton, diff_newton = newton_method(F, jacobian_jax, X0_newton)
    print("Método de Newton convergiu.")
except ValueError as e:
    print(e)
    sol_newton = X0_newton
    diff_newton = []

print("Último ponto encontrado usando o método da secante:", xlist_secante[-1])
print("Diferenças usando o método da secante:", diff_secante)
ratios_secante = [np.log(diff_secante[i]) / np.log(diff_secante[i - 1]) for i in range(1, len(diff_secante))]
print("Razão entre as diferenças usando o método da secante:", ratios_secante)

print("\nMétodo de Newton:")
print("Solução:")
for sol in sol_newton:
    print(f"{sol:.10f}", end = " ")
print()
print("Iterações:", len(diff_newton))

iterations_secante = range(len(diff_secante))
iterations_newton = np.arange(len(diff_newton))

plt.plot(iterations_secante, diff_secante, label='Método de Broyden', marker='o', linestyle='-')
plt.plot(iterations_newton, diff_newton, label='Método de Newton', marker='x', linestyle='-')
plt.yscale('log')
plt.xlabel('Número de Iterações')
plt.ylabel('Erro')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('/home/ingrid/Relatorios_Numerico/Lista5/Figuras/sist2_jax.pdf', format='pdf')
plt.show()

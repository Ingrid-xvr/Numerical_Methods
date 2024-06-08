import numpy as np
import matplotlib.pyplot as plt

# Funções do sistema não linear
def f1(x, y):
    return np.exp(x * y) + x**2 + y - 1.2

def f2(x, y):
    return x**2 + y**2 + x - 0.55

# Vetor das funções
def F(X):
    x, y = X
    return np.array([f1(x, y), f2(x, y)])

# Matriz Jacobiana 
def jacobian(X):
    x, y = X
    df1dx = y * np.exp(x * y) + 2 * x
    df1dy = x * np.exp(x * y) + 1
    df2dx = 2 * x + 1
    df2dy = 2 * y
    return np.array([[df1dx, df1dy], [df2dx, df2dy]])

# Método de Newton
def newton_method(F, jacobian, X0, tol=1e-15, max_iter=100):
    X = np.array(X0, dtype=float)
    errors = []
    print("Metodo de Newton")
    print("N° de Iterações.\t\tSolução\t\tErro")
    for i in range(max_iter):
        J = jacobian(X)
        FX = F(X)
        delta = np.linalg.solve(J, -FX)
        X += delta
        error = np.linalg.norm(F(X))
        errors.append(error)
        #print(f"{len(errors)}&{X}&{errors[i]:.4e}{chr(92)*2}")
        #print(f"{len(errors)}&{X}&{errors[i]:.4e}".replace('e+00', '').replace('e', '$ \\cdot 10^{') + '}$\\\\')

        if error < 1e-2 or error >= 1e+2 or (1e-2 <= error < 1 and 'e' in f"{error:.4e}"):
            formatted_error = f"${error:.4e}".replace('e', ' \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
        else:
            formatted_error = f"{error:.4f}"
        print(f"{len(errors)}&{X}&{formatted_error}\\\\")

        if error < tol:
            break
       
    return X, errors
    


# Método da Broyden
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
x0 = np.array([3, 7])
X0_newton = [3, 7]

# x0 = np.array([0.1, 0.1])
# X0_newton = [0.1, 0.1]

xlist_secante, diff_secante = broyden_method(F, jacobian, x0)

sol_newton, errors_newton = newton_method(F, jacobian, X0_newton)

print("Último ponto encontrado usando o método da secante:", xlist_secante[-1])
print("Diferenças usando o método da secante:", diff_secante)
ratios_secante = [np.log(diff_secante[i]) / np.log(diff_secante[i - 1]) for i in range(1, len(diff_secante))]
print("Razão entre as diferenças usando o método da secante:", ratios_secante)

print("\nMétodo de Newton:")
print("Solução:", sol_newton)
print("Iterações:", len(errors_newton))

iterations_secante = range(len(diff_secante))
iterations_newton = np.arange(len(errors_newton))

plt.plot(iterations_secante, diff_secante, label='Método de Broyden', marker='o', linestyle='-')
plt.plot(iterations_newton, errors_newton, label='Método deNewton', marker='x', linestyle='-')
plt.yscale('log')
plt.xlabel('Número de Iterações')
plt.ylabel('Erro')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig('/home/ingrid/Relatorios_Numerico/Lista5/Figuras/sist1_3_7.pdf', format='pdf')
plt.show()



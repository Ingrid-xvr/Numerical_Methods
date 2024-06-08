import numpy as np
import matplotlib.pyplot as plt

# Sistema de Equações Não Lineares
def eq1(x1, x2, _):
    return -x1 * np.cos(x2) - 1

def eq2(x1, x2, x3):
    return x1 * x2 + x3 - 2

def eq3(x1, x2, x3):
    return np.exp(-x3) * np.sin(x1 + x2) + x1**2 + x2**2 - 1

# Vetor das Funções
def F(X):
    x1, x2, x3 = X
    return np.array([eq1(x1, x2, x3), eq2(x1, x2, x3), eq3(x1, x2, x3)])

# Matriz Jacobiana
def jacobian(x):
    x1, x2, x3 = x
    return np.array([
        [-np.cos(x2), x1 * np.sin(x2), 0],
        [x2, x1, 1],
        [np.exp(-x3) * np.cos(x1 + x2) + 2 * x1, np.exp(-x3) * np.cos(x1 + x2) + 2 * x2, -np.exp(-x3) * np.sin(x1 + x2)]
    ])

# Método de Newton
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

        #print(f"{len(errors)}&{X}&{errors[i]:.4e}{chr(92)*2}")
        #print(f"{len(errors)}&{X}&{errors[i]:.4e}".replace('e+00', '').replace('e', '$ \\cdot 10^{') + '}$\\\\')

        # if error < 1e-2 or error >= 1e+2 or (1e-2 <= error < 1 and 'e' in f"{error:.4e}"):
        #     formatted_error = f"{error:.4e}".replace('e', '$ \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
        # else:
        #     formatted_error = f"{error:.4f}"
        # print(f"{len(errors)}&{X}&{formatted_error}\\\\")

        if error < 1e-2 or error >= 1e+2 or (1e-2 <= error < 1 and 'e' in f"{error:.4e}"):
            formatted_error = f"${error:.4e}".replace('e', ' \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
        else:
            formatted_error = f"{error:.4f}"
        print(f"{len(errors)}&{X}&{formatted_error}\\\\")

        if error < tol:
            break
        
    return np.array(X), errors

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

def main():
    # Ponto inicial para ambos os métodos
    #x0 = np.array([0.1, 0.1, 0.1])
    x0 = np.array([3, 7, 10])

    xlist_broyden, diff_broyden = broyden_method(F, jacobian, x0)

    sol_newton, diff_newton = newton_method(F, jacobian, x0)

    print(f"Solucao de Broyden: {xlist_broyden[-1]}")
    print(f"Diferencas de Broyden: {diff_broyden}")
    ratios_broyden = [np.log(diff_broyden[i]) / np.log(diff_broyden[i - 1]) for i in range(1, len(diff_broyden))]
    print("Razão entre as diferenças usando o método de Broyden:", ratios_broyden)

    print("\nMétodo de Newton:")
    print("Solução:")
    for sol in sol_newton:
        print(f"{sol:.10f}", end=" ")
    print()
    print("Iterações:", len(diff_newton))

    iterations_secante = range(len(diff_broyden))
    iterations_newton = np.arange(len(diff_newton))

    plt.plot(iterations_secante, diff_broyden, label='Método de Broyden', marker='o', linestyle='-')
    plt.plot(iterations_newton, diff_newton, label='Método de Newton', marker='x', linestyle='-')
    plt.yscale('log')
    plt.xlabel('Número de Iterações')
    plt.ylabel('Erro')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig('/home/ingrid/Relatorios_Numerico/Lista5/Figuras/sist2_3_7_10.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    main()

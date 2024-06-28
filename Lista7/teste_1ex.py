import numpy as np
import matplotlib.pyplot as plt

def fty(t, y):
    return 1 + (t - y)**2

def exact_solution(t): 
    return (-1 - t + t**2) / (-1 + t)

def RungeKuttaButcher(fty, t0, tf, dt, y0):
    # Coeficientes do Butcher Tableau
    a = np.array([
        [0, 0, 0, 0],
        [1/3, 0, 0, 0],
        [-1/3, 1, 0, 0],
        [1, -1, 1, 0]
    ])
    b = np.array([1/8, 3/8, 3/8, 1/8])
    c = np.array([0, 1/3, 2/3, 1])
    
    t = t0
    w = y0
    sol = [[t, w]]
    n = int((tf - t0) / dt)
    
    for i in range(1, n + 1):
        K = np.zeros(4)
        K[0] = dt * fty(t + c[0]*dt, w)
        K[1] = dt * fty(t + c[1]*dt, w + a[1,0]*K[0])
        K[2] = dt * fty(t + c[2]*dt, w + a[2,0]*K[0] + a[2,1]*K[1])
        K[3] = dt * fty(t + c[3]*dt, w + a[3,0]*K[0] + a[3,1]*K[1] + a[3,2]*K[2])
        
        w = w + np.dot(b, K)
        t = t + dt
        
        sol.append([t, w])
    
    return sol

# Condições iniciais
t0 = 2.0
tf = 3.0
dt = 0.1
y0 = 1.0  # Valor inicial de y(t0)

sol = RungeKuttaButcher(fty, t0, tf, dt, y0)

# Solução exata para os mesmos tempos discretos para comparar
t_values = [t for t, y in sol]
exact_values = exact_solution(np.array(t_values))

print(f"{'Iteração'} {'t'} {'w'} {'Solução Exata'} {'Erro'}")
errors = []
for i, (t, w) in enumerate(sol):
    exact = exact_solution(t)
    error = exact - w
    errors.append(error)
    print(f"{i} {t:.2f} {w:.6f} {exact:.6f} {error:.6e}")

iterations = np.arange(len(sol))

plt.figure(figsize=(10, 6))

if all(error > 0 for error in errors):
    plt.yscale('log')
else:
    plt.yscale('linear')

plt.plot(iterations, errors, marker='o', linestyle='-', color='b', label='Erro (Exato - Runge-Kutta Butcher)')
plt.title('Erro entre a solução exata e a solução numérica (Runge-Kutta Butcher)')
plt.xlabel('Iterações')
plt.ylabel('Erro')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/ingrid/Relatorios_Numerico/Lista7/Figuras/erro.pdf', format='pdf')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t_values, [y for t, y in sol], label='Runge-Kutta Butcher (numérico)', marker='o')
plt.plot(t_values, exact_values, label='Exato', linestyle='--')
plt.title('Comparação da solução numérica com a solução exata')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/ingrid/Relatorios_Numerico/Lista7/Figuras/solucao.pdf', format='pdf')
plt.show()

def formatar_taxa_ou_erro(valor):
    if (valor < 1e-2 or valor >= 1e+2 or (1e-2 <= valor < 1 and 'e' in f"{valor:.4e}")) and not (1e-1 <= abs(valor) < 1e+1):
        formatted_valor = f"${valor:.4e}".replace('e', ' \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
    else:
        formatted_valor = f"{valor:.4f}"
    return formatted_valor

table_tex = []
table_tex.append(f"\\begin{{table}}[H]")
table_tex.append(f"\\centering")
table_tex.append(f"\caption{{Legenda.}}")
table_tex.append(f"\\label{{tab:}}")
table_tex.append(f"\\begin{{tabular}}{{|ccccc|}}")
table_tex.append(f"\\hline")
table_tex.append(f"Iteração & t & w & Solução Exata & Erro \\\\")
table_tex.append(f"\\hline")
for i, (t, w) in enumerate(sol):
    exact = exact_solution(t)
    error = exact - w
    error_formatted = formatar_taxa_ou_erro(error)
    table_tex.append(f"{i} & {t:.2f} & {w:.6f} & {exact:.6f} & {error_formatted} \\\\")
table_tex.append(f"\\hline")
table_tex.append(f"\\end{{tabular}}")
table_tex.append(f"\\end{{table}}")

table_str = "\n".join(table_tex)

with open('/home/ingrid/Numerical_Methods/Lista7/table.txt', 'w') as file:
    file.write(table_str)

import numpy as np
import matplotlib.pyplot as plt

# Função para formatar a taxa de convergência e o erro
def formatar_taxa_ou_erro(valor):
    if (valor < 1e-2 or valor >= 1e+2 or (1e-2 <= valor < 1 and 'e' in f"{valor:.4e}")) and not (1e-1 <= abs(valor) < 1e+1):
        formatted_valor = f"${valor:.4e}".replace('e', ' \\cdot 10^{').replace('+0', '+').replace('-0', '-') + '}$'
    else:
        formatted_valor = f"{valor:.4f}"
    return formatted_valor

# Função power_method_deflation modificada
def power_method_deflation(A, tol=1e-13, max_iterations=200):
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    errors = []
    taxa_convergencia = []

    B = np.copy(A)

    # Lista de strings para montar a tabela em LaTeX
    table_str = []

    for eig_idx in range(n):
        v = np.random.uniform(-1, 1, n)
        v = v / np.linalg.norm(v)
        eigenvalue = 0.0

        # Para facilitar a visualização
        table = []

        error_for_eigenvalue = []

        for iteration in range(max_iterations):
            w = np.dot(B, v)
            new_eigenvalue = np.linalg.norm(w)
            sign = np.dot(v, w)
            if sign < 0:
                new_eigenvalue *= -1
            v = w / np.linalg.norm(w)
            error = np.abs(new_eigenvalue - eigenvalue) / np.abs(eigenvalue)
            #error = np.abs(new_eigenvalue - eigenvalue)
            error_for_eigenvalue.append(error)

            if error < tol:
                break

            eigenvalue = new_eigenvalue

            # Taxa de convergência
            if iteration > 0:
                taxa_conv = (np.log(error_for_eigenvalue[iteration - 1]) - np.log(error_for_eigenvalue[iteration])) if iteration < len(error_for_eigenvalue) else 0
            else:
                taxa_conv = 0

            # Adiciona os dados da iteração à tabela
            table.append((iteration + 1, eigenvalue, error, taxa_conv))

        errors.append(error_for_eigenvalue)

        if len(error_for_eigenvalue) > 1:
            taxa_autovalor = [(np.log(error_for_eigenvalue[j]) - np.log(error_for_eigenvalue[j + 1])) / ((j) - (j + 1)) for j in range(1, len(error_for_eigenvalue)-1)]
            taxa_convergencia.append(taxa_autovalor)
        else:
            taxa_convergencia.append([])

        # Monta a tabela em LaTeX para o autovalor atual
        table_tex = []
        table_tex.append(f"\\begin{{tabular}}{{|cccc|}}")
        table_tex.append(f"\\hline")
        table_tex.append(f"Iteração & Autovalor & Erro & Taxa de Convergência \\\\")
        table_tex.append(f"\\hline")
        for iteration, eigenvalue, error, taxa_conv in table:
            error_formatted = formatar_taxa_ou_erro(error)
            taxa_conv_formatted = formatar_taxa_ou_erro(taxa_conv)
            table_tex.append(f"{iteration} & {eigenvalue:.4f} & {error_formatted} & {taxa_conv_formatted} \\\\")
        table_tex.append(f"\\hline")
        table_tex.append(f"\\end{{tabular}}")

        # Adiciona a tabela formatada em LaTeX à lista
        table_str.append("\n".join(table_tex))

        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)

        # Deflação
        B = B - eigenvalue * np.outer(v, v)

    # Imprime todas as tabelas em LaTeX geradas
    for idx, tex in enumerate(table_str):
        print(f"Tabela de iterações para o autovalor {idx + 1}:")
        print(tex)
        print()
    with open('/home/ingrid/Numerical_Methods/Lista6/tabelas_autovalores.txt', 'w') as file:
        for eig_idx, table in enumerate(table_str):
            file.write(f"\\noindent\\textbf{{Autovalor {eig_idx + 1}:}}\n\n{table}\n\n")

    # Plotar gráfico de erro para cada autovalor
    for idx, error_list in enumerate(errors):
        plt.plot(range(1, len(error_list) + 1), error_list, label=f'Autovalor {idx + 1}')

    plt.yscale('log')  # Escala logarítmica no eixo y para melhor visualização
    plt.xlabel('Iteração')
    plt.ylabel('Erro')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('/home/ingrid/Relatorios_Numerico/Lista6/Figuras/erro_iteracao.pdf', format='pdf')
    plt.show()

    return eigenvalues, eigenvectors, errors, taxa_convergencia

# Exemplo de uso
if __name__ == "__main__":
    A = np.array([[815.732, 418.604, 0, 420.445, -1., 281.869, 232.076, 397.658, 65.1193, 0, 421.275, 2.34737, 0.0571453, -0.139624, 419.713],
                  [418.604, 520.872, 0, 519.833, -1., 488.09, 334.978, 527.169, -9.12066, 0, 520.524, 2.60818, -1.5467, -0.182023, 519.888],
                  [0, 0, 0, 0, 0, 0, -1., -1., 1., 1., 0, 0, 0, 0, 0],
                  [420.445, 519.833, 0, 3.33333e10, -1., 486.144, 336.579, 528.043, -11.6014, 0, 519.504, 2.57783, -0.263573, -0.877493, 1.66667e10],
                  [-1., -1., 0, -1., 0, -1., -1., -1., 0, 0, -1., 0, 0, 0, -1.],
                  [281.869, 488.09, 0, 486.144, -1., 948.696, 348.62, 565.821, -86.6241, 0, 484.754, 2.56467, -1.13458, 0.0776692, 486.978],
                  [232.076, 334.978, -1., 336.579, -1., 348.62, -61966.4, -30722.5, 106707., 0, 337.077, 2.02037, 0.489366, 0.555045, 335.855],
                  [397.658, 527.169, -1., 528.043, -1., 565.821, -30722.5, -8229.76, 53353.6, 0, 527.669, 2.96425, 0.437361, 1.49892, 527.173],
                  [65.1193, -9.12066, 1., -11.6014, 0, -86.6241, 106707., 53353.6, -160083., 0, -12.3841, -1.09425, -2.00416, -1.09425, -9.90337],
                  [0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [421.275, 520.524, 0, 519.504, -1., 484.754, 337.077, 527.669, -12.3841, 0, 520.207, 2.60707, 0.469261, -0.183137, 519.552],
                  [2.34737, 2.60818, 0, 2.57783, 0, 2.56467, 2.02037, 2.96425, -1.09425, 0, 2.60707, -0.317777, -0.290078, 0.0278837, 3.30254],
                  [0.0571453, -1.5467, 0, -0.263573, 0, -1.13458, 0.489366, 0.437361, -2.00416, 0, 0.469261, -0.290078, -0.580992, -0.290078, -0.813864],
                  [-0.139624, -0.182023, 0, -0.877493, 0, 0.0776692, 0.555045, 1.49892, -1.09425, 0, -0.183137, 0.0278837, -0.290078, -0.317777, -0.152792],
                  [419.713, 519.888, 0, 1.66667e10, -1., 486.978, 335.855, 527.173, -9.90337, 0, 519.552, 3.30254, -0.813864, -0.152792, 3.33333e10]])

    eigenvalues, eigenvectors, errors, taxa_convergencia = power_method_deflation(A)


import numpy as np

# Matriz de exemplo do Mathematica
mat_inicial = np.array([[4, 1, 3, 3],
                [2, 0, 1, 1],
                [8, 5, 9, 7],
                [6, 8, 9, 7]])
print(mat_inicial)

def encontrar_zero(mat):
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            if mat[i][j] == 0:
                return i, j
    return -1, -1

def pivoteamento(mat):
    linha_zero, coluna_zero = encontrar_zero(mat)
    if linha_zero == -1 and coluna_zero == -1:
        print("Não foi encontrado nenhum elemento zero na matriz.")
        return

    # Encontrar o índice do maior elemento na matriz
    indice_maior = np.unravel_index(np.argmax(mat), mat.shape)
    linha_maior, coluna_maior = indice_maior[0], indice_maior[1]

    # Trocar a linha ou coluna do maior elemento pelo elemento zero
    mat[linha_maior], mat[linha_zero] = mat[linha_zero], mat[linha_maior].copy()
    mat[:, coluna_maior], mat[:, coluna_zero] = mat[:, coluna_zero], mat[:, coluna_maior].copy()

    print("Matriz após o pivoteamento:")
    print(mat)

pivoteamento(mat_inicial)


# # Encontrar o índice do elemento zero na matriz
# indice_zero = np.where(mat_inicial == 0)

# # Verificar se há algum elemento zero na matriz
# if len(indice_zero[0]) > 0:
#     linha_zero, coluna_zero = indice_zero[0][0], indice_zero[1][0]

#     # Trocar a linha ou coluna pelo elemento zero
#     mat_inicial[linha_zero], mat_inicial[:, coluna_zero] = mat_inicial[:, coluna_zero], mat_inicial[linha_zero].copy()

#     print("Matriz após a troca:")
#     print(mat_inicial)
# else:
#     print("Não foi encontrado nenhum elemento zero na matriz.")

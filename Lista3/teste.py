import numpy as np

#Matriz de exemplo do Mathematica
# mat_inicial = np.array([[0.000790584, 0.0650192, 0.989555, 0.968768, 0.200866],
#                         [0.819521, 0.0897634, 0.970701, 0.22991, 0.612503],
#                         [0.096816, 0.548855, 0.132548, 0.232332, 0.776135],
#                         [0.550949, 0.0586896, 0.960602, 0.0982487, 0.0343521],
#                         [0.806562, 0.439186, 0.142284, 0.027687, 0.0794711]])

mat_inicial = np.array([[1.0, 3.0, 5.0],
                        [7.0, 9.0, 2.0],
                        [4.0, 6.0, 8.0]])

# Função para imprimir na forma matricial
def print_matrix(mat):
    max_len = max(len(str(element)) for row in mat for element in row)
    for row in mat:
        print(" ".join(str(element).ljust(max_len) for element in row))

def encontrar_zero(mat: np.array):
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i, j] == 0:
                return i, j
    return -1, -1

def pivoteamento(mat):
    linha_zero, coluna_zero = encontrar_zero(mat)

    # Encontrar o índice do maior elemento na matriz, np.argmax() retorna o índice do elemento máximo em um array multidimensional
    indice_maior = np.unravel_index(np.argmax(mat[linha_zero:, coluna_zero:]), mat[linha_zero:, coluna_zero:].shape)
    linha_maior, coluna_maior = indice_maior[0] + linha_zero, indice_maior[1] + coluna_zero

    # Trocar a linha ou coluna do maior elemento pelo elemento zero
    mat[linha_maior], mat[linha_zero] = mat[linha_zero].copy(), mat[linha_maior].copy()
    mat[:, coluna_maior], mat[:, coluna_zero] = mat[:, coluna_zero].copy(), mat[:, coluna_maior].copy()

    # Criar matriz permutação
    P = np.identity(len(mat))
    P[linha_maior], P[linha_zero] = P[linha_zero].copy(), P[linha_maior].copy()
    P[:, coluna_maior], P[:, coluna_zero] = P[:, coluna_zero].copy(), P[:, coluna_maior].copy()

    # Multiplicar a matriz permutação pela matriz original
    mat = np.dot(P, mat)
    print(P)
    return mat


# Função para realizar o algoritmo
def LU_decomposition(mat_inicial: np.array):
    # mat2 = np.copy(mat)
    mat = np.copy(mat_inicial)
    n =  len(mat)
    L = np.identity(n)
    U = np.identity(n)

    for i in range(n-1):
        if mat[i, i] == 0:
            raise ValueError("Null Pivot")
            
        mat[i+1:,i] /= mat[i,i]
        mat[i+1:,i+1:] -= np.outer(mat[i+1:,i],mat[i,i+1:])     
        # print(f"Iteração {i}")   
        # print(f"{mat = }")    

    #Python começa em zero e Mathematica começa em 1
    for i in range(1, n):
        for j in range(i):
            L[i, j] = mat[i, j]
    # print("\nMatriz L:", L)

    for j in range(n):
        for i in range(j+1):
            U[i,j] = mat[i, j]
    # print("\nMatriz U:", U)

    return L, U, mat

L, U, mat = LU_decomposition(mat_inicial)

# print(mat)
# print("\nMatriz L:")
# print(L)
# print("\nL*U:")
# print(U)
# print("\nL*U:")
# print(a)

# Execução do algoritmo
# L, U, mat2, a = LU_decomposition(mat_inicial)

# Define a opção de impressão para suprimir a notação científica e imprimir em formato decimal
np.set_printoptions(suppress=True, formatter={'float': '{:0.5f}'.format})

#Se coloca LU dentro da função não funciona
LU = L @ U
#Saída da matriz transformada
print("\nMatriz L:")
print_matrix(np.round(L, decimals=5))
print("\nMatriz U:")
print_matrix(np.round(U, decimals=5))
print("\nMatriz transformada:")
#print_matrix(mat2)
print_matrix(np.round(mat, decimals=5))
print("\nL*U:")
print_matrix(np.round(LU, decimals=5))
print("\nMatriz inicial:")
print_matrix(np.round(mat_inicial, decimals=5))




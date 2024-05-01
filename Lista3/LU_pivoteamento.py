import numpy as np
from tabulate import tabulate
class Pivoteamento:
    def __init__(self, matriz):
        self.matriz = matriz
        if (len(matriz) != len(matriz[0])):
            raise ValueError("Matriz não é quadrada")
        
        if (np.linalg.det(matriz) == 0):
            raise ValueError("Determinante é zero")
        else:
            print("Determinante = ", np.linalg.det(matriz))
    
        self.perml = np.identity(len(matriz))
        self.permc = np.identity(len(matriz))

    def pivotear(self):
        n = len(self.matriz)
        m = len(self.matriz[0])  # número de colunas

        for i in range(n-1):
            permute_line = np.array([i for i in range(n)])
            permute_colum = np.array([i for i in range(n)])

            # Encontrar o maior elemento na coluna atual
            max_value = abs(self.matriz[i][i])
            max_row = i
            max_col = i 
            
            for j in range(i + 1, n):
                for k in range(i, m):
                    if abs(self.matriz[j][k]) > max_value:
                        max_value = abs(self.matriz[j][k])
                        max_row = j
                        max_col = k
            permute_line[i] = max_row
            permute_line[max_row] = i
            permute_colum[i] = max_col
            permute_colum[max_col] = i
            print("Permute line")
            print(permute_line)
            print("Permute colum")
            print(permute_colum)
            perm_line = np.zeros((n,n))
            perm_colum = np.zeros((n,n))
            for j in range(n):
                perm_line[j, permute_line[j]] = 1.
                perm_colum[permute_colum[j], j] = 1.
            self.perml = perm_line @ self.perml
            self.permc = self.permc @ perm_colum
            print("Permute line matrix it")
            print(perm_line)
            print("Permute colum matrix it")
            print(perm_colum)
            print("Permute line matrix ac")
            print(self.perml)
            print("Permute colum matrix ac")
            print(self.permc)

            self.matriz = perm_line @ self.matriz @ perm_colum

            print(f"matriz após pivotear {i}")
            print(self.matriz)

            
            # Ranquear                  
            self.matriz[i+1:,i] /= self.matriz[i,i]
            self.matriz[i+1:,i+1:] -= np.outer(self.matriz[i+1:,i],self.matriz[i,i+1:]) 

            print(f"matriz após ranquear {i}")
            #print(self.matriz)
            wolfram_format = "{{" + "},\n{".join([", ".join([f"{element:.4f}" for element in row]) for row in self.matriz]) + "}}"
            print(wolfram_format)

    

        return self.matriz

    def LU_decomposition(self):
        n =  len(self.matriz)
        L = np.identity(n)
        U = np.identity(n)

        #Python começa em zero e Mathematica começa em 1
        for i in range(1, n):
            for j in range(i): 
                L[i, j] = self.matriz[i, j]
        print("\nMatriz L:")
        #print(L)
        wolfram_format = "{{" + "},\n{".join([", ".join([f"{element:.4f}" for element in row]) for row in L]) + "}}"
        print(wolfram_format)


        for j in range(n):
            for i in range(j+1):
                U[i,j] = self.matriz[i, j]
        print("\nMatriz U:")
        print(U)
        wolfram_format = "{{" + "},\n{".join([", ".join([f"{element:.4f}" for element in row]) for row in U]) + "}}"
        print(wolfram_format)


        print("\nL*U")
        print(self.perml.T @ L @ U @ self.permc.T)
        #np.savetxt('/home/ingrid/Numerical_Methods/Lista3/LU_pivoteamento.txt', L)

        Verif = matriz - self.perml.T @ L @ U @ self.permc.T
        print("\nVerificação")
        #print(Verif)
        wolfram_format = "{{" + "},\n{".join([", ".join([f"{element:.2f}" for element in row]) for row in Verif]) + "}}"
        print(wolfram_format)
        return L, U
    

# matriz = np.array([[1., 1., 1.],
#                    [1., 0., 1.],
#                    [1., 1., 1.]])

# matriz = np.array([[4., 1., 3., 3.],
#                    [2., 0., 1., 1.],
#                    [8., 5., 9., 7.],
#                    [6., 8., 9., 7.]])

# matriz = np.array([[4., 1., 3.],
#                    [2., 0., 1.],
#                    [8., 5., 9.]])



# matriz = np.array([[0.000790584, 0.0650192, 0.989555, 0.968768, 0.200866],
#                     [0.819521, 0.0897634, 0.970701, 0.22991, 0.612503],
#                     [0.096816, 0.548855, 0.132548, 0.232332, 0.776135],
#                     [0.550949, 0.0586896, 0.960602, 0.0982487, 0.0343521],
#                     [0.806562, 0.439186, 0.142284, 0.027687, 0.0794711]])

matriz = np.array([[815.732, 418.604, 0, 420.445, -1., 281.869, 232.076, 397.658, 65.1193, 0, 421.275, 2.34737, 0.0571453, -0.139624, 419.713],
                    [418.604, 520.872, 0, 519.833, -1., 488.09, 334.978, 527.169, -9.12066, 0, 520.524, 2.60818, -1.5467, -0.182023, 519.888],
                    [0, 0, 0, 0, 0, 0, -1., -1., 1., 1., 0, 0, 0, 0, 0],
                    [420.445, 519.833, 0, 3.33333e10, -1., 486.144, 336.579, 528.043, -11.6014, 0,  519.504, 2.57783, -0.263573, -0.877493, 1.66667e10],
                    [-1., -1., 0, -1., 0, -1., -1., -1., 0, 0, -1., 0, 0, 0, -1.],
                    [281.869, 488.09, 0, 486.144, -1., 948.696, 348.62, 565.821, -86.6241, 0, 484.754, 2.56467, -1.13458, 0.0776692, 486.978],
                    [232.076, 334.978, -1., 336.579, -1., 348.62, -61966.4, -30722.5, 106707., 0, 337.077, 2.02037, 0.489366, 0.555045, 335.855],
                    [397.658, 527.169, -1., 528.043, -1., 565.821, -30722.5, -8229.76, 53353.6, 0, 527.669, 2.96425, 0.437361, 1.49892, 527.173],
                    [65.1193, -9.12066, 1., -11.6014, 0, -86.6241, 106707., 53353.6, -160083., 0, -12.3841, -1.09425, -2.00416, -1.09425, -9.90337],
                    [0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [421.275, 520.524, 0, 519.504, -1., 484.754, 337.077, 527.669, -12.3841, 0, 520.207, 2.60707, 0.469261, -0.183137, 519.552],
                    [2.34737, 2.60818, 0, 2.57783, 0, 2.56467, 2.02037, 2.96425, -1.09425, 0, 2.60707,-0.317777, -0.290078, 0.0278837, 3.30254],
                    [0.0571453, -1.5467, 0, -0.263573, 0, -1.13458, 0.489366, 0.437361, -2.00416, 0, 0.469261, -0.290078, -0.580992, -0.290078, -0.813864],
                    [-0.139624, -0.182023, 0, -0.877493, 0, 0.0776692, 0.555045, 1.49892, -1.09425, 0, -0.183137, 0.0278837, -0.290078, -0.317777, -0.152792],
                    [419.713, 519.888, 0, 1.66667e10, -1., 486.978, 335.855, 527.173, -9.90337, 0, 519.552, 3.30254, -0.813864, -0.152792, 3.33333e10]])

pivoteamento = Pivoteamento(matriz)
matriz_pivotada = pivoteamento.pivotear()
L,U = pivoteamento.LU_decomposition()




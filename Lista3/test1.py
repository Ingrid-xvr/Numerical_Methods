import numpy as np

class Pivoteamento:
    def __init__(self, matriz):
        self.matriz = matriz

    def pivotear(self):
        n = len(self.matriz)
        m = len(self.matriz[0])  # número de colunas
        permute_line = np.array(n)
        permute_colum = np.array(n)
        
        for i in range(n):
            permute_line[i] = i
            permute_colum[i] = i

        for i in range(n):
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

            # Trocar a linha atual pela linha com o maior elemento
            self.matriz[i], self.matriz[max_row] = self.matriz[max_row].copy(), self.matriz[i].copy()

            # Trocar a coluna atual pela coluna com o maior elemento
            for j in range(n):
                self.matriz[j][i], self.matriz[j][max_col] = self.matriz[j][max_col], self.matriz[j][i]

            print(f"matriz após pivotear {i}")
            print(self.matriz)
            
            # Ranquear
            # for k in range(i+1, n):
                  
            self.matriz[i+1:,i] /= self.matriz[i,i]
            self.matriz[i+1:,i+1:] -= np.outer(self.matriz[i+1:,i],self.matriz[i,i+1:]) 

            print(f"matriz após ranquear {i}")
            print(self.matriz)
    
    

        return self.matriz

    def LU_decomposition(self):
    # mat2 = np.copy(mat)
        mat = np.copy(self.matriz)
        n =  len(mat)
        L = np.identity(n)
        U = np.identity(n)

        for i in range(n-1):
            if mat[i, i] == 0:
                raise ValueError("Null Pivot")

            # pivot = Pivoteamento(mat)
            print("\nMatriz final:")
            print(mat)
            mat[i+1:,i] /= mat[i,i]
            mat[i+1:,i+1:] -= np.outer(mat[i+1:,i],mat[i,i+1:])     
            # print(f"Iteração {i}")   
            # print(f"{mat = }")    

        #Python começa em zero e Mathematica começa em 1
        for i in range(1, n):
            for j in range(i): 
                L[i, j] = mat[i, j]
        print("\nMatriz L:")
        print(L)

        for j in range(n):
            for i in range(j+1):
                U[i,j] = mat[i, j]
        print("\nMatriz U:")
        print(U)

        print("\nL*U")
        print(L @ U)

        return L, U, mat
        # def ranquear(self):
    #     # Ranquear
    #     n = len(self.matriz)
    #     for i in range(n):
    #         #self.matriz[k, i:] -= (self.matriz[k, i] / self.matriz[i, i]) * self.matriz[i, i:]
    #         matriz_pivot = self.pivotear(matriz)
    #         self.matriz[i+1:,i] /= self.matriz[i,i]
    #         self.matriz[i+1:,i+1:] -= np.outer(self.matriz[i+1:,i],self.matriz[i,i+1:]) 

        # print(f"matriz após ranquear {i}")
        # print(self.matriz)
    

# matriz = np.array([[1., 1., 1.],
#                    [1., 0., 1.],
#                    [1., 1., 1.]])

# matriz = np.array([[4., 1., 3., 3.],
#                    [2., 0., 1., 1.],
#                    [8., 5., 9., 7.],
#                    [6., 8., 9., 7.]])

matriz = np.array([[4., 1., 3.],
                   [2., 0., 1.],
                   [8., 5., 9.]])



# matriz = np.array([[0.000790584, 0.0650192, 0.989555, 0.968768, 0.200866],
#                     [0.819521, 0.0897634, 0.970701, 0.22991, 0.612503],
#                     [0.096816, 0.548855, 0.132548, 0.232332, 0.776135],
#                     [0.550949, 0.0586896, 0.960602, 0.0982487, 0.0343521],
#                     [0.806562, 0.439186, 0.142284, 0.027687, 0.0794711]])

pivoteamento = Pivoteamento(matriz)
matriz_pivotada = pivoteamento.pivotear()
#matriz_final = pivoteamento.ranquear(matriz)
matriz_decomposta = pivoteamento.LU_decomposition()




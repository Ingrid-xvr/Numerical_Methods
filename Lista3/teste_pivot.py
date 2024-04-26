import numpy as np

class Pivoteamento:
    def __init__(self, matriz):
        self.matriz = matriz

    def pivotear(self):
        n = len(self.matriz)
        m = len(self.matriz[0])  # número de colunas

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

            # Trocar a linha atual pela linha com o maior elemento
            self.matriz[i], self.matriz[max_row] = self.matriz[max_row].copy(), self.matriz[i].copy()

            # Trocar a coluna atual pela coluna com o maior elemento
            for j in range(n):
                self.matriz[j][i], self.matriz[j][max_col] = self.matriz[j][max_col], self.matriz[j][i]

            print(f"matriz após pivotear {i}")
            print(matriz)
            
            #Rank1
            self.matriz[i+1:,i] /= self.matriz[i,i]
            self.matriz[i+1:,i+1:] -= np.outer(self.matriz[i+1:,i],self.matriz[i,i+1:]) 

            print(f"matriz após rankear {i}")
            print(matriz)
        return self.matriz

matriz = np.array([[4., 1., 3., 3.],
                   [2., 0., 1., 1.],
                   [8., 5., 9., 7.],
                   [6., 8., 9., 7.]])

pivoteamento = Pivoteamento(matriz)
matriz_pivotada = pivoteamento.pivotear()
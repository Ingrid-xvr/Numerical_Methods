import numpy as np

class Pivoteamento:
    def __init__(self, matriz):
        self.matriz = matriz
        if len(matriz) != len(matriz[0]):
            raise ValueError("Matriz não é quadrada")
        self.perml = np.identity(len(matriz))
        self.permc = np.identity(len(matriz))

    def pivotear(self):
        n = len(self.matriz)
        m = len(self.matriz[0])  # número de colunas

        for i in range(n - 1):
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

            # Realizar permutação de linhas
            self.matriz[[i, max_row]] = self.matriz[[max_row, i]]
            self.perml[[i, max_row]] = self.perml[[max_row, i]]

            # Realizar permutação de colunas
            self.matriz[:, [i, max_col]] = self.matriz[:, [max_col, i]]
            self.permc[:, [i, max_col]] = self.permc[:, [max_col, i]]

            # Atualizar a matriz após pivoteamento
            self.matriz = self.permc @ self.matriz @ self.perml

        return self.matriz

    def LDLt_decomposition(self):
        n = len(self.matriz)
        L = np.identity(n)
        D = np.zeros((n, n))

        for i in range(n):
            D[i, i] = self.matriz[i, i]
            for j in range(i + 1, n):
                L[j, i] = self.matriz[j, i] / D[i, i]
                self.matriz[j, i+1:] -= L[j, i] * self.matriz[i, i+1:]

        print("\nL * D * Lt:")
        print(L @ D @ L.T)

        Verif = matriz - L @ D @ L.T
        print("\nVerificação:")
        print(Verif)

        return L, D, L.T

matriz = np.array([[4., 1., 8.],
                  [1., 0., 5.],
                  [8., 5., 9.]])

# matriz = np.array([[815.732, 418.604, 0, 420.445, -1., 281.869, 232.076, 397.658, 65.1193, 0, 421.275, 2.34737, 0.0571453, -0.139624, 419.713],
#                     [418.604, 520.872, 0, 519.833, -1., 488.09, 334.978, 527.169, -9.12066, 0, 520.524, 2.60818, -1.5467, -0.182023, 519.888],
#                     [0, 0, 0, 0, 0, 0, -1., -1., 1., 1., 0, 0, 0, 0, 0],
#                     [420.445, 519.833, 0, 3.33333e10, -1., 486.144, 336.579, 528.043, -11.6014, 0,  519.504, 2.57783, -0.263573, -0.877493, 1.66667e10],
#                     [-1., -1., 0, -1., 0, -1., -1., -1., 0, 0, -1., 0, 0, 0, -1.],
#                     [281.869, 488.09, 0, 486.144, -1., 948.696, 348.62, 565.821, -86.6241, 0, 484.754, 2.56467, -1.13458, 0.0776692, 486.978],
#                     [232.076, 334.978, -1., 336.579, -1., 348.62, -61966.4, -30722.5, 106707., 0, 337.077, 2.02037, 0.489366, 0.555045, 335.855],
#                     [397.658, 527.169, -1., 528.043, -1., 565.821, -30722.5, -8229.76, 53353.6, 0, 527.669, 2.96425, 0.437361, 1.49892, 527.173],
#                     [65.1193, -9.12066, 1., -11.6014, 0, -86.6241, 106707., 53353.6, -160083., 0, -12.3841, -1.09425, -2.00416, -1.09425, -9.90337],
#                     [0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                     [421.275, 520.524, 0, 519.504, -1., 484.754, 337.077, 527.669, -12.3841, 0, 520.207, 2.60707, 0.469261, -0.183137, 519.552],
#                     [2.34737, 2.60818, 0, 2.57783, 0, 2.56467, 2.02037, 2.96425, -1.09425, 0, 2.60707,-0.317777, -0.290078, 0.0278837, 3.30254],
#                     [0.0571453, -1.5467, 0, -0.263573, 0, -1.13458, 0.489366, 0.437361, -2.00416, 0, 0.469261, -0.290078, -0.580992, -0.290078, -0.813864],
#                     [-0.139624, -0.182023, 0, -0.877493, 0, 0.0776692, 0.555045, 1.49892, -1.09425, 0, -0.183137, 0.0278837, -0.290078, -0.317777, -0.152792],
#                     [419.713, 519.888, 0, 1.66667e10, -1., 486.978, 335.855, 527.173, -9.90337, 0, 519.552, 3.30254, -0.813864, -0.152792, 3.33333e10]])

pivoteamento = Pivoteamento(matriz)
matriz_pivotada = pivoteamento.pivotear()
LDLt = pivoteamento.LDLt_decomposition()

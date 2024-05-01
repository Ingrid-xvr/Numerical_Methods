import numpy as np

class SparseMatrix:
    def __init__(self, matrix):
        self.rows = len(matrix)
        self.cols = len(matrix[0])
        self.data = [] #valores nao nulos
        self.col = [] #indices das colunas correspondentes aos valores nao nulos
        self.ptr = [0] #indices de inicio de cada linha na lista de dados e na lista de indices de coluna

        print(f"n linhas: {self.rows}")
        print(f" n colunas: {self.cols}")
        
        #le a matriz, armazena os valores nao nulos e seus respectivos índices de coluna, e faz uma lista de índices que indicam onde cada linha começa na lista de valores não nulos
        for linha in matrix:
            for coluna, val in enumerate(linha):
                if val != 0:
                    self.data.append(val)
                    self.col.append(coluna)
            self.ptr.append(len(self.data))



    def print_matrix(self):
        print("data:", self.data)
        print("N° elementos data:", len(self.data))
        print("col:", self.col)
        print("N° elementos col:", len(self.col))
        print("ptr:", self.ptr)
        print("N° elementos ptr:", len(self.ptr))

    #calcula o número de zeros, subtrai o tamanho da matriz pelo número de elementos nao nulos. O indice de esparsidade = (número de zeros/pelo tamanho total da matriz)*100 (porcentagem)
    def sparsity_index(self):
        num_zeros = self.rows * self.cols - len(self.data)
        sparsity = (num_zeros / (self.rows * self.cols)) * 100
        return sparsity

    def multiply_sparse_vector(self, vector):
        result = [0] * self.rows
        for i in range(self.rows):
            for j in range(self.ptr[i], self.ptr[i+1]):
                result[i] += self.data[j] * vector[self.col[j]]
        return result
    
    def consumo_memoria_cheia(self, matrix):

       tamanho = self.rows * self.cols * 8
       return tamanho
    
    def consumo_memoria_esparsa(self):
        tamanho_data = len(self.data) * 8
        tamanho_col = len(self.col) * 4
        tamanho_ptr = len(self.ptr) * 4
        tamanho_total = tamanho_data + tamanho_col + tamanho_ptr
        return tamanho_total



#Exemplo da aula - ok!
# matrix = [
#     [1.0, 0, 2, 0, 0],
#     [0, 3, 0, 0, 2],
#     [0, 0, 0, 0, 0],
#     [1, 0, 0, 4, 0],
#     [0, 0, 0, 0, 5]
# ]

#Exemplo da aula - ok!
# matrix = [
#     [0, 0, 0, 0, 0, 2],
#     [0, 0, 1, 0, 0, 0],
#     [3, 4, 0, 0, 0, 0],
#     [0, 0, 0, 0, 5, 0],
#     [0, 0, 1, 0, 0, 0],
#     [2, 0, 0, 0, 0, 0]
# ]

matrix = [[815.732, 418.604, 0, 420.445, -1., 281.869, 232.076, 397.658, 65.1193, 0, 421.275, 2.34737, 0.0571453, -0.139624, 419.713],
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
[419.713, 519.888, 0, 1.66667e10, -1., 486.978, 335.855, 527.173, -9.90337, 0, 519.552, 3.30254, -0.813864, -0.152792, 3.33333e10]]

#vetor qualquer
#vector = [1, 2, 3, 4, 5]

#vetor qualquer
vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

sparse_matrix = SparseMatrix(matrix)
sparse_matrix.print_matrix()

print(f"Índice de Esparsidade: {sparse_matrix.sparsity_index()} %")

result_sparse = sparse_matrix.multiply_sparse_vector(vector)
print("Resultado da multiplicação da matriz esparsa pelo vetor:", result_sparse)


# Convertendo a matriz esparsa para matriz cheia
dense_matrix = np.array(matrix)

# Multiplicação da matriz cheia pelo vetor
result_dense = np.dot(dense_matrix, vector)

print("Resultado da multiplicação da matriz cheia pelo vetor:", result_dense)

memoria_cheia = sparse_matrix.consumo_memoria_cheia(matrix)
print(f"Consumo de memória matriz cheia: {memoria_cheia} bytes")

memoria_esparsa = sparse_matrix.consumo_memoria_esparsa()
print(f"Consumo de memória matriz esparsa: {memoria_esparsa} bytes")




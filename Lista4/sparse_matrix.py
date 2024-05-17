import numpy as np
import re

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
matrix = [
    [1.0, 0, 2, 0, 0],
    [0, 3, 0, 0, 2],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 4, 0],
    [0, 0, 0, 0, 5]
]

#vetor qualquer
vector = [1, 2, 3, 4, 5]

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




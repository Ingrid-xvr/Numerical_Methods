import re
import numpy as np
import tokenize

class SparseMatrix:
    def __init__(self, matrix=None, file_path=None):

        if file_path:
            with tokenize.open(file_path) as file:
                tokens = tokenize.generate_tokens(file.readline)
                nnz = 0
                col = -1
                line = -2

                self.data = []
                self.col = []
                self.ptr = [0]
                for token in tokens:
                    if token.string == '{':
                        line+=1
                        nnz = 0
                    elif token.type == tokenize.NUMBER:
                        col += 1
                        el = float(token.string)
                        if el >= 1e-10:
                            self.data.append(el)
                            self.col.append(col)
                            nnz += 1
                    elif token.string == '}':
                        self.ptr.append(self.ptr[-1] + nnz)
                        col = -1
            self.ptr.pop()
            self.rows = len(self.ptr)-1
            self.cols = self.rows

        elif matrix:
            # Se uma matriz for fornecida diretamente
            self.matrix = matrix

            self.rows, self.cols = self.matrix.shape
            self.data = [] 
            self.col = [] 
            self.ptr = [0]

            for i in range(self.rows):
                for j in range(self.cols):
                    if self.matrix[i][j] != 0:
                        self.data.append(self.matrix[i][j])
                        self.col.append(j)
                self.ptr.append(len(self.data))
        else:
            raise ValueError("Fornecer ou uma matriz ou um caminho de arquivo.")

        

    def print_matrix(self):
        #print("data:", self.data)
        print("N° elementos data:", len(self.data))
        #print("col:", self.col)
        print("N° elementos col:", len(self.col))
        #print("ptr:", self.ptr)
        print("N° elementos ptr:", len(self.ptr))

    def sparsity_index(self):
        num_zeros = self.rows * self.cols - len(self.data)
        sparsity = (num_zeros / (self.rows * self.cols)) * 100
        return sparsity

    def multiply_sparse_vector(self, vector):
        if len(vector) != self.cols:
            raise ValueError("O vetor de multiplicação deve ter o mesmo número de elementos que o número de colunas da matriz.")

        result = [0] * self.rows
        for i in range(self.rows):
            for j in range(self.ptr[i], self.ptr[i+1]):
                result[i] += self.data[j] * vector[self.col[j]]
        return result
    
    def consumo_memoria_cheia(self):
        tamanho = self.rows * self.cols * 8
        return tamanho
    
    def consumo_memoria_esparsa(self):
        tamanho_data = len(self.data) * 8
        tamanho_col = len(self.col) * 4
        tamanho_ptr = len(self.ptr) * 4
        tamanho_total = tamanho_data + tamanho_col + tamanho_ptr
        return tamanho_total

# Lendo a matriz do arquivo .dat 
sparse_matrix_file = SparseMatrix(file_path='matrix.dat')
sparse_matrix_file.print_matrix()

print(f"Índice de Esparsidade: {sparse_matrix_file.sparsity_index()} %")

# Lendo o vetor do arquivo .dat
with open('rhs.dat', 'r') as file:
    linhas = file.readlines()
    # O vetor é representado entre chaves e cada elemento está em uma linha separada
    vector = [float(re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', linha)[0]) for linha in linhas if re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', linha)]

print("Tamanho do vetor:", len(vector))
print("Número de colunas da matriz:", sparse_matrix_file.cols)

# Multiplicação da matriz esparsa pelo vetor
result_sparse = sparse_matrix_file.multiply_sparse_vector(vector)
#print("Resultado da multiplicação da matriz esparsa pelo vetor:", result_sparse)

# Comparando o consumo de memória
memoria_cheia = sparse_matrix_file.consumo_memoria_cheia()
print(f"Consumo de memória matriz cheia: {memoria_cheia} bytes")

memoria_esparsa = sparse_matrix_file.consumo_memoria_esparsa()
print(f"Consumo de memória matriz esparsa: {memoria_esparsa} bytes")

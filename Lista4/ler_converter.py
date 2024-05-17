import re
import numpy as np

class SparseMatrix:
    def __init__(self, matrix=None, file_path=None):
        if file_path:
            # Se o caminho do arquivo for fornecido, lê a matriz do arquivo
            with open(file_path, 'r') as file:
                linhas = file.readlines()

            matriz_linhas = []

            for linha in linhas[2:-2]:
                elementos = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', linha)  
                matriz_linhas.append(elementos)

            max_len = max(len(linha) for linha in matriz_linhas)

            for linha in matriz_linhas:
                linha.extend(['0'] * (max_len - len(linha)))

            self.matrix = np.array(matriz_linhas, dtype=float)
        elif matrix:
            # Se uma matriz for fornecida diretamente
            self.matrix = matrix
        else:
            raise ValueError("Fornecer ou uma matriz ou um caminho de arquivo.")

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

    def print_matrix(self):
        print("data:", self.data)
        print("N° elementos data:", len(self.data))
        print("col:", self.col)
        print("N° elementos col:", len(self.col))
        print("ptr:", self.ptr)
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

# Lendo a matriz do arquivo .dat e usando-a
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

# Convertendo a matriz esparsa para matriz cheia
dense_matrix = np.array(sparse_matrix_file.matrix)

# Multiplicação da matriz esparsa pelo vetor
result_sparse = sparse_matrix_file.multiply_sparse_vector(vector)
print("Resultado da multiplicação da matriz esparsa pelo vetor:", result_sparse)

# Multiplicação da matriz cheia pelo vetor
result_dense = np.dot(dense_matrix, vector)
print("Resultado da multiplicação da matriz cheia pelo vetor:", result_dense)

# Comparando o consumo de memória
memoria_cheia = sparse_matrix_file.consumo_memoria_cheia()
print(f"Consumo de memória matriz cheia: {memoria_cheia} bytes")

memoria_esparsa = sparse_matrix_file.consumo_memoria_esparsa()
print(f"Consumo de memória matriz esparsa: {memoria_esparsa} bytes")

# import re
# import numpy as np

# # Abre o arquivo .dat para leitura
# with open('matrix.dat', 'r') as file:
#     # Lê todas as linhas do arquivo
#     linhas = file.readlines()

# # Inicializa uma lista para armazenar as linhas da matriz
# matriz_linhas = []

# # Itera sobre as linhas do arquivo, ignorando as primeiras 2 e as últimas 2
# for linha in linhas[2:-2]:
#     # Encontra todos os números na linha 
#     elementos = re.findall(r'[-+]?\d*\.\d+|\d+', linha)
    
#     # Converte os elementos para valores numéricos e adiciona à lista de linhas
#     matriz_linhas.append(elementos)

# # Verifica o comprimento das linhas para encontrar a maior
# max_len = max(len(linha) for linha in matriz_linhas)

# # Preenche linhas com menos elementos com zeros à direita
# for linha in matriz_linhas:
#     linha.extend(['0'] * (max_len - len(linha)))

# # Converte a lista de linhas para uma matriz NumPy
# matriz = np.array(matriz_linhas, dtype=float)

# # Imprime a matriz
# print(matriz)
# print(matriz.T)

import re
import numpy as np

# Abre o arquivo .dat para leitura
with open('matrix.dat', 'r') as file:
    # Lê todas as linhas do arquivo
    linhas = file.readlines()

# Inicializa uma lista para armazenar as linhas da matriz
matriz_linhas = []

# Itera sobre as linhas do arquivo, ignorando as primeiras 2 e as últimas 2
for linha in linhas[2:-2]:
    # Encontra todos os números na linha 
    elementos = re.findall(r'[-+]?\d*\.\d+|\d+', linha)
    
    # Converte os elementos para valores numéricos e adiciona à lista de linhas
    matriz_linhas.append(elementos)

# Verifica o comprimento das linhas para encontrar a maior
max_len = max(len(linha) for linha in matriz_linhas)

# Preenche linhas com menos elementos com zeros à direita
for linha in matriz_linhas:
    linha.extend(['0'] * (max_len - len(linha)))

# Converte a lista de linhas para uma matriz NumPy
matriz = np.array(matriz_linhas, dtype=float)

# Imprime o número de linhas e colunas da matriz
print("Número de linhas:", matriz.shape[0])
print("Número de colunas:", matriz.shape[1])

# Imprime a matriz
print("Matriz:")
print(matriz)

# Imprime a transposta da matriz
print("Transposta:")
print(matriz.T)



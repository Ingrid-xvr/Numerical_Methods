import re
import numpy as np
import tokenize
import matplotlib
import matplotlib.pyplot as plt

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
                factor = 1
                for token in tokens:
                    if token.string == '{':
                        line+=1
                        nnz = 0
                    elif token.string == '-':
                        factor = -1
                    elif token.type == tokenize.NUMBER:
                        col += 1
                        el = float(token.string)
                        if abs(el) >= 1e-10:
                            self.data.append(factor*el)
                            self.col.append(col)
                            nnz += 1
                            factor = 1
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

        result = np.zeros(self.rows)
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

def iterative_conjugate_gradient_pc(A, B, solini, niter, method):
    neq = len(solini)
    sol = solini.copy()
    zero = np.zeros_like(solini)
    res = B - A.multiply_sparse_vector(sol)
    z, _ = method(A, 1, zero, res)
    zk = z.copy()
    p = z.copy()
    resk = res.copy()
    resnorm = [np.linalg.norm(res)]

    for i in range(niter):
        Ap = A.multiply_sparse_vector(p)
        alpha = np.dot(res.T, z) / np.dot(p.T, Ap)
        sol = sol + alpha * p
        res = resk - alpha * Ap
        z, _ = method(A, 1, zero, res)
        beta = np.dot(res.T, z) / np.dot(resk.T, zk)
        zk = z.copy()
        p = z + beta * p
        resk = res.copy()
        resnorm.append(np.linalg.norm(res))

    return sol, resnorm

def jacobi(A, omega, u, res):
    # Calcular a diagonal da matriz esparsa A
    diagonal = []
    for i in range(A.rows):
        for j in range(A.ptr[i], A.ptr[i+1]):
            if A.col[j] == i:
                diagonal.append(A.data[j])
                break
    Diag = np.array(diagonal) 
    
    #print("res:", res)
    #print("Diag:", Diag)

    resloc = res.copy()
    delu = np.zeros_like(u)
    delu = omega * res / Diag
    result = A.multiply_sparse_vector(delu)
    resloc -= result
    #print("resloc:", resloc)
    return u + delu, resloc

def iterative_jacobi(A, B, solini, niter):
    sol = solini.copy()
    res = np.array(B) - np.array(A.multiply_sparse_vector(sol))
    resnorm = [np.linalg.norm(res)]
    for i in range(0, niter):
        sol, res = jacobi(A, 1, sol, res)  # omega = 1
        resnorm.append(np.linalg.norm(res))
    return sol, resnorm

def find_element(self, row, col):
        for i in range(self.ptr[row], self.ptr[row+1]):
            if self.col[i] == col:
                return self.data[i]
        return 0

def gauss_seidel_f(A, omega, u, res):
    
    n = len(res)
    delu = np.zeros_like(u)
    resc = res.copy()
    delu[0] = omega * res[0] / find_element(A, 0, 0)
    for i in range(1, n):
        resc[i] -= np.dot(A.data[A.ptr[i]:A.ptr[i+1]], delu[A.col[A.ptr[i]:A.ptr[i+1]]])
        delu[i] += omega * resc[i] / find_element(A, i, i)
    resc = res - A.multiply_sparse_vector(delu)
    return u + delu, resc

def gauss_seidel_b(A, omega, u, res):
    n = len(res)
    delu = np.zeros_like(u)
    resc = res.copy()
    delu[n-1] = omega * res[n-1] / find_element(A, n-1, n-1)
    for i in range(n-2, -1, -1):
        start = A.ptr[i]
        end = A.ptr[i+1]
        col = A.col[start:end]
        data = A.data[start:end]
        resc[i] -= np.dot(data, delu[col])
        delu[i] += omega * resc[i] / find_element(A, i, i)
    resc = res - A.multiply_sparse_vector(delu)
    return u + delu, resc

def ssor(A, omega, u, res):
    delu, resc = gauss_seidel_f(A, omega, u, res)
    delu2, resc2 = gauss_seidel_b(A, omega, delu, resc)
    return delu2, resc2

def iterative_ssor(A, B, solini, omega, niter):
    sol = solini.copy()
    res = B - A.multiply_sparse_vector(sol)
    resnorm = [np.linalg.norm(res)]
    for i in range(niter):
        sol, res = ssor(A, omega, sol, res)
        resnorm.append(np.linalg.norm(res))
        #print(f"Iteração {i+1}: Norma do resíduo = {resnorm[i]}")
    return sol, resnorm

def iterative_conjugate_gradient(A, B, solini, niter):
    sol = solini.copy()
    res = B - A.multiply_sparse_vector(sol)
    p = res.copy()
    resnorm = [np.linalg.norm(res)]
    
    for i in range(niter):
        Ap = A.multiply_sparse_vector(p)
        alpha = np.dot(res, res) / np.dot(p, Ap)
        sol = sol + alpha * p
        res_new = res - alpha * Ap
        resnorm.append(np.linalg.norm(res_new))
        
        # if np.linalg.norm(res_new) < 1e-20:
        #     break
        
        beta = np.dot(res_new, res_new) / np.dot(res, res)
        p = res_new + beta * p
        res = res_new

    return sol, resnorm

# Lendo a matriz do arquivo .dat 
sparse_matrix_file = SparseMatrix(file_path='mathematica.dat')
sparse_matrix_file.print_matrix()

print(f"Índice de Esparsidade: {sparse_matrix_file.sparsity_index()} %")

B = np.ones(sparse_matrix_file.rows)

print("Tamanho do vetor:", len(B))
print("Número de colunas da matriz:", sparse_matrix_file.cols)

# Solução inicial
solini = np.zeros_like(B)

# Número de iterações
niter = 100

# Chamando a função para resolver o sistema
sol, resnorm = iterative_jacobi(sparse_matrix_file, B, solini, niter+1)

sparse_matrix_file.print_matrix()

# Exibindo a solução e a norma do resíduo em cada iteração
#print("Solução:", sol)
#print("Norma do resíduo em cada iteração:", resnorm)



# Chamando a função para resolver o sistema
sol_jacobi_itera, resnorm_jacobi_itera = iterative_jacobi(sparse_matrix_file, B, solini, niter)

sol_ssor, resnorm_ssor = iterative_ssor(sparse_matrix_file, B, solini, 1.5, niter)


sol_grad, resnorm_grad = iterative_conjugate_gradient(sparse_matrix_file, B, solini, niter)


sol_jacobi_p, resnorm_jacobi_p = iterative_conjugate_gradient_pc(sparse_matrix_file, B, solini, niter, jacobi)


sol_ssor_p, resnorm_ssor_p = iterative_conjugate_gradient_pc(sparse_matrix_file, B, solini, niter, ssor)

def plot_graph(x, y, label, filename):
    plt.scatter(x, y, label=label, marker=',',s=5)
    plt.legend()
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('N° iterações')
    plt.ylabel('resnorm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.close()

#niter = 100
x = np.linspace(1, niter + 1, niter+1)
y_data = [
    (resnorm_jacobi_itera, 'resnorm_jacobi_itera', '/home/ingrid/L4/Figuras/jacobi_iter_validacao.pdf'),
    (resnorm_ssor, 'resnorm_ssor', '/home/ingrid/L4/Figuras/ssor_validacao.pdf'),
    (resnorm_grad, 'resnorm_grad', '/home/ingrid/L4/Figuras/grad_validacao.pdf'),
    (resnorm_jacobi_p, 'resnorm_jacobi_p', '/home/ingrid/L4/Figuras/jacobi_p_validacao.pdf'),
    (resnorm_ssor_p, 'resnorm_ssor_p', '/home/ingrid/L4/Figuras/ssor_p_validacao.pdf')
]

for y, label, filename in y_data:
    plot_graph(x, y, label, filename)

y_data1 = [
    (resnorm_ssor, 'resnorm_ssor', '/home/ingrid/L4/Figuras/ssor_validacao.pdf'),
    (resnorm_grad, 'resnorm_grad', '/home/ingrid/L4/Figuras/grad_validacao.pdf'),
    (resnorm_jacobi_p, 'resnorm_jacobi_p', '/home/ingrid/L4/Figuras/jacobi_p_validacao.pdf'),
    (resnorm_ssor_p, 'resnorm_ssor_p', '/home/ingrid/L4/Figuras/ssor_p_validacao.pdf')
]

# Gráfico completo
plt.figure()
for y, label, _ in y_data:
    plt.scatter(x, y, label=label)

plt.legend()
plt.xscale('linear')
plt.yscale('log')
plt.xlabel('N° iterações')
plt.ylabel('resnorm')
plt.grid(True)
plt.tight_layout()
plt.savefig('/home/ingrid/L4/Figuras/completo_validacao.pdf', format='pdf')
plt.close()





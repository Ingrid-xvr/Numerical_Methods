import numpy as np
import teste as t

def jacobi(A, omega, u, res):
    Diag = np.diag(A)
    resloc = res.copy()
    delu = np.zeros_like(u)
    delu = omega * res / Diag
    resloc -= np.dot(A, delu)
    #resloc = resloc.astype(res.dtype)  # Convertendo para o mesmo tipo de dados que res
    return u + delu, resloc

def iterative_jacobi(A, B, solini, niter):
    sol = solini.copy()
    res = B - np.dot(A, sol)
    resnorm = [np.linalg.norm(res)]
    for i in range(0, niter):
        sol, res = jacobi(A, 1, sol, res)  # omega = 1
        resnorm.append(np.linalg.norm(res))
    return sol, resnorm

# Matriz de coeficientes A
# A = np.array([[4.0, -1.0, 0.0, 0.0],
#               [-1.0, 4.0, -1.0, 0.0],
#               [0.0, -1.0, 4.0, -1.0],
#               [0.0, 0.0, -1.0, 3.0]])

A = t.A

# Vetor de termos independentes B
B = np.ones(A.shape[0])

# Solução inicial
solini = np.zeros_like(B)

# Número de iterações
niter = 100

# Chamando a função para resolver o sistema
sol, resnorm = iterative_jacobi(A, B, solini, niter)

# Exibindo a solução e a norma do resíduo em cada iteração
print("Solução:", sol)
print("Norma do resíduo em cada iteração:", resnorm)

mathematica_sol = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.27616, 2.05283, 2.52624, 2.7841, 2.86618, 2.7841, 2.52624, 2.05283, 1.27616, 1.0, 1.0, 2.05283, 3.41084, 4.27067, 4.7471, 4.89976, 4.7471, 4.27067, 3.41084, 2.05283, 1.0, 1.0, 2.52624, 4.27067, 5.4035, 6.03978, 6.24484, 6.03978, 5.4035, 4.27067, 2.52624, 1.0, 1.0, 2.7841, 4.7471, 6.03978, 6.7718, 7.00859, 6.7718, 6.03978, 4.7471, 2.7841, 1.0, 1.0, 2.86618, 4.89976, 6.24484, 7.00859, 7.25595, 7.00859, 6.24484, 4.89976, 2.86618, 1.0, 1.0, 2.7841, 4.7471, 6.03978, 6.7718, 7.00859, 6.7718, 6.03978, 4.7471, 2.7841, 1.0, 1.0, 2.52624, 4.27067, 5.4035, 6.03978, 6.24484, 6.03978, 5.4035, 4.27067, 2.52624, 1.0, 1.0, 2.05283, 3.41084, 4.27067, 4.7471, 4.89976, 4.7471, 4.27067, 3.41084, 2.05283, 1.0, 1.0, 1.27616, 2.05283, 2.52624, 2.7841, 2.86618, 2.7841, 2.52624, 2.05283, 1.27616, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

mathematica_resnorm = np.array([11.0, 8.10864, 7.51977, 7.04616, 6.63867, 6.27547, 5.94465, 5.63899, 5.35384, 5.08611, 4.83363, 4.59484, 4.36858, 4.15391, 3.95007, 3.75641, 3.57235, 3.39738, 3.23102, 3.07283, 2.9224, 2.77935, 2.64331, 2.51393, 2.39088, 2.27386, 2.16257, 2.05673, 1.95606, 1.86032, 1.76927, 1.68268, 1.60032, 1.522, 1.44751, 1.37666, 1.30928, 1.2452, 1.18426, 1.12629, 1.07117, 1.01874, 0.968882, 0.921461, 0.876362, 0.833469, 0.792677, 0.75388, 0.716983, 0.681891, 0.648517, 0.616776, 0.586589, 0.557879, 0.530575, 0.504607, 0.479909, 0.456421, 0.434082, 0.412837, 0.392631, 0.373414, 0.355138, 0.337756, 0.321225, 0.305504, 0.290551, 0.276331, 0.262806, 0.249943, 0.23771, 0.226076, 0.215011, 0.204488, 0.194479, 0.184961, 0.175908, 0.167299, 0.15911, 0.151323, 0.143917, 0.136873, 0.130174, 0.123803, 0.117743, 0.111981, 0.1065, 0.101287, 0.09633, 0.0916153, 0.0871313, 0.0828668, 0.078811, 0.0749537, 0.0712852, 0.0677963, 0.0644781, 0.0613223, 0.058321, 0.0554666, 0.0527518])

def compare_solutions(sol1, sol2):
    return np.allclose(sol1, sol2)

are_equal = compare_solutions(mathematica_sol, sol)
print("As soluções são iguais:", are_equal)

#verif2 = mathematica_resnorm - resnorm
#print("Diferença entre a norma do resíduo obtida e a norma do resíduo do Mathematica:", verif2)

import numpy as np
import teste as t
#from ler_converter import SparseMatrix
import re

# Função para o método Jacobi
def jacobi(A, omega, u, res):
    Diag = np.diag(A)

    #print("res:", res)
    #print("Diag:", Diag)

    resloc = res.copy()
    delu = np.zeros_like(u)
    delu = omega * res / Diag
    resloc -= np.dot(A, delu)
    #print("resloc:", resloc)
    return u + delu, resloc

# Função iterativa para o método Jacobi
def iterative_jacobi(A, B, solini, niter):
    sol = solini.copy()
    res = B - np.dot(A, sol)
    resnorm = [np.linalg.norm(res)]
    for i in range(0, niter):
        sol, res = jacobi(A, 1, sol, res)  # omega = 1
        resnorm.append(np.linalg.norm(res))
    return sol, resnorm

def gauss_seidel_f(A, omega, u, res):
    n = len(res)
    delu = np.zeros_like(u)
    resc = res.copy()
    delu[0] = omega * res[0] / A[0, 0]
    for i in range(1, n):
        resc[i] -= np.dot(A[i, :i], delu[:i])
        delu[i] += omega * resc[i] / A[i, i]
    resc = res - np.dot(A, delu)
    return u + delu, resc

def gauss_seidel_b(A, omega, u, res):
    n = len(res)
    delu = np.zeros_like(u)
    resc = res.copy()
    delu[n-1] = omega * res[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        resc[i] -= np.dot(A[i, i+1:], delu[i+1:])
        delu[i] += omega * resc[i] / A[i, i]
    resc = res - np.dot(A, delu)
    return u + delu, resc

def ssor(A, omega, u, res):
    delu, resc = gauss_seidel_f(A, omega, u, res)
    delu2, resc2 = gauss_seidel_b(A, omega, delu, resc)
    return delu2, resc2

def iterative_ssor(A, B, solini, omega, niter):
    sol = solini.copy()
    res = B - np.dot(A, sol)
    resnorm = [np.linalg.norm(res)]
    for i in range(niter):
        sol, res = ssor(A, omega, sol, res)
        resnorm.append(np.linalg.norm(res))
        #print(f"Iteração {i+1}: Norma do resíduo = {resnorm[i]}")
    return sol, resnorm

def iterative_conjugate_gradient_pc(A, B, solini, niter, method):
    neq = len(solini)
    sol = solini.copy()
    zero = np.zeros_like(solini)
    res = B - np.dot(A,sol)
    z, _ = method(A, 1, zero, res)
    zk = z.copy()
    p = z.copy()
    resk = res.copy()
    resnorm = [np.linalg.norm(res)]

    eign = np.linalg.eigvals(A)
    print("Autovalores de A:")
    print(*np.round(eign, 4), sep=", ")
    print("Número de condicionamento da matriz original: ", np.linalg.cond(A))
    print("Maior autovalor ", np.max(eign))
    print("Menor autovalor ", np.min(eign))
    Z = np.zeros((neq, neq))
    Z = np.diag(z)
    eign1 = np.linalg.eigvals(Z)
    print("Número de condicionamento da matriz precondicionada: ", np.linalg.cond(Z))
    print("Autovalores da matriz A precondicionada: ")
    print(*np.round(np.linalg.eigvals(Z), 4), sep=", ")
    print("Maior autovalor ", np.max(eign1))
    print("Menor autovalor ", np.min(eign1))

    for i in range(niter):
        Ap = np.dot(A, p)
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

def iterative_conjugate_gradient(A, B, solini, niter):
    sol = solini.copy()
    res = B - np.dot(A, sol)
    p = res.copy()
    resnorm = [np.linalg.norm(res)]
    
    for i in range(niter):
        Ap = np.dot(A, p)
        alpha = np.dot(res, res) / np.dot(p, Ap)
        sol = sol + alpha * p
        res_new = res - alpha * Ap
        resnorm.append(np.linalg.norm(res_new))
        
        # if np.linalg.norm(res_new) < 1e-90:
        #     break
        
        beta = np.dot(res_new, res_new) / np.dot(res, res)
        p = res_new + beta * p
        res = res_new

    return sol, resnorm




# Comparar soluções
def compare_solutions(sol1, sol2):
    return np.allclose(sol1, sol2)

# Importar a matriz A do arquivo teste.py
A = t.A

# Vetor de termos independentes B
B = np.ones(A.shape[0])

#print("Tamanho do vetor:", len(B))

# Solução inicial
solini = np.zeros_like(B)

# Número de iterações
niter = 100

# Chamando a função para resolver o sistema
sol_jacobi, resnorm_jacobi = iterative_jacobi(A, B, solini, niter)

sol_ssor, resnorm_ssor = iterative_ssor(A, B, solini, 1.5, niter)

sol_grad, resnorm_grad = iterative_conjugate_gradient(A, B, solini, niter)

sol_jacobi_p, resnorm_jacobi_p = iterative_conjugate_gradient_pc(A, B, solini, niter, ssor)


# Exibindo a solução e a norma do resíduo em cada iteração
# print("Solução:", sol_jacobi)
# print("Norma do resíduo em cada iteração:", resnorm_jacobi)

# Dados do Mathematica
mathematica_sol_jacobi = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.27616, 2.05283, 2.52624, 2.7841, 2.86618, 2.7841, 2.52624, 2.05283, 1.27616, 1.0, 1.0, 2.05283, 3.41084, 4.27067, 4.7471, 4.89976, 4.7471, 4.27067, 3.41084, 2.05283, 1.0, 1.0, 2.52624, 4.27067, 5.4035, 6.03978, 6.24484, 6.03978, 5.4035, 4.27067, 2.52624, 1.0, 1.0, 2.7841, 4.7471, 6.03978, 6.7718, 7.00859, 6.7718, 6.03978, 4.7471, 2.7841, 1.0, 1.0, 2.86618, 4.89976, 6.24484, 7.00859, 7.25595, 7.00859, 6.24484, 4.89976, 2.86618, 1.0, 1.0, 2.7841, 4.7471, 6.03978, 6.7718, 7.00859, 6.7718, 6.03978, 4.7471, 2.7841, 1.0, 1.0, 2.52624, 4.27067, 5.4035, 6.03978, 6.24484, 6.03978, 5.4035, 4.27067, 2.52624, 1.0, 1.0, 2.05283, 3.41084, 4.27067, 4.7471, 4.89976, 4.7471, 4.27067, 3.41084, 2.05283, 1.0, 1.0, 1.27616, 2.05283, 2.52624, 2.7841, 2.86618, 2.7841, 2.52624, 2.05283, 1.27616, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
mathematica_resnorm_jacobi = np.array([11.0, 8.10864, 7.51977, 7.04616, 6.63867, 6.27547, 5.94465, 5.63899, 5.35384, 5.08611, 4.83363, 4.59484, 4.36858, 4.15391, 3.95007, 3.75641, 3.57235, 3.39738, 3.23102, 3.07283, 2.9224, 2.77935, 2.64331, 2.51393, 2.39088, 2.27386, 2.16257, 2.05673, 1.95606, 1.86032, 1.76927, 1.68268, 1.60032, 1.522, 1.44751, 1.37666, 1.30928, 1.2452, 1.18426, 1.12629, 1.07117, 1.01874, 0.968882, 0.921461, 0.876362, 0.833469, 0.792677, 0.75388, 0.716983, 0.681891, 0.648517, 0.616776, 0.586589, 0.557879, 0.530575, 0.504607, 0.479909, 0.456421, 0.434082, 0.412837, 0.392631, 0.373414, 0.355138, 0.337756, 0.321225, 0.305504, 0.290551, 0.276331, 0.262806, 0.249943, 0.23771, 0.226076, 0.215011, 0.204488, 0.194479, 0.184961, 0.175908, 0.167299, 0.15911, 0.151323, 0.143917, 0.136873, 0.130174, 0.123803, 0.117743, 0.111981, 0.1065, 0.101287, 0.09633, 0.0916153, 0.0871313, 0.0828668, 0.078811, 0.0749537, 0.0712852, 0.0677963, 0.0644781, 0.0613223, 0.058321, 0.0554666, 0.0527518])

# Dados do Mathematica para SSOR
mathematica_sol_ssor = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.28131, 2.06262, 2.53971, 2.79994, 2.88283, 2.79994, 2.53971, 2.06262, 1.28131, 1., 1., 2.06262, 3.42946, 4.29629, 4.77723, 4.93143, 4.77723, 4.29629, 3.42946, 2.06262, 1., 1., 2.53971, 4.29629, 5.43877, 6.08125, 6.28844, 6.08125, 5.43877, 4.29629, 2.53971, 1., 1., 2.79994, 4.77723, 6.08125, 6.82054, 7.05984, 6.82054, 6.08125, 4.77723, 2.79994, 1., 1., 2.88283, 4.93143, 6.28844, 7.05984, 7.30984, 7.05984, 6.28844, 4.93143, 2.88283, 1., 1., 2.79994, 4.77723, 6.08125, 6.82054, 7.05984, 6.82054, 6.08125, 4.77723, 2.79994, 1., 1., 2.53971, 4.29629, 5.43877, 6.08125, 6.28844, 6.08125, 5.43877, 4.29629, 2.53971, 1., 1., 2.06262, 3.42946, 4.29629, 4.77723, 4.93143, 4.77723, 4.29629, 3.42946, 2.06262, 1., 1., 1.28131, 2.06262, 2.53971, 2.79994, 2.88283, 2.79994, 2.53971, 2.06262, 1.28131, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

mathematica_resnorm_ssor = np.array([11., 5.93824, 3.67816, 2.39459, 1.57782, 1.04255, 0.689156, 0.455444, 0.300867, 0.198669, 0.131135, 0.0865289, 0.0570802, 0.0376453, 0.0248231, 0.0163658, 0.0107887, 0.00711143, 0.0046872, 0.00308919, 0.0020359, 0.00134169, 0.000884168, 0.00058265, 0.000383948, 0.000253006, 0.000166719, 0.000109859, 0.0000723905, 0.0000477009, 0.0000314318, 0.0000207114, 0.0000136474, 8.99267*10**-6,  5.92553*10**-6, 3.90449*10**-6, 2.57277*10**-6, 1.69527*10**-6,  1.11706*10**-6, 7.36057*10**-7, 4.85006*10**-7, 3.19583*10**-7,  2.10581*10**-7, 1.38757*10**-7, 9.14303*10**-8, 6.02457*10**-8,  3.96973*10**-8, 2.61575*10**-8, 1.72358*10**-8, 1.13571*10**-8, 
 7.48347*10**-9, 4.93104*10**-9, 3.24918*10**-9, 2.14096*10**-9,  1.41073*10**-9, 9.29566*10**-10, 6.12513*10**-10, 4.036*10**-10,  2.65942*10**-10, 1.75235*10**-10, 1.15467*10**-10, 7.60838*10**-11,  5.01334*10**-11, 3.30341*10**-11, 2.1767*10**-11, 1.43428*10**-11,  9.45081*10**-12, 6.22736*10**-12, 4.10336*10**-12, 2.7038*10**-12,  1.7816*10**-12, 1.17394*10**-12, 7.73537*10**-13, 5.09702*10**-13,  3.35855*10**-13, 2.21303*10**-13, 1.45822*10**-13, 9.60855*10**-14,  6.3313*10**-14, 4.17185*10**-14, 2.74893*10**-14, 1.81134*10**-14,  1.19353*10**-14, 7.86448*10**-15, 5.18209*10**-15, 3.4146*10**-15,  2.24997*10**-15, 1.48256*10**-15, 9.76892*10**-16, 6.43698*10**-16,  4.24148*10**-16, 2.79481*10**-16, 1.84157*10**-16, 1.21345*10**-16,  7.99574*10**-17, 5.26858*10**-17, 3.4716*10**-17, 2.28752*10**-17,  1.5073*10**-17, 9.93197*10**-18, 6.54441*10**-18])

mathematica_sol_grad = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.28131, 2.06262, 2.53971, 2.79994, 2.88283, 2.79994, 2.53971, 2.06262, 1.28131, 1., 1., 2.06262, 3.42946, 4.29629, 4.77723, 4.93143, 4.77723, 4.29629, 3.42946, 2.06262, 1., 1., 2.53971, 4.29629, 5.43877, 6.08125, 6.28844, 6.08125, 5.43877, 4.29629, 2.53971, 1., 1., 2.79994, 4.77723, 6.08125, 6.82054, 7.05984, 6.82054, 6.08125, 4.77723, 2.79994, 1., 1., 2.88283, 4.93143, 6.28844, 7.05984, 7.30984, 7.05984, 6.28844, 4.93143, 2.88283, 1., 1., 2.79994, 4.77723, 6.08125, 6.82054, 7.05984, 6.82054, 6.08125, 4.77723, 2.79994, 1., 1., 2.53971, 4.29629, 5.43877, 6.08125, 6.28844, 6.08125, 5.43877, 
4.29629, 2.53971, 1., 1., 2.06262, 3.42946, 4.29629, 4.77723, 4.93143, 4.77723, 4.29629, 3.42946, 2.06262, 1., 1., 1.28131, 2.06262, 2.53971, 2.79994, 2.88283, 2.79994, 2.53971, 2.06262, 1.28131, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

mathematica_resnorm_grad = np.array([11., 9.58766, 10.2429, 9.42444, 5.05636, 2.19319, 0.762956, 0.345221, 0.222629, 0.127196, 0.0518932, 0.0234696, 0.00960064, 0.00185361, 4.11181*10**-17, 5.40005*10**-17, 1.63011*10**-17, 2.28124*10**-17, 1.83679*10**-17, 1.06918*10**-17, 7.79379*10**-18, 3.62015*10**-18, 1.95526*10**-18, 9.1634*10**-19, 5.32364*10**-19, 2.28229*10**-19, 1.35821*10**-19, 6.99142*10**-20, 4.40938*10**-20, 2.06858*10**-20, 
8.7996*10**-21, 4.47032*10**-21, 2.44769*10**-21, 7.78856*10**-22, 4.30171*10**-22, 1.75606*10**-22, 8.61457*10**-23, 4.93066*10**-23, 1.83134*10**-23, 7.46848*10**-24, 2.03253*10**-24, 6.07529*10**-25, 1.58051*10**-25, 4.50954*10**-26, 1.21101*10**-26, 4.50421*10**-27, 1.47924*10**-27, 5.16809*10**-28, 8.54395*10**-29, 1.96639*10**-29, 1.36267*10**-30, 1.26732*10**-31, 2.21126*10**-32, 1.75336*10**-33, 8.00434*10**-34, 1.61673*10**-34, 7.08602*10**-35, 1.22263*10**-34, 2.33312*10**-35, 2.18944*10**-36, 3.05342*10**-38, 8.65172*10**-40, 1.14231*10**-39, 1.43991*10**-41, 1.38033*10**-41, 5.09839*10**-43, 4.85082*10**-43, 5.95496*10**-44, 1.38396*10**-44, 7.17053*10**-45, 1.26819*10**-45, 4.76792*10**-46, 8.33548*10**-47, 3.55464*10**-47, 1.34379*10**-47, 1.34387*10**-48, 7.26912*10**-49, 1.86354*10**-49, 5.02925*10**-50, 1.44105*10**-50, 5.56304*10**-51, 4.58386*10**-51, 
7.05731*10**-51, 3.49828*10**-51, 1.36177*10**-51, 4.79987*10**-52, 6.82147*10**-53, 1.18072*10**-53, 2.45038*10**-54, 1.64838*10**-54, 1.36316*10**-54, 2.79664*10**-55, 3.96596*10**-56, 9.09007*10**-57, 1.25705*10**-57, 2.70519*10**-58, 4.87623*10**-59, 1.65339*10**-59, 2.22804*10**-60, 1.38346*10**-60, 5.50359*10**-61, 7.59015*10**-62, 1.44993*10**-62, 3.11549*10**-63, 4.53521*10**-64, 3.00286*10**-64, 2.9784*10**-65, 1.12192*10**-65, 1.04706*10**-65, 6.14511*10**-67, 2.46513*10**-67, 1.30696*10**-67, 2.45813*10**-68, 1.20406*10**-68, 5.23175*10**-69, 3.63399*10**-71, 1.30505*10**-71, 2.65039*10**-72, 1.55263*10**-72, 9.46723*10**-73, 1.28854*10**-73, 7.44938*10**-74, 1.19922*10**-74, 3.68948*10**-75, 8.17339*10**-76, 8.45931*10**-77, 2.45632*10**-77, 8.55733*10**-78, 2.19844*10**-80, 8.56804*10**-81, 6.45314*10**-81, 3.48396*10**-81, 1.41818*10**-81, 1.36986*10**-82, 
3.12187*10**-83, 1.04186*10**-83, 1.1064*10**-83, 1.43726*10**-83, 2.55747*10**-84, 6.02236*10**-85, 1.8788*10**-85, 1.16687*10**-85, 1.83962*10**-86, 9.8394*10**-87, 2.85519*10**-87, 2.17281*10**-87, 2.05378*10**-88, 4.9625*10**-89, 1.27269*10**-89, 6.9221*10**-90, 1.3782*10**-90])

# Comparar soluções Jacobi
are_equal_jacobi = compare_solutions(mathematica_sol_jacobi, sol_jacobi)
print("As soluções Jacobi são iguais:", are_equal_jacobi)

are_equal_jacobi_resnorm = compare_solutions(mathematica_resnorm_jacobi, resnorm_jacobi)
print("resnorm Jacobi são iguais:", are_equal_jacobi_resnorm)

print("tamanho mathematica jacobi:", len(mathematica_sol_jacobi))
print("tamanho python jacobi:", len(resnorm_jacobi))

print("tamanho mathematica ssor:", len(mathematica_sol_ssor))
print("tamanho python ssor:", len(resnorm_ssor))

print("tamanho mathematica grad:", len(mathematica_sol_grad))
print("tamanho python grad:", len(resnorm_grad))

are_equal_ssor = compare_solutions(mathematica_sol_ssor, sol_ssor)
print("As soluções SSOR são iguais:", are_equal_ssor)
are_equal_ssor_resnorm = compare_solutions(mathematica_resnorm_ssor, resnorm_ssor)
print("resnorm SSOR são iguais:", are_equal_ssor_resnorm)

are_equal_grad = compare_solutions(mathematica_sol_grad, sol_grad)
print("As soluções Grad. Conjugado são iguais:", are_equal_grad)



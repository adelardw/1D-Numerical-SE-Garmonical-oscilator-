import numpy as np
import matplotlib.pyplot as plt
import math

# input
p = 50
x_min = -20
x_max = 20
h = (x_max - x_min) / (p)
x0 = np.linspace(x_min, x_max, p + 1)
w = 1
m = 1
h_pl = 1


# Potential func
def U(x):
    return (w * x ** 2) / 2


u = [U(i) for i in x0]
plt.figure(0)
plt.xlabel('x')
plt.ylabel('U(x)')
plt.plot(x0, u)


# Hamilton operator:
def Hamilton():
    ham = np.zeros((p + 1, p + 1))
    for i in range(p):
        ham[i, i] = 2 * U(x0[i]) + 2 / (h ** 2)
        ham[i + 1, i] = ham[i, i + 1] = -1 / (h ** 2)
    ham[p, p] = 2 * U(x0[p]) * (h ** 2) + 2 / (h ** 2)
    return ham


a = Hamilton()

"""
Eig_val; Eig_vec in numpy
"""
eva, eve = np.linalg.eigh(a)

"""
Analytic solve
"""


# Hermite polynomial.
def Hermite_pol(x, n):
    if n == 0:
        return 1
    if n == 1:
        return 2 * x
    else:
        return 2 * x * Hermite_pol(x, n - 1) - 2 * (n - 1) * Hermite_pol(x, n - 2)


# WF Harmonic oscillator
def psy(x, n):
    return (1 / ((math.factorial(n) * (2 ** n) * (math.pi) ** 0.5) ** 0.5)) * Hermite_pol(x, n) * math.exp(
        -(x ** 2) / 2)


psy_h = [psy(i, 1) for i in x0]

"""
QR - algorithm
"""


# Numerical QR_decomposition
def projection_operator(A: list, B: list):
    return (np.dot(A, B) / np.dot(B, B)) * B


def QR_decompositon(A: list):
    # Gram - Schmidt orthonormalize vector process
    n = len(A)
    Q = [A.T[0] / np.dot(A.T[0], A.T[0]) ** 0.5]
    for i in range(1, n):
        sum_vect = 0
        for k in range(0, i):
            sum_vect += projection_operator(A.T[i], Q[k])
        project = A.T[i] - sum_vect
        Q.append(project / np.dot(project, project) ** 0.5)
    # Q and R - matrix
    Q = np.array(Q).T
    R = np.around(np.dot(Q.T, A), 10)
    return Q, R


# QR - algorithm
def QR_algorithm(A: list):
    k = 1000
    multiply_by_Q = 1
    for i in range(k):
        q, r = QR_decompositon(A)
        A = np.dot(r, q)
        multiply_by_Q = np.dot(multiply_by_Q, q)
    eigen_values = sorted([A[i][i] for i in range(len(A))])
    eigen_vect_matrix = np.array(multiply_by_Q.T)
    return np.array(eigen_values), eigen_vect_matrix


pl1, pl2 = QR_algorithm(a)

# Test
print("Eigen values; QR - algorithm \n", pl1, "\n")
print("Eigen values; Numpy \n", eva, "\n")
print("Eigen vect ground state; QR - algorithm \n", pl2[p], "\n")
print("Eigen vect ground state; Numpy \n", eve.T[0])

d = []
for i in range(1, p + 1):
    d.append(eva[i] - eva[i - 1])
d = np.array(sorted(np.around(d, 1)))
print("\n Checking the spectrum equidistant condition :\n", list(d), '\n')

plt.figure(1)
plt.xlabel('x')
plt.ylabel('ψ(x)')
plt.plot(x0, -pl2[p - 1], color='red', label=f'Numerical Methods:{1} -st excited state ')
plt.plot(x0, eve.T[1], "o", color='blue', markersize=5, label=f'Numpy :{1} -st excited state ')
plt.plot(x0, psy_h, markersize=5, color='black', label=f'Analytic:{1} -st excited state')
plt.legend(bbox_to_anchor=(1 / 2, 1.15), loc="upper center")
plt.show()

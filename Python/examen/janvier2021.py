import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
A = np.array([[66, 1, -9, -2],
              [1, 35, -5, -9],
              [-9, -5, 50, 27],
              [-2, -9, 27, 85]])
b = np.array([97, 95, -196,-186 ])
print(np.linalg.solve(A,b))
X0=np.ones((4,1))
Tol=10**-6
w=1
def iterative(A, b, X0, w, Tol):
    D = np.diag(A)
    E = -np.tril(A, -1)
    F = -np.triu(A, 0)
    M = (1/w)*D-E
    N = ((1-w)/w)*D+F
    invM = np.linalg.inv(M)
    B = invM.dot(N)
    C = invM.dot(b)
    k = 0
    while(np.linalg.norm(A.dot(X0)-b,1) > Tol):
        X0 = B.dot(X0)+C
        k += 1
    return X0, k
print(iterative(A, b, X0, w, Tol)[1])

W=np.linspace(1/5,9/5,1)
N_it=[]
for w in W:
    sol=iterative(A, b, X0,w, Tol)
    N_it.append(sol[1])
    
plt.figure(figsize=(20,10))
plt.plot(W,N_it,'ro--', linewidth=3,markersize=12)
plt.yscale('log')
#plt.show()
#t=sp.symbols('x')
#print(sp.integrate((t**2)*np.exp**t,t))
#x, y = sp.symbols('xy')
#print(sp.integrate(6*x**5, x))
t,e= sp.symbols('t e')
  
# Using sympy.primitive() method
gfg = sp.primitive((pow(t,2))*(pow(e,t)))
  
print(gfg)
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
# definir une matrice
A = np.array([[66, 1, -9],
              [1, 35, -5],
              [-9, -5, 50],
              [-2, -9, 27]])

# taille d'une matrice
n = A.shape[0]

#Multiplie les elements d'une matrice
A.dot(n)

# Mettre En Valeur Absolue
p = np.abs(A[2, 0])

#Somme de plusieur element d'une matrice
p= np.sum(A[0,:])

#matrice diagonale de 1 
np.eye(4)

#cree une matrice de 1 
np.ones((4,5))

#cree une matrice avec la digonale au choix avec décalage
np.diagflat([1,2,3],4)

#inverser une matrice
np.inv() 

#norme
np.norm()
#Donne la diagonale de la matrice 
np.diag(x)

#changer la forme d'une matrice
np.reshape(A, (2, 3)) # C-like index ordering

#Resoudre l'equation Ax=b
np.linalg.solve(A,b)

#Sasir un Polynome
P=Polynomial([1,0,6,1]) #P(X)=1+6*X^2+1*X^3=1+0*X+6*X^2+1*X^3.
#Les coeficient d'un polynome 
p.coef

#charger une base de donnée 
df = pd.read_csv('cons_veh.csv')

# Pour afficher le nombre de lignes et de colonnes
df.shape

# la méthode head() affiche les 5 premières lignes par defaut
df.head()

# tail() affiche les 5 dernières lignes. 
#Pour voir moins ou plusieurs lignes, passer un entier, par exemple: tail(3)
df.tail(10)

# afficher les noms de colonnes 
df.columns

#courbe
pd.plotting.scatter_matrix(df,alpha=1, figsize=(15,12));

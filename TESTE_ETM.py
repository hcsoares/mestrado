import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
nu = 0.001
E = 1e-3

# Matrizes do exemplo Cabral
A = np.array([[1.01, 0.1], [0.1, 0.98]])
B = np.array([[1], [0]])
K = np.array([[-1.29779646, 0.01910723]])

F1 = A + B @ K
F2 = np.block([[-B, B]])
F3 = np.vstack([K, -K])
F4 = np.zeros_like(F1)
F5 = np.array([[-10], [-5]])

n = 2
ny = 2
nz = ny


rd = 30 # Defina o valor de rd aqui

xc =  np.array([[-12],[0]])# Defina o valor de xc aqui

#%%
IS = np.block([[np.ones([1,1]),np.zeros([1,n]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,n])],
               [np.zeros([n,1]),np.eye(n),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,n])],
               [np.zeros([nz,1]),np.zeros([nz,n]),np.eye(nz),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,n])],
               [np.zeros([nz,1]),np.zeros([nz,n]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.eye(nz),np.zeros([nz,nz]),np.zeros([nz,n])]
               ])

IS2 = np.block([[np.ones([1,1]),np.zeros([1,n]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,nz]),np.zeros([1,2*n])],
               [np.zeros([n,1]),np.eye(n),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,nz]),np.zeros([n,2*n])],
               [np.zeros([nz,1]),np.zeros([nz,n]),np.eye(nz),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,2*n])],
               [np.zeros([nz,1]),np.zeros([nz,n]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.eye(nz),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,2*n])],
               [np.zeros([nz,1]),np.zeros([nz,n]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.eye(nz),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,2*n])],
               [np.zeros([nz,1]),np.zeros([nz,n]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.zeros([nz,nz]),np.eye(nz),np.zeros([nz,nz]),np.zeros([nz,2*n])],
               ])
#%%
nx = n
nz = ny
# ETM
Qx = cp.Variable((nx, nx), symmetric=True)
Qphi = cp.Variable((nz, nz), symmetric=True)
Qdel = cp.Variable((nx, nx), symmetric=True)
# Qsig = cp.Variable((nx, nx), symmetric=True)  # Não mais utilizado
Qphi_ = cp.bmat([[Qphi, np.zeros((nz, nz))], [np.zeros((nz, nz)), np.zeros((nz, nz))]])
Qdel_ = cp.bmat([[Qdel, np.zeros((nz, nz))], [np.zeros((nz, nz)), np.zeros((nz, nz))]])

# Restrições das matrizes do ETM


T0 = cp.diag(cp.Variable(ny))
R0 = cp.Variable((1 + n + 2 * ny, ny))
MA0 = cp.Variable((1 + 2 * ny, 1 + 2 * ny), symmetric=True)
# Nx = 1 + nx + 2 * ny
# Ha0 = cp.Variable((Nx, Nx), symmetric=True)
# Hb0 = cp.Variable((2 * ny, Nx))

I1 = np.block([
    [1, np.zeros((1, ny)), np.zeros((1, ny))],
    [np.zeros((n, 1)), np.zeros((n, ny)), np.zeros((n, ny))],
    [np.zeros((ny, 1)), np.eye(ny), np.zeros((ny, ny))],
    [np.zeros((ny, 1)), np.zeros((ny, ny)), np.eye(ny)]
]).T

XI1 = np.block([[F5, F3, F4 - np.eye(ny), np.eye(ny)]])

S1_0 = cp.bmat([
    [np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny))],
    [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, ny)), np.zeros((n, ny))],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), T0],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny))]
])

Q_T0 = cp.bmat([
    [np.zeros((1, 1)),  np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny)), np.zeros((1, ny)), np.zeros((1, ny)),  np.zeros((1, n))],
    [np.zeros((n, 1)),  -Qx, np.zeros((n,ny)), np.zeros((n,ny)), np.zeros((n,ny)), np.zeros((n, ny)),  np.zeros((n, n))],
    [np.zeros((ny, 1)),  np.zeros((ny,n)), -Qphi, np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)),  np.zeros((ny, n))],
    [np.zeros((ny, 1)),  np.zeros((ny,n)), np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)),  np.zeros((ny, n))],
    [np.zeros((ny, 1)),  np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)),  np.zeros((ny, n))],
    [np.zeros((ny, 1)),  np.zeros((ny,n)), np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)), np.zeros((ny, ny)),  np.zeros((ny, n))],
    [np.zeros((n, 1)),  np.zeros((n, n)), np.zeros((n, ny)), np.zeros((n, ny)), np.zeros((n, ny)), np.zeros((n, ny)),  0*Qdel] #degradação
])

LMIETM = (S1_0+R0@XI1)+(S1_0+R0@XI1)-I1.T@MA0@I1
LMIETM = Q_T0+IS.T@LMIETM@IS
#%%LMI1

# Definições das variáveis de decisão
P1 = cp.Variable((n, n), symmetric=True)
P2 = cp.Variable((n, ny))
P3 = cp.Variable((ny, ny), symmetric=True)

#matrizes que dão a maior RA para K = np.array([[-1.29779646, 0.01910723]]) obtido
#pelo DE, minimizando o traço de P1, r = 21.
# P2 = np.array([[0.00579002,0.01326095],[-0.00031506,-0.00089166]])
# P3 = np.array([[0.013648,0.00609129],[0.00609129,-0.01732456]])

H1 = cp.bmat([
    [np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny))],
    [np.zeros((n, 1)), P1 - E * np.eye(n), P2, np.zeros((n, ny))],
    [np.zeros((ny, 1)), P2.T, P3, np.zeros((ny, ny))],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny))]
])

#T1 = cp.Variable((ny, ny), diag=True)
T1 = cp.bmat([[cp.Variable(),0],[0,cp.Variable()]])

S1_1 = cp.bmat([
    [np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny))],
    [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, ny)), np.zeros((n, ny))],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), T1],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny))]
])


R1 = cp.Variable((1 + n + 2 * ny, ny))

M1 = cp.Variable((2 * ny, 2 * ny), symmetric=True)
M1 = cp.bmat([[np.zeros([1,1+2*ny])],
          [np.zeros([2*ny,1]),M1]])

Sa = cp.Variable((1 + n + 2 * ny, 1 + n + 2 * ny), symmetric=True)
Sp = cp.Variable((1 + n + 2 * ny, ny))
Spb = cp.Variable((1 + n + 2 * ny, ny))

MS = cp.bmat([
    [np.zeros((1, 1 + n + 2 * ny))],
    [np.hstack([np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, 2 * ny))])],
    [Sp.T],
    [Spb.T]
])

LMI1 = H1 + (S1_1 + R1 @ XI1 + (S1_1 + R1 @ XI1).T) - (I1.T) @ M1 @ I1 - Sa - (MS + MS.T)
LMI1 = IS.T@LMI1@IS
#%%
F3_2 = np.vstack((F3, F3@F1))
F4_2 = np.block([[F4, np.zeros((ny, ny))], [F3@F2, F4]])
F5_2 = np.vstack((F5, F5))

N1 = F1.T @ P1 @ F1 - (1 - nu) * P1
N2 = cp.bmat([[F1.T @ P1 @ F2 - (1 - nu) * P2, F1.T @ P2]])
N3 = cp.bmat([[F2.T @ P1 @ F2 - (1 - nu) * P3, F2.T @ P2], [P2.T @ F2, P3]])

H2 = cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, 2 * ny)), np.zeros((1, 2 * ny))],
               [np.zeros((1, n)).T, N1, N2, np.zeros((n, 2 * ny))],
               [np.zeros((2 * ny, 1)), N2.T, N3, np.zeros((2 * ny, 2 * ny))],
               [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), np.zeros((2 * ny, 2 * ny))]])


T2 = cp.bmat([[cp.Variable(),0,0,0],[0,cp.Variable(),0,0],[0,0,cp.Variable(),0],[0,0,0,cp.Variable()]])

S1_2 = cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, 2 * ny)), np.zeros((1, 2 * ny))],
                 [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, 2 * ny)), np.zeros((n, 2 * ny))],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), T2],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), np.zeros((2 * ny, 2 * ny))]])

I2 = np.block([[1, np.zeros((1, 2 * ny)), np.zeros((1, 2 * ny))],
               [np.zeros((n, 1)), np.zeros((n, 2 * ny)), np.zeros((n, 2 * ny))],
               [np.zeros((2 * ny, 1)), np.eye(2 * ny), np.zeros((2 * ny, 2 * ny))],
               [np.zeros((2 * ny, 1)), np.zeros((2 * ny, 2 * ny)), np.eye(2 * ny)]]).T

XI2 = np.block([[F5_2, F3_2, F4_2 - np.eye(2 * ny), np.eye(2 * ny)]])

R2 = cp.Variable((1 + n + 4 * ny, 2 * ny))

# Novas Adições

M2 = cp.Variable((4 * ny, 4 * ny), symmetric=True)
M2 = cp.bmat([[np.zeros([1,1+4*ny])],
          [np.zeros([4*ny,1]),M2]])


Sa2 = cp.Variable((1 + n + 4 * ny, 1 + n + 4 * ny), symmetric=True)
Sp2 = cp.Variable((1 + n + 4 * ny, 2 * ny))
Spb2 = cp.Variable((1 + n + 4 * ny, 2 * ny))
MS2 = cp.bmat([[np.zeros((1, 1 + n + 4 * ny))],
                [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, 4 * ny))],
                [Sp2.T],
                [Spb2.T]])

Q_T = cp.bmat([
    [np.zeros((1, 1)),  np.zeros((1, n)), np.zeros((1, 2*ny)), np.zeros((1, 2*ny)), np.zeros((1, 2*ny)), np.zeros((1, 2*ny)),  np.zeros((1, 2*n))],
    [np.zeros((n, 1)),  Qx, np.zeros((n,2*ny)), np.zeros((n,2*ny)), np.zeros((n,2*ny)), np.zeros((n, 2*ny)),  np.zeros((n, 2*n))],
    [np.zeros((2*ny, 1)),  np.zeros((2*ny,n)), 0*Qphi_, np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)),  np.zeros((2*ny, 2*n))],
    [np.zeros((2*ny, 1)),  np.zeros((2*ny,n)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)),  np.zeros((2*ny, 2*n))],
    [np.zeros((2*ny, 1)),  np.zeros((2*ny, n)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)),  np.zeros((2*ny,2*n))],
    [np.zeros((2*ny, 1)),  np.zeros((2*ny,n)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)), np.zeros((2*ny, 2*ny)),  np.zeros((2*ny, 2*n))],
    [np.zeros((2*n, 1)),  np.zeros((2*n, n)), np.zeros((2*n, 2*ny)), np.zeros((2*n, 2*ny)), np.zeros((2*n, 2*ny)), np.zeros((2*n, 2*ny)),  Qdel_] #degradação
])

LMI2 = -H2 + (S1_2 + R2 @ XI2 + (S1_2 + R2 @ XI2).T) - (I2.T) @ M2 @ I2 - Sa2 - (MS2 + MS2.T)
LMI2 = IS2.T@LMI2@IS2#+Q_T
#%% LMI3
alpha3 = cp.Variable()

D = np.block([[rd**2 - xc.T @ np.eye(n) @ xc, xc.T @ np.eye(n), np.zeros((1, ny)), np.zeros((1, ny))],
              [(xc.T @ np.eye(n)).T, -np.eye(n), np.zeros((n, ny)), np.zeros((n, ny))],
              [np.zeros([2*ny, 1+n+2*ny])]])

T3 = cp.bmat([[cp.Variable(),0],[0,cp.Variable()]])

R3 = cp.Variable((1+n+2*ny, ny))

S1_3 = cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny))],
                 [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, ny)), np.zeros((n, ny))],
                 [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), T3],
                 [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny))]])

M3 = cp.Variable((1+2*ny, 1+2*ny), symmetric=True)

LMI3 = Sa - alpha3*D + (S1_3 + R3 @ XI1 + (S1_3 + R3 @ XI1).T) - (I1.T) @ M3 @ I1
LMI3 = IS.T@LMI3@IS
#%% LMI4
alpha4 = []
R4 = []
T4 = []
S1_4 = []
M4 = []
LMIs4 = []
Sp4 = []
LMI4 = []
for i in range(ny):

    alpha4.append(cp.Variable())


    R4.append(cp.Variable((1+n+2*ny, ny)))

    T4.append(cp.bmat([[cp.Variable(),0],[0,cp.Variable()]]))

    S1_4.append(cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1,ny)), np.zeros((1,ny))],
                [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n,ny)), np.zeros((n,ny))],
                [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny,ny)), T4[i]],
                [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny,ny)), np.zeros((ny,ny))]]))

    M4.append(cp.Variable((1+2*ny, 1+2*ny), symmetric=True))
    #OLHA QUE TRETA, O PYTHON SO PEGA INTERVALO, ENTÃO PARA PEGAR UMA POSIÇÃO É PRECISO COLOCAR UM INTERVALO QUE VAI DA POSIÇÃO QUE VC QUER ATÉ A PRÓXIMA.
    Sp4.append(cp.bmat([[Sp[:,i:i+1].T],
                        [np.zeros([n,1+n+2*ny])],
                        [np.zeros([ny,1+n+2*ny])],
                        [np.zeros([n,1+n+2*ny])]]))
    LMI4.append(IS.T@(0.5 * (Sp4[i] + Sp4[i].T) - alpha4[i] * D + (S1_4[i] + R4[i] @ XI1 + (S1_4[i] + R4[i] @ XI1).T) - cp.transpose(I1) @ M4[i] @ I1)@IS)
    
#%% LMI5
alpha5 = []
R5 = []
T5 = []
S1_5 = []
M5 = []
LMIs5 = []
Sp5 = []
LMI5 = []
for i in range(ny):

    alpha5.append(cp.Variable())


    R5.append(cp.Variable((1+n+2*ny, ny)))

    T5.append(cp.bmat([[cp.Variable(),0],[0,cp.Variable()]]))

    S1_5.append(cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1,ny)), np.zeros((1,ny))],
                [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n,ny)), np.zeros((n,ny))],
                [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny,ny)), T5[i]],
                [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny,ny)), np.zeros((ny,ny))]]))

    M5.append(cp.Variable((1+2*ny, 1+2*ny), symmetric=True))
    #OLHA QUE TRETA, O PYTHON SO PEGA INTERVALO, ENTÃO PARA PEGAR UMA POSIÇÃO É PRECISO COLOCAR UM INTERVALO QUE VAI DA POSIÇÃO QUE VC QUER ATÉ A PRÓXIMA.
    Sp5.append(cp.bmat([[Spb[:,i:i+1].T],
                        [np.zeros([n,1+n+2*ny])],
                        [np.zeros([ny,1+n+2*ny])],
                        [np.zeros([n,1+n+2*ny])]]))
    
    LMI5.append(IS.T@(0.5 * (Sp5[i] + Sp5[i].T) - alpha5[i] * D + (S1_5[i] + R5[i] @ XI1 + (S1_5[i] + R5[i] @ XI1).T) - cp.transpose(I1) @ M5[i] @ I1)@IS)

#%% LMI6
alpha6 = cp.Variable()

Ib = np.block([
    [np.array([1]), np.zeros((1, n)), np.zeros((1, 4 * ny))],
    [np.zeros((n, 1)), np.eye(n), np.zeros((n, 4 * ny))],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.eye(ny), np.zeros((ny, ny)), np.zeros((ny, 2 * ny))],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, 2 * ny)), np.eye(ny), np.zeros((ny, ny))]
])

Db = Ib.T@D@Ib

T6 = cp.bmat([[cp.Variable(),0,0,0],[0,cp.Variable(),0,0],[0,0,cp.Variable(),0],[0,0,0,cp.Variable()]])


S1_6 = cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, 2 * ny)), np.zeros((1, 2 * ny))],
                 [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, 2 * ny)), np.zeros((n, 2 * ny))],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), T6],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), np.zeros((2 * ny, 2 * ny))]])

R6 = cp.Variable((1 + n + 4 * ny, 2 * ny))

M6 = cp.Variable((1+4*ny, 1+4*ny), symmetric=True)

LMI6 = Sa2 - alpha6*Db + (S1_6 + R6 @ XI2 + (S1_6 + R6 @ XI2).T) - (I2.T) @ M6 @ I2
LMI6 = IS2.T@LMI6@IS2
#%% LMI7
alpha7 = []
R7 = []
T7 = []
S1_7 = []
M7 = []
LMIs7 = []
Sp7 = []
LMI7 = []
for i in range(2*ny):

    alpha7.append(cp.Variable())


    R7.append(cp.Variable((1+n+4*ny, 2*ny)))
    T7.append(cp.bmat([[cp.Variable(),0,0,0],[0,cp.Variable(),0,0],[0,0,cp.Variable(),0],[0,0,0,cp.Variable()]]))

    S1_7.append(cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, 2 * ny)), np.zeros((1, 2 * ny))],
                 [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, 2 * ny)), np.zeros((n, 2 * ny))],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), T7[i]],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), np.zeros((2 * ny, 2 * ny))]]))

    M7.append(cp.Variable((1+4*ny, 1+4*ny), symmetric=True))
    #OLHA QUE TRETA, O PYTHON SO PEGA INTERVALO, ENTÃO PARA PEGAR UMA POSIÇÃO É PRECISO COLOCAR UM INTERVALO QUE VAI DA POSIÇÃO QUE VC QUER ATÉ A PRÓXIMA.
    Sp7.append(cp.bmat([[Sp2[:,i:i+1].T],
                        [np.zeros([n,1+n+4*ny])],
                        [np.zeros([2*ny,1+n+4*ny])],
                        [np.zeros([2*ny,1+n+4*ny])]]))
    LMI7.append(IS2.T@(0.5 * (Sp7[i] + Sp7[i].T) - alpha7[i] * Db + (S1_7[i] + R7[i] @ XI2 + (S1_7[i] + R7[i] @ XI2).T) - cp.transpose(I2) @ M7[i] @ I2)@IS2)

#%% LMI8
alpha8 = []
R8 = []
T8 = []
S1_8 = []
M8 = []
LMIs8 = []
Sp8 = []
LMI8 = []
for i in range(2*ny):

    alpha8.append(cp.Variable())


    R8.append(cp.Variable((1+n+4*ny, 2*ny)))

    T8.append(cp.bmat([[cp.Variable(),0,0,0],[0,cp.Variable(),0,0],[0,0,cp.Variable(),0],[0,0,0,cp.Variable()]]))

    S1_8.append(cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, 2 * ny)), np.zeros((1, 2 * ny))],
                 [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, 2 * ny)), np.zeros((n, 2 * ny))],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), T8[i]],
                 [np.zeros((2 * ny, 1)), np.zeros((2 * ny, n)), np.zeros((2 * ny, 2 * ny)), np.zeros((2 * ny, 2 * ny))]]))

    M8.append(cp.Variable((1+4*ny, 1+4*ny), symmetric=True))
    #OLHA QUE TRETA, O PYTHON SO PEGA INTERVALO, ENTÃO PARA PEGAR UMA POSIÇÃO É PRECISO COLOCAR UM INTERVALO QUE VAI DA POSIÇÃO QUE VC QUER ATÉ A PRÓXIMA.
    Sp8.append(cp.bmat([[Spb2[:,i:i+1].T],
                        [np.zeros([n,1+n+4*ny])],
                        [np.zeros([2*ny,1+n+4*ny])],
                        [np.zeros([2*ny,1+n+4*ny])]]))
    LMI8.append(IS2.T@(0.5 * (Sp8[i] + Sp8[i].T) - alpha8[i] * Db + (S1_8[i] + R8[i] @ XI2 + (S1_8[i] + R8[i] @ XI2).T) - cp.transpose(I2) @ M8[i] @ I2)@IS2)

#%% LMI3
alphaL = cp.Variable()


TL =  cp.diag(cp.Variable(ny))

RL = cp.Variable((1+n+2*ny, ny))

S1_L = cp.bmat([[np.zeros((1, 1)), np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny))],
                 [np.zeros((n, 1)), np.zeros((n, n)), np.zeros((n, ny)), np.zeros((n, ny))],
                 [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), TL],
                 [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny))]])

ML = cp.Variable((1+2*ny, 1+2*ny), symmetric=True)


HL = cp.bmat([
    [np.ones((1, 1)), np.zeros((1, n)), np.zeros((1, ny)), np.zeros((1, ny))],
    [np.zeros((n, 1)), -P1 , -P2, np.zeros((n, ny))],
    [np.zeros((ny, 1)), -P2.T, -P3, np.zeros((ny, ny))],
    [np.zeros((ny, 1)), np.zeros((ny, n)), np.zeros((ny, ny)), np.zeros((ny, ny))]
])

LMIL =  alphaL*D - HL + (S1_L + RL @ XI1 + (S1_L + RL @ XI1).T) - (I1.T) @ ML @ I1
LMIL = IS.T@LMIL@IS
#%%
v = []
v.append(np.array([[-1],[0]]))
v.append(np.array([[0],[-1]]))
v.append(np.array([[1],[0]]))
v.append(np.array([[0],[1]]))

lambda_vals = []
lambda_vals.append(20)
lambda_vals.append(1)
lambda_vals.append(1)
lambda_vals.append(1)


# Cálculo de L1, L2, L3 e L4
y = F3[0:2] @ (lambda_vals[0] * v[0]) + F5[0:2]
L1 = np.block([[lambda_vals[0]*v[0]],[np.maximum(y, 0)]]).T@cp.bmat([[P1, P2], [P2.T, P3]])@ np.block([[lambda_vals[0]*v[0]],[np.maximum(y, 0)]])


y = F3[0:2] @ (lambda_vals[1] * v[1]) + F5[0:2]
L2 = np.block([[lambda_vals[1]*v[1]],[np.maximum(y, 0)]]).T@cp.bmat([[P1, P2], [P2.T, P3]])@ np.block([[lambda_vals[1]*v[1]],[np.maximum(y, 0)]])

y = F3[0:2] @ (lambda_vals[2] * v[2]) + F5[0:2]
L3 = np.block([[lambda_vals[2]*v[2]],[np.maximum(y, 0)]]).T@cp.bmat([[P1, P2], [P2.T, P3]])@ np.block([[lambda_vals[2]*v[2]],[np.maximum(y, 0)]])

y = F3[0:2] @ (lambda_vals[3] * v[3] + F5[0:2])
L4 = np.block([[lambda_vals[3]*v[3]],[np.maximum(y, 0)]]).T@cp.bmat([[P1, P2], [P2.T, P3]])@ np.block([[lambda_vals[3]*v[3]],[np.maximum(y, 0)]])





#%%
# Continuar com a definição das LMIs restantes conforme o código original
# Substituir sdpvar por cp.Variable, definir as LMI conforme o exemplo acima

# Objetivo
Folga = 0
obj = cp.trace(P1)

constraints = []
# Restrições
#constraints.append(P1 >> E)

constraints.append(LMIETM >> Folga)
constraints.append(Qdel >> 0)

constraints.append(L1 <= 1)
constraints.append(L2 <= 1)
constraints.append(L3 <= 1)
constraints.append(L4 <= 1)

constraints.append(M1 >= 0)
constraints.append(M2 >= 0)
constraints.append(M3 >= 0)
for i in range(ny):
    constraints.append(M4[i]>>0)
for i in range(ny):
    constraints.append(M5[i]>>0)
constraints.append(M6 >= 0)
for i in range(2*ny):
    constraints.append(M7[i]>>0)
for i in range(2*ny):
    constraints.append(M8[i]>>0)
constraints.append(ML >= 0)

constraints.append(LMI1 >> Folga)
constraints.append(LMI2 >> Folga)
constraints.append(LMI3 >> Folga)
for i in range(ny):
    constraints.append(LMI4[i]>>Folga)
for i in range(ny):
    constraints.append(LMI5[i]>>Folga)
constraints.append(LMI6 >> Folga)
for i in range(2*ny):
    constraints.append(LMI7[i]>>Folga)
for i in range(2*ny):
    constraints.append(LMI8[i]>>Folga)
constraints.append(LMIL >> Folga)

constraints.append(alpha3 >= 0)
constraints.append(alpha6 >= 0)
constraints.append(alphaL >= 0)
for i in range(ny):
    constraints.append(alpha4[i]>=0)
for i in range(ny):
    constraints.append(alpha5[i]>=0)
for i in range(2*ny):
    constraints.append(alpha7[i]>=0)
for i in range(2*ny):
    constraints.append(alpha8[i]>=0)

# Otimização
prob = cp.Problem(cp.Minimize(obj), constraints)
prob.solve(verbose = True,solver=cp.MOSEK)

P = np.block([[P1.value,P2.value],[(P2.T).value,P3.value]])
# Verificar o resultado
a = prob.status
#res = min([constraint.violation() for constraint in constraints])
#print("Status:", a)
#print("Violation:", res)
#print("nu:", nu)
#print("P:", P1.value, P2.value, P3.value)


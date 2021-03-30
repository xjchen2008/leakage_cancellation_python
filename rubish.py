import numpy as np
def solution(A):
    A1 = []
    for i in A:
        if i not in A1:
            A1.append(i)

    A1.sort()
    print(A1)
    if max(A1) <=0:
        return 1
    else:
        for i in range(len(A1)):
            print(i)
            if i+1 == A1[i]:
                pass
            else:
                return i+1
        return i+2


A = [-1, 0 , 1,2,3,4, 6,7,8]
#A = [np.linspace(0,100),np.linspace(102,200),]
sol = solution(A)
print('min positive int is ', sol)









from matplotlib import pyplot as plt
import numpy as np
'''L = 1
x = np.linspace(0,L,1000)
N = 1

psi = N*x*(L-x)
phi = np.sqrt(2/L)*np.sin(np.pi/L*x)
plt.plot(x, psi)
plt.plot(x, phi)
plt.show()'''
'''# Qestuion 4
h = 6.626e-34
m = 9.109e-31
n = 2578422
A = 1.0e-10
L = np.array([10.0*A,100.0*A,1000.0*A,10000.0*A,1e8*A])
E = h**2 * n**2 / (8 * m * L**2)  / 1.602176634e-19*1e3
print(E)


# Question 3
h = 6.626e-34
m = 9.109e-31
KBT = 0.0259* 1.602176634e-19
L = np.sqrt(3* h**2/ (2* 0.1*m * KBT))
print(L)'''

#HW6
'''
h = 6.626e-34
hbar = h / 2 / np.pi
m = 9.109e-31
eV2Joules = 1.602176634e-19
V0 = 10e-3 /eV2Joules
E = np.array([0.5e-3/eV2Joules, 5e-3/eV2Joules, 9.5e-3/eV2Joules])
xp = -1/2/np.sqrt(2*m*(V0 - E) / hbar**2 ) * np.log(0.5)
print xp'''
'''
h = 6.626e-34
hbar = h / 2 / np.pi
m = 9.109e-31
eV2Joules = 1.602176634e-19
V0 = 10e-3 /eV2Joules
E = 0.025

kapa = np.sqrt(2*m*(V0-E)/hbar**2)
V0 = 0.25/eV2Joules
a = np.linspace(0,10)
T = (1+1/4*V0/(E*(V0-E)))*np.sinh(2*kapa*a)
'''
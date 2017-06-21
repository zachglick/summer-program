import numpy as np
import configparser
import sys
import psi4
import scipy.linalg

config = configparser.ConfigParser()
config.read('Options.ini')
#print('!'+config['DEFAULT']['basis']+'!')

molecule = psi4.geometry('\n0 1\nO\nH 1 R\nH 1 R 2 A\nR = 1.0\nA = 104.5')
molecule.update_geometry()

basis = psi4.core.BasisSet.build(molecule, "BASIS", "STO-3G",puream=0)
mints = psi4.core.MintsHelper(basis)

#Here the to array function is converting psi4 arrays to np arrays

# Overlap integrals
S = mints.ao_overlap().to_array()
#Kinetic energy portion
T = mints.ao_kinetic().to_array()
# Potential energy portion
V = mints.ao_potential().to_array()
#Two-electron repulsion
I = mints.ao_eri().to_array()

Enuc   = molecule.nuclear_repulsion_energy()
natom  = molecule.natom()
charge = molecule.molecular_charge() # 0 if neutral, -1 if anion, +1 if cation, etc.
norb   = mints.basisset().nbf() # number of (spatial) orbitals = number of basis functions

nuclear_charges = [molecule.Z(A) for A in range(natom)] # nuclear charges by atom

#print('S (overlap) shape '+ str(S.shape))
#print('T (kinetic) shape '+ str(T.shape))
#print('V (potential) shape '+ str(V.shape))
#print('I (two elec rep) shape '+ str(I.shape))
#print("Enuc   = {:<.7f}".format(Enuc))
#print("natom  = {:d}".format(natom))
#print("charge = {:d}".format(charge))
#print("norb   = {:d}".format(norb))
#print("\nnuclear_charges = {:s}".format(str(nuclear_charges)))


#Don't Change
X = scipy.linalg.fractional_matrix_power(S,-0.5)
H = T + V

#Do Change
D = np.zeros((norb,norb))

Prev_Energy = 10.0e18
Current_Energy = 10.0e9
eps = 10e-3

while abs(Prev_Energy - Current_Energy > eps) :
#    print('|%d - %f| < %f' % (Prev_Energy, Current_Energy, eps))

    U = np.zeros((norb,norb))
    for u in range(norb):
        for v in range(norb):
            sum = 0.0
            for p in range(norb):
                for s in range(norb):
                    sum += I[u,v,p,s]*D[p,s]
            U[u,v] = sum
    F = H + U
    F_orth = X*F*X

    energies, C_orth = np.linalg.eig(F_orth)
    C = X*C_orth

    for u in range(norb):
        for v in range(norb):
            D[u,v] = 2*np.sum(C[u]*C[v])


    Prev_Energy = Current_Energy
    Current_Energy = np.sum(np.multiply(H+0.5*U, np.transpose(D)))
    print(Current_Energy)


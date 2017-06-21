import numpy as np
import configparser
import sys
import psi4
import scipy.linalg

# Use configparser library to read geometry, basis set from 'Config.ini'
config = configparser.ConfigParser()
config.read('Config.ini')
psi4.core.be_quiet()
molecule = psi4.geometry(config['DEFAULT']['molecule'])
molecule.update_geometry()
basis = psi4.core.BasisSet.build(molecule, "BASIS", config['DEFAULT']['basis'] ,puream=0)
if config['DEFAULT']['nalpha'] != config['DEFAULT']['nbeta'] :
    raise ValueError('Expected equal alpha and beta electrons')

# Get Molecular Integrals from PSi4 (Overlap, Kinetic, Potential, and Two-Electron)
mints = psi4.core.MintsHelper(basis)
S = mints.ao_overlap().to_array()
T = mints.ao_kinetic().to_array()
V = mints.ao_potential().to_array()
I = mints.ao_eri().to_array()

Enuc   = molecule.nuclear_repulsion_energy()
#print(Enuc)
natom  = molecule.natom()
charge = molecule.molecular_charge()
norb   = mints.basisset().nbf() # number of (spatial) orbitals = number of basis functions
nuclear_charges = [molecule.Z(A) for A in range(natom)] # nuclear charges by atom

# Iterate until energy converges
X = scipy.linalg.fractional_matrix_power(S,-0.5)
H = T + V
D = np.zeros((norb,norb))
Prev_Energy = 10.0e18
Current_Energy = 10.0e9
eps = 10e-5
U = np.zeros((norb,norb))

while abs(Prev_Energy - Current_Energy) > eps :
#    print('|%f - %f| > %f' % (Prev_Energy, Current_Energy, eps))
#    for u in range(norb):
#        for v in range(norb):
#            sum = 0.0
#            for p in range(norb):
#                for s in range(norb):
#                    sum += (I[u,p,v,s]-0.5*I[u,p,s,v])*D[s,p]
#            U[u,v] = sum
    U = np.einsum('uvps, sp -> uv', I, D) - 0.5*np.einsum('upvs, sp -> uv', I, D)
    print(U)
    
    F = H + U
    Prev_Energy = Current_Energy
    #    Current_Energy = np.sum((H+0.5*U) * np.transpose(D))
    E = H + 0.5*U
    Current_Energy = np.einsum('uv, vu ->',E,D)
    F_orth =(X.dot(F)).dot(X)
    energies, C_orth = np.linalg.eigh(F_orth)
    C = X.dot(C_orth)
    C = C[:,:5]
    D = 2*C.dot(C.T)
#    print(D.shape)
#    D = np.zeros((norb,norb))
#    for u in range(norb):
#        for v in range(norb):
#            for i in range(5):
#                D[u,v] += (2*C[u,i]*C[v,i])
#            D[u,v] = 2*np.sum(C[u]*C[v])
#    D = 2*np.einsum('ui,vi->vu',C,C)
#    U = np.einsum('upvs, sp -> uv', I, D) - 0.5*np.einsum('upsv, sp -> uv', I, D)



    print(Current_Energy + Enuc)


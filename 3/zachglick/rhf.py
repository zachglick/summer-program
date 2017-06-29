import numpy as np
import configparser
import sys
import psi4
import scipy.linalg
import math

# Use configparser module to read geometry from 'Config.ini'
config = configparser.ConfigParser()
config.read('Config.ini')
psi4.core.be_quiet()
molecule = psi4.geometry(config['DEFAULT']['molecule'])
molecule.update_geometry()

# Optional specifications: basis set, maximum number of iterations, and convergence criteria
basis_name = config['DEFAULT']['basis'] if config.has_option('DEFAULT','basis') else 'STO-3G'
basis = psi4.core.BasisSet.build(molecule, 'BASIS', basis_name ,puream=0)
max_iter = int(config['SCF']['max_iter']) if config.has_option('SCF','max_iter') else math.inf
energy_conv = float(config['SCF']['energy_conv']) if config.has_option('SCF','energy_conv') else 10e-11

# RHF requires fully occupied orbitals (multiplicity of 1)
if molecule.multiplicity() != 1 :
    raise ValueError('Multiplicity of 1 required for Restricted Hartree-Fock')

# Get Molecular Integrals from PSi4 (Overlap, Kinetic, Potential, and Two-Electron)
mints = psi4.core.MintsHelper(basis)
S = mints.ao_overlap().to_array()
T = mints.ao_kinetic().to_array()
V = mints.ao_potential().to_array()
I = mints.ao_eri().to_array() # Note: array is in chemist's notation (?)

nuclear_energy = molecule.nuclear_repulsion_energy() # Constant (function of geometry)
natom = molecule.natom()
net_charge = molecule.molecular_charge()
norb = mints.basisset().nbf() # number of (spatial) orbitals = number of basis functions
nuclear_charges = [int(molecule.Z(A)) for A in range(natom)] # nuclear charges by atom
nocc = int((sum(nuclear_charges) - net_charge)/2) # the number of occupied (non-virtual) orbitals

X = scipy.linalg.fractional_matrix_power(S,-0.5)    # Orthagonalization matrix
D = np.zeros((norb,norb))                           # Density matrix (initial guess 0)
H = T + V                                           # One electron matrix
U = np.zeros((norb,norb))                           # Two Electron matrix
F = U + H                                           # Fock Matrix = U + H

prev_energy = math.inf
current_energy = 10.0e9
iteration_num = 0

# Iterate until energy converges or maximum number of iterations reached
while (abs(prev_energy - current_energy) > energy_conv) and (iteration_num < max_iter) :
    # Calculating U with the two electron integrals and density matrix
    for u in range(norb):
        for v in range(norb):
            sum = 0.0
            for p in range(norb):
                for s in range(norb):
                    sum += (I[u,v,p,s]-0.5*I[u,s,p,v])*D[s,p]
            U[u,v] = sum
    # Alternatively: U = np.einsum('uvps, sp -> uv', I, D) - 0.5*np.einsum('upsv, sp -> uv', I, D)

    # Updating the energy
    prev_energy = current_energy
    current_energy = np.sum((H+0.5*U) * D.T) # Alternatively: = np.einsum('uv, vu ->',H + ).5*U,D)

    # Updating the fock matrix to get the basis coeffiecients (C) which gives the densities (D)
    F = H + U
    F_orth = X.dot(F).dot(X)
    energies, C_orth = np.linalg.eigh(F_orth)
    C = X.dot(C_orth)
    C = C[:,:nocc] # Only use coefficients of occupied MOs to compute density matrix
    D = 2*C.dot(C.T)

    iteration_num += 1
    print("After %d iteration(s) the energy is %.12f Hartree" % (iteration_num, current_energy+nuclear_energy))

print('Final energy: %.12f Hartrees' % (current_energy+nuclear_energy))
    





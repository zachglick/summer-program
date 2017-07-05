import numpy as np
import configparser
import sys
import psi4
import scipy.linalg
import math

class RHF:
    def __init__(self, config_file, printResults=True):
        # Use configparser module to read geometry from 'Config.ini'
        config = configparser.ConfigParser()
        config.read(config_file)
        psi4.core.be_quiet()
        self.molecule = psi4.geometry(config['DEFAULT']['molecule'])
        self.molecule.update_geometry()

        # Optional specifications: basis set, maximum number of iterations, and convergence criteria
        basis_name = config['DEFAULT']['basis'] if config.has_option('DEFAULT','basis') else 'STO-3G'
        basis = psi4.core.BasisSet.build(self.molecule, 'BASIS', basis_name ,puream=0)
        max_iter = int(config['SCF']['max_iter']) if config.has_option('SCF','max_iter') else math.inf
        energy_conv = float(config['SCF']['energy_conv']) if config.has_option('SCF','energy_conv') else 10e-11
    
        # RHF requires fully occupied orbitals (multiplicity of 1)
        if self.molecule.multiplicity() != 1 :
            raise ValueError('Multiplicity of 1 required for Restricted Hartree-Fock')


        # Get Molecular Integrals from PSi4 (Kinetic, Potential, Overlap, and Two-Electron)
        mints = psi4.core.MintsHelper(basis)
        T = mints.ao_kinetic().to_array()
        V = mints.ao_potential().to_array()
        self.overlap = mints.ao_overlap().to_array()
        self.twoelec = mints.ao_eri().to_array() # Note: array is in chemist's notation (?)

        self.nuclear_energy = self.molecule.nuclear_repulsion_energy() # Constant (function of geometry)
        self.natom = self.molecule.natom()
        self.norb = mints.basisset().nbf() # number of (spatial) orbitals = number of basis functions

        net_charge = self.molecule.molecular_charge()
        nuclear_charges = [int(self.molecule.Z(A)) for A in range(self.natom)] # nuclear charges by atom
        self.nocc = int((np.sum(nuclear_charges) - net_charge)/2) # the number of occupied (non-virtual) spatial orbitals

        self.X = scipy.linalg.fractional_matrix_power(self.overlap,-0.5)         # Orthagonalization matrix
        self.D = np.zeros((self.norb,self.norb))                           # Density matrix (initial guess 0)
        self.H = T + V                                                     # One electron matrix
        self.U = np.zeros((self.norb,self.norb))                           # Two Electron matrix
        self.F = self.U + self.H                                           # Fock Matrix = U + H

        prev_energy = math.inf
        current_energy = 10.0e9
        iteration_num = 0

        # Iterate until energy converges or maximum number of iterations reached
        while (abs(prev_energy - current_energy) > energy_conv) and (iteration_num < max_iter) :
            # Calculating U with the two electron integrals and density matrix
            for u in range(self.norb):
                for v in range(self.norb):
                    sum = 0.0
                    for p in range(self.norb):
                        for s in range(self.norb):
                            sum += (self.twoelec[u,v,p,s]-0.5*self.twoelec[u,s,p,v])*self.D[s,p]
                    self.U[u,v] = sum
            # Alternatively: U = np.einsum('uvps, sp -> uv', I, D) - 0.5*np.einsum('upsv, sp -> uv', I, D)

            # Updating the energy
            prev_energy = current_energy
            current_energy = np.sum((self.H+0.5*self.U) * (self.D).T) # Alternatively: = np.einsum('uv, vu ->',H + ).5*U,D)

            # Updating the fock matrix to get the basis coeffiecients (C) which gives the densities (D)
            self.F = self.H + self.U
            F_orth = (self.X).dot(self.F).dot(self.X)
            self.energies, C_orth = np.linalg.eigh(F_orth)
            self.C = (self.X).dot(C_orth)
            C_occ = self.C[:,:self.nocc] # Only use coefficients of occupied MOs to compute density matrix
            self.D = 2*C_occ.dot(C_occ.T)

            iteration_num += 1
            if printResults:
                print("After %d iteration(s) the energy is %.12f Hartree" % (iteration_num, current_energy+self.nuclear_energy))

        self.energy_SCF = current_energy
        
        if printResults:
            print('Final energy: %.12f Hartrees' % (current_energy+self.nuclear_energy))



if __name__ == '__main__':
    # Default Config File name is 'Config.ini'
    config_file = 'Config.ini' if len(sys.argv) == 1 else sys.argv[1]
    myobj = RHF(config_file)


    





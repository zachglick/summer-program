import sys
sys.path.append('../../../0/zachglick')
sys.path.append('../../../extra-files')
import configparser
import math
import numpy as np
from scipy import special, misc
from molecule import Molecule
import itertools

# Hardcoded values for STO-3G

NGAUSS = 3 # Every Slater Type Orbital (STO) is approxd w 3 Gaussians (3G) in STO-3G
SUBSHELLS = { 'H'  : ['1s'], # The orbitals used for each atom (first two rows included)
              'He' : ['1s'],
              'Li' : ['1s','2s','2px','2py','2pz'],
              'Be' : ['1s','2s','2px','2py','2pz'],
              'B'  : ['1s','2s','2px','2py','2pz'],
              'C'  : ['1s','2s','2px','2py','2pz'],
              'N'  : ['1s','2s','2px','2py','2pz'],
              'O'  : ['1s','2s','2px','2py','2pz'],
              'F'  : ['1s','2s','2px','2py','2pz'],
              'Ne' : ['1s','2s','2px','2py','2pz'] }
ANG_COORDS = ['l', 'm', 'n']

class Integrals:
    """ For a given basis function and molecule, calculates useful integrals for
        Hartree Fock calculations (Overlap, Kinetic, Potential, Two Electron) """
    
    #
    # Initialization / Input
    #
    
    def __init__(self, basis_file, molecule):
        """ Initializes the integrals object """
        self.config = configparser.ConfigParser()
        self.config.read(basis_file)
        self.molecule = molecule
        
        self.sto3g = [] # list of dictionaries, each of which describes an atomic orbital
        self.norb = 0 # the number of atomic orbitals in this molecule / basis set
        for atom_name, atom_coords in molecule:
            for subshell in SUBSHELLS[atom_name]:
                self.norb += 1
                ao_info = {
                    'a' : self.read_alpha(atom_name, subshell),         # list of alpha exponents (NGAUSS)
                    'd' : self.read_contraction(atom_name, subshell),   # list of contraction coeffiecients (NGAUSS)
                    'R' : atom_coords,                                  # list of atom's cartesian coordinates (3)
                    'l' : 1 if 'px' in subshell else 0,                 # Angular momentum?
                    'm' : 1 if 'py' in subshell else 0,                 # Angular momentum?
                    'n' : 1 if 'pz' in subshell else 0,                 # Angular momentum?
                }
                self.sto3g.append(ao_info)
                    
        self.orbital_pairs = list(itertools.product(range(self.norb),range(self.norb)))     # for easier iteration over orbitals
        self.gaussian_pairs = list(itertools.product(range(NGAUSS),range(NGAUSS)))          # for easier iteration over gaussians
  
        self.calc_overlap()
        self.calc_kinetic_energy()
        self.calc_nuclear_attraction()
        self.calc_electron_repulsion()
        
#        print('\nMy Overlap:')
#        for row in self.overlap:
#            print(' '.join(['%.10f'%(num) for num in row]))
#        print('\nPsi4 Overlap:')
#        psi4overlap = np.load('psi4_test/psi4_overlap.npy')
#        for row in psi4overlap:
#            print(' '.join(['%.10f'%(num) for num in row]))
#        print('')
#
#        print('\nMy Kinetic:')
#        for row in self.kinetic:
#            print(' '.join(['%.10f'%(num) for num in row]))
#        print('\nPsi4 Kinetic:')
#        psi4kinetic = np.load('psi4_test/psi4_kinetic.npy')
#        for row in psi4kinetic:
#            print(' '.join(['%.10f'%(num) for num in row]))
#        print('')
#
#        print('\nMy Nuclear:')
#        for row in self.nuclear:
#            print(' '.join(['%.10f'%(num) for num in row]))
#        print('\nPsi4 Nuclear:')
#        psi4nuclear = np.load('psi4_test/psi4_potential.npy')
#        for row in psi4nuclear:
#            print(' '.join(['%.10f'%(num) for num in row]))
#        print('')

    def read_alpha(self, atom_name, subshell):
        """ returns tuple of alpha values for that atom/subshell"""
        if 'p' in subshell: # look for p instead of px vs py vs pz
            subshell = subshell[:-1]
        gaussians = self.config[atom_name][subshell].split()
        return (float(gaussians[0]), float(gaussians[2]), float(gaussians[4]))

    def read_contraction(self, atom_name, subshell):
        """ returns tuple of contraction coefficients for that atom/subshell """
        if 'p' in subshell: # look for p instead of px vs py vs pz
            subshell = subshell[:-1]
        gaussians = self.config[atom_name][subshell].split()
        return (float(gaussians[1]), float(gaussians[3]), float(gaussians[5]))

    #
    # Integral Calculations
    #
    
    def calc_overlap(self):
        """ calculates overlap integrals """
        self.overlap = np.zeros((self.norb,self.norb))
        for orbA_i, orbB_i in self.orbital_pairs:
            temp_ans = 0.0
            for gaussA_i, gaussB_i in self.gaussian_pairs:
                self.overlap[ orbA_i, orbB_i ] += self.normalized_gaussian_overlap(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i)

    def normalized_gaussian_overlap(self, orbA, orbB, gaussA_i, gaussB_i):
        """ this func calculates the contribution of these two gaussians to the total overlap """
        return ( orbA['d'][gaussA_i]
                 * orbB['d'][gaussB_i]
                 * self.normalization(orbA, gaussA_i)
                 * self.normalization(orbB, gaussB_i)
                 * self.gaussian_overlap(orbA, orbB, gaussA_i, gaussB_i) )

    def gaussian_overlap(self, orbA, orbB, gaussA_i, gaussB_i):
        """ this func calculates the contribution of these two gaussians to the total overlap """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        gamma = alphaA + alphaB
        return ( math.exp ( -alphaA
                            * alphaB
                            * self.AB2(orbA['R'], orbB['R']) * 1.8897161646320724 * 1.8897161646320724
                            / gamma )
                * self.s_func(orbA, orbB, gaussA_i, gaussB_i, 0)   # Sx
                * self.s_func(orbA, orbB, gaussA_i, gaussB_i, 1)   # Sy
                * self.s_func(orbA, orbB, gaussA_i, gaussB_i, 2) ) # Sz

    def calc_kinetic_energy(self):
        """ calc kinetic energy """
        self.kinetic = np.zeros((self.norb,self.norb))
        for orbA_i, orbB_i in self.orbital_pairs:
            for gaussA_i, gaussB_i in self.gaussian_pairs:
                self.kinetic[ orbA_i, orbB_i ] += self.normalized_gaussian_kinetic(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i)

    def normalized_gaussian_kinetic(self, orbA, orbB, gaussA_i, gaussB_i):
        """ this func calculates the contribution of these two gaussians to the total kinetic """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        gamma = alphaA + alphaB
        lb, mb, nb = orbB['l'], orbB['m'], orbB['n']
        
        return ( orbA['d'][gaussA_i]
                 * orbB['d'][gaussB_i]
                 * self.normalization(orbA, gaussA_i)
                 * self.normalization(orbB, gaussB_i)
                 * ( self.gaussian_overlap(orbA, orbB, gaussA_i, gaussB_i) * alphaB*(2*(lb+mb+nb)+3)
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'l', 2), gaussA_i, gaussB_i) * alphaB * alphaB * -2
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'm', 2), gaussA_i, gaussB_i) * alphaB * alphaB * -2
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'n', 2), gaussA_i, gaussB_i) * alphaB * alphaB * -2
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'l', -2), gaussA_i, gaussB_i) * -0.5 * lb * (lb-1)
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'm', -2), gaussA_i, gaussB_i) * -0.5 * mb * (mb -1)
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'n', -2), gaussA_i, gaussB_i) * +0.5 * nb * (nb-1) ) ) # should this be + or - ?

    def calc_nuclear_attraction(self):
        """ calc nuclear attraction """
        self.nuclear = np.zeros((self.norb,self.norb))
        for orbA_i, orbB_i in self.orbital_pairs:
            for gaussA_i, gaussB_i in self.gaussian_pairs:
                self.nuclear[ orbA_i, orbB_i ] += self.normalized_gaussian_nuclear(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i)



    def normalized_gaussian_nuclear(self, orbA, orbB, gaussA_i, gaussB_i):
        """ this func calculates the contribution of these two gaussians to the total kinetic """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        gamma = alphaA + alphaB
        lb, mb, nb = orbB['l'], orbB['m'], orbB['n']
        
        return ( orbA['d'][gaussA_i]
                * orbB['d'][gaussB_i]
                * self.normalization(orbA, gaussA_i)
                * self.normalization(orbB, gaussB_i)
                * -1.0 # charge
                * ( 2 * math.pi / gamma )
                * math.exp ( -alphaA
                             * alphaB
                             * self.AB2(orbA['R'], orbB['R']) * 1.8897161646320724 * 1.8897161646320724
                             / gamma )
                * 1.0 )

    def calc_electron_repulsion(self):
        """ calc electron repulsion """
        self.twoelec = np.zeros((self.norb,self.norb,self.norb,self.norb))

    def normalization(self, orb, gauss_i):
        """ The normalization constant for a gaussian """
        l, m, n = orb['l'], orb['m'], orb['n']
        a = orb['a'][gauss_i]
        
#        return math.sqrt( (2 ** (1*(l+m+n) )) # ???
#                          * ( (2*a) ** (l+m+n) )
#                          * ( (2*a/math.pi) ** 1.5 )
#                          / ( misc.factorial2( 2*l - 1 , exact=True )
#                            * misc.factorial2( 2*m - 1 , exact=True )
#                            * misc.factorial2( 2*n - 1 , exact=True ) ) )

        return math.sqrt( ((8*a)**(l+m+n)) # ???
                         * ( math.factorial(l) * math.factorial(m) * math.factorial(n) )
                         * ( (2*a/math.pi) ** 1.5 )
                         / ( math.factorial(2*l) * math.factorial(2*m) * math.factorial(2*n) ) )


    def s_func(self, orbA, orbB, gaussA_i, gaussB_i, coord):
        """ no idea what this even is """
        gamma = orbA['a'][gaussA_i] + orbB['a'][gaussB_i]
        P = self.P( orbA, orbB, gaussA_i, gaussB_i )
        PA_coord = (P[coord]-orbA['R'][coord] ) * 1.8897161646320724
        PB_coord = (P[coord]-orbB['R'][coord] ) * 1.8897161646320724
        
        if coord == 0:
            char = 'l'
        elif coord == 1:
            char = 'm'
        elif coord == 2:
            char = 'n'

        j_max_exc = int( (orbA[char] + orbB[char]) / 2 ) + 1
        sum = 0.0
        for j in range(j_max_exc):
            sum += ( self.f_func(2*j, orbA[char], orbB[char], PA_coord, PB_coord)
                     * misc.factorial2( 2*j - 1 )
                     / ( (2*gamma) ** j) )
        return math.sqrt( math.pi / gamma ) * sum
    
    
    def f_func( self, j, l, m, a, b ):
        """ no idea what this even is """
        sum_min = max(0, j-m)
        sum_max_exc = min(j, l) + 1
        ans = 0.0
        for k in range(sum_min, sum_max_exc):
            ans += ( special.binom(m, j-k)
                     * special.binom(l, k)
                     * ( a ** (l - k) )
                     * ( b ** (m + k - j) ) )
        return ans

    def sum_v_func(self, orbA, orbB, gaussA_i, gaussB_i, coord):
        """ sums v_funcs """
        char = ANG_COORDS[coord]  # 'l', 'm', or 'n'
        max_exc = orbA[char] + orbB[char]+1
        sum = 0.0
        for ind1 in range(max_exc):
            for ind2 in range(int(ind1/2)+1):
                for ind3 in range(int((ind1-2*ind2)/2)+1):
                    sum += 1.0
        
        return 0.0
    
    def v_func( self, l, r, i, lA, lB, Ax, Bx, Cx, gamma) :
        """ lotsa math """
        eps = 1/(4*gamma)
        return ( ((-1) ** l)
                 * self.f_func(j, l, m, a, b)
                 * ((-1) ** i)
                 * (math.factorial(l))
                 * (PCx ** (l -2*r -2*i))
                 * (eps ** (r+i))
                 / math.factorial(r)
                 / math.factorial(i)
                 / math.factorial( l -2*r -2*i ) )

    def boys_func(self, v, x) :
        if x < 10e-6:
            return  1/(2*v+1) - x/(2*v+3)
        else :
            return ( 0.5 * ( x ** (-v - 0.5))
                     * special.gammainc(v + 0.5, x)
                     * special.gamma(v + 0.5) )

    #
    # Vector functions
    #
    
    def P( self, orbA, orbB , gaussA_i, gaussB_i ):
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        return (alphaA*orbA['R'] + alphaB* orbB['R'])/(alphaA + alphaB)

    def AB2( self, vec1, vec2 ):
        return ( (vec1[0]-vec2[0]) ** 2
                 + (vec1[1]-vec2[1]) ** 2
                 + (vec1[2]-vec2[2]) ** 2 )
    
    #
    # Utility functions
    #
    
    def clone_ao(self, old_ao, ang_coord='l', change=0):
        """ duplicates the atomic orbital, changing one of the angular coordinates """
        new_ao = {
            'a' : old_ao['a'] ,
            'd' : old_ao['d'] ,
            'R' : old_ao['R'] ,
            'l' : old_ao['l'] ,
            'm' : old_ao['m'] ,
            'n' : old_ao['n']
        }
        new_ao[ang_coord] += change
        return new_ao

if __name__ == '__main__':
#    molecule_string = open('../../../extra-files/molecule.xyz','r').read()
    molecule_string = open('psi4_test/molecule.xyz','r').read()
    molecule = Molecule(molecule_string)
    ints = Integrals('sto3g.ini', molecule)

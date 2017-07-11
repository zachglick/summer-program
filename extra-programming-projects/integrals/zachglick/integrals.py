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

        
        print('\nMy Overlap:')
        for row in self.overlap:
            print(' '.join(['%.10f'%(num) for num in row]))
        print('\nPsi4 Overlap:')
        psi4overlap = np.load('psi4_test/psi4_overlap.npy')
        for row in psi4overlap:
            print(' '.join(['%.10f'%(num) for num in row]))
        print('')

        print('\nMy Kinetic:')
        for row in self.kinetic:
            print(' '.join(['%.10f'%(num) for num in row]))
        print('\nPsi4 Kinetic:')
        psi4kinetic = np.load('psi4_test/psi4_kinetic.npy')
        for row in psi4kinetic:
            print(' '.join(['%.10f'%(num) for num in row]))
        print('')

        print('\nMy Nuclear:')
        for row in self.nuclear:
            print(' '.join(['%.10f'%(num) for num in row]))
        print('\nPsi4 Nuclear:')
        psi4nuclear = np.load('psi4_test/psi4_potential.npy')
        for row in psi4nuclear:
            print(' '.join(['%.10f'%(num) for num in row]))
        print('')


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
        for ind, atom in enumerate(molecule):
            nuc_coords = atom[1]
            nuc_charge = molecule.charges[ind]
            for orbA_i, orbB_i in self.orbital_pairs:
#            print('~~~~~~~~~ nuc charge: %i ~~~~~~~~~'% nuc_charge)
#            for orbA_i, orbB_i in [(0,0),(1,1),(2,2),(3,3),(4,4)]:
                print('~~~~~~~ p: %i ~~~~~~~'% orbA_i)
                for gaussA_i, gaussB_i in self.gaussian_pairs:
                    ans = self.normalized_gaussian_nuclear(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i, nuc_coords, nuc_charge)
                    print('gauss %i and %i are %f' % (gaussA_i,gaussB_i,ans))
                    self.nuclear[ orbA_i, orbB_i ] += ans

    def normalized_gaussian_nuclear(self, orbA, orbB, gaussA_i, gaussB_i, nuc_coords, nuc_charge):
        """ this func calculates the contribution of these two gaussians to the total kinetic """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        gamma = alphaA + alphaB
        
#        print('!'+str(self.nuclear_summation(orbA, orbB, gaussA_i, gaussB_i, nuc_coords)))
        return ( orbA['d'][gaussA_i]
                * orbB['d'][gaussB_i]
                * self.normalization(orbA, gaussA_i)
                * self.normalization(orbB, gaussB_i)
                * (-1.0 * nuc_charge)
                * ( 2 * math.pi / gamma )
                * math.exp ( -alphaA
                             * alphaB
                             * self.AB2(orbA['R'], orbB['R']) * 1.8897161646320724 * 1.8897161646320724
                             / gamma )
                * self.nuclear_summation(orbA, orbB, gaussA_i, gaussB_i, nuc_coords))
    

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
    
    
    def f_func( self, jj, ll, mm, a, b ):
        """ no idea what this even is """
        sum_min = max(0, jj-mm)
        sum_max_exc = min(jj, ll) + 1
        ans = 0.0
        for kk in range(sum_min, sum_max_exc):
            ans += ( special.binom(mm, jj-kk)
                     * special.binom(ll, kk)
                     * ( a ** (ll - kk) )
                     * ( b ** (mm + kk - jj) ) )
        return ans

    def nuclear_summation(self, orbA, orbB, gaussA_i, gaussB_i, nuc_coords):
        """ sums v_funcs """
        P = self.P(orbA, orbB, gaussA_i, gaussB_i) # The 'center of mass' between atomic orbitals A and B
        PC2 = ((P[0]-nuc_coords[0])**2 + (P[1]-nuc_coords[1])**2 + (P[2]-nuc_coords[2])**2) * 1.8897161646320724 * 1.8897161646320724
        gamma = orbA['a'][gaussA_i] + orbB['a'][gaussB_i]
        
        PA = (P - orbA['R'])  * 1.8897161646320724
        PB = (P - orbB['R'])  * 1.8897161646320724
        PC = (P - nuc_coords) * 1.8897161646320724
        outer_sum = 0.0
        for l, r, i in self.weird_sum_thing(orbA, orbB, 0):
            middle_sum = 0.0
            for m, s, j in self.weird_sum_thing(orbA, orbB, 1):
                inner_sum = 0.0
                for n, t, k in self.weird_sum_thing(orbA, orbB, 2):
                    lmn = l + m + n
                    rst = r + s + t
                    ijk = i + j + k
                    inner_sum += ( self.boys_func(lmn - 2*rst - ijk, gamma * PC2)
                                   * self.v_func(n, t, k, orbA['n'], orbB['n'], gamma, PA[2], PB[2], PC[2]) )
                middle_sum += inner_sum * self.v_func(m, s, j, orbA['m'], orbB['m'], gamma, PA[1], PB[1], PC[1])
            outer_sum += middle_sum * self.v_func(l, r, i, orbA['l'], orbB['l'], gamma, PA[0], PB[0], PC[0])

        return outer_sum




        v_sums = [0.0, 0.0, 0.0]
        for coord in range(3): # sum over 3 cartesian coordinates (x, y, and z)
#            print('~~~%i~~~' % (coord))
            char = ANG_COORDS[coord]  # (x, y, or z) -> (l, m, n)
            PA = (P[coord] - orbA['R'][coord]) * 1.8897161646320724
            PB = (P[coord] - orbB['R'][coord]) * 1.8897161646320724
            PC = (P[coord] - nuc_coords[coord]) * 1.8897161646320724
            for lmn, rst, ijk in self.weird_sum_thing(orbA, orbB, coord):
                print((coord,self.v_func(lmn, rst, ijk, orbA[char], orbB[char], gamma, PA, PB, PC)))
                v_sums[coord] += self.v_func(lmn, rst, ijk, orbA[char], orbB[char], gamma, PA, PB, PC)
        boys_sum = 0.0
        for l, r, i in self.weird_sum_thing(orbA, orbB, 0):
            for m, s, j in self.weird_sum_thing(orbA, orbB, 1):
                for n, t, k in self.weird_sum_thing(orbA, orbB, 2):
                    lmn = l + m + n
                    rst = r + s + t
                    ijk = i + j + k
                    boys_sum += self.boys_func(lmn - 2*rst - ijk, gamma * PC2)

        print((v_sums[0],v_sums[1],v_sums[2]))
        return v_sums[0]*v_sums[1]*v_sums[2]*boys_sum
    
    def v_func(self, lmn, rst, ijk, ang_A, ang_B, gamma, PA, PB, PC) :
        """ lotsa math """
        eps = 1/(4*gamma)
        return ( ((-1) ** lmn)
                 * self.f_func(lmn, ang_A, ang_B, PA, PB)
                 * ((-1) ** ijk)
                 * (math.factorial(lmn))
                 * (PC ** (lmn -2*rst -2*ijk))
                 * (eps ** (rst+ijk))
                 / math.factorial( rst )
                 / math.factorial( ijk )
                 / math.factorial( lmn -2*rst -2*ijk ) )

    def boys_func(self, v, x) :
        """ something """
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
        return (alphaA*orbA['R'] + alphaB*orbB['R'])/(alphaA + alphaB)

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

    def weird_sum_thing(self, orbA, orbB, coord):
        """ this returns a list of all tuples """
        ang_char = ANG_COORDS[coord] # [0,1,2] -> ['l','m','n']
        lmn_max_exc = orbA[ang_char] + orbB[ang_char] + 1 # for summing to (lA+lB) or (mA+mB) or (nA+nB)
        lmn_rst_ijk = [] # tuples of (l,r,i) or (m,s,j) or (n,t,k)
        for lmn in range(lmn_max_exc):
            rst_max_exc = int(lmn/2.0)+1
            for rst in range(rst_max_exc):
                ijk_max_exc = int((lmn-2*rst)/2.0)+1
                for ijk in range(ijk_max_exc):
                    lmn_rst_ijk.append((lmn,rst,ijk)) # adding a tuple of valid coordinates to the list
        return lmn_rst_ijk

if __name__ == '__main__':
#    molecule_string = open('../../../extra-files/molecule.xyz','r').read()
    molecule_string = open('psi4_test/ho.xyz','r').read()
    molecule = Molecule(molecule_string)
    ints = Integrals('sto3g.ini', molecule)

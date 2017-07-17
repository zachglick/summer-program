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
ANGSTROM_TO_BOHR = 1.8897161646320724

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
                    
        # for cleaner iteration over orbitals and gaussians
        self.orbital_pairs = list(itertools.product(range(self.norb),range(self.norb)))
        self.gaussian_pairs = list(itertools.product(range(NGAUSS),range(NGAUSS)))
        self.orbital_quartets = list(itertools.product(range(self.norb),range(self.norb),range(self.norb),range(self.norb)))
        self.gaussian_quartets = list(itertools.product(range(NGAUSS),range(NGAUSS),range(NGAUSS),range(NGAUSS)))
                                     
        self.calc_overlap()
        self.calc_kinetic_energy()
        self.calc_nuclear_attraction()
        self.calc_electron_repulsion()

        np.set_printoptions(threshold=np.inf) # suppresses "...", printing all elements
        np.set_printoptions(precision=5) # reduces number of decimal places per element to 5
        np.set_printoptions(linewidth=200) # lengthens printable line-width from 75 characters
        np.set_printoptions(suppress=True) # suppresses hard-to-read exponential notation

#        For Testing:
#        print('\nMy Overlap:')
#        print(self.overlap)
#        print('\nPsi4 Overlap:')
#        psi4overlap = np.load('psi4_test/psi4_overlap.npy')
#        print(psi4overlap)
#        print('')
#
#        print('\nMy Kinetic:')
#        print(self.kinetic)
#        print('\nPsi4 Kinetic:')
#        psi4kinetic = np.load('psi4_test/psi4_kinetic.npy')
#        print(psi4kinetic)
#        print('')
#
#        print('\nMy Nuclear:')
#        print(self.nuclear)
#        print('\nPsi4 Nuclear:')
#        psi4nuclear = np.load('psi4_test/psi4_potential.npy')
#        print(psi4nuclear)
#        print('')
#
#        print('\n~~~~~~~~My Repulsion:~~~~~~~~')
#        print(self.twoelec)
#        print('\n~~~~~~~~Psi4 Repulsion:~~~~~~~~')
#        psi4twoelec = np.load('psi4_test/psi4_twoelec.npy')
#        print(psi4twoelec)
#        print('\n~~~~~~~~Difference:~~~~~~~~')
#        print(self.twoelec - psi4twoelec)
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
            for gaussA_i, gaussB_i in self.gaussian_pairs:
                self.overlap[ orbA_i, orbB_i ] += self.normalized_gaussian_overlap(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i)

    def normalized_gaussian_overlap(self, orbA, orbB, gaussA_i, gaussB_i):
        """ calculates gaussian contribution to overlap integral """
        return ( orbA['d'][gaussA_i]
                 * orbB['d'][gaussB_i]
                 * self.normalization(orbA, gaussA_i)
                 * self.normalization(orbB, gaussB_i)
                 * self.gaussian_overlap(orbA, orbB, gaussA_i, gaussB_i) )

    def gaussian_overlap(self, orbA, orbB, gaussA_i, gaussB_i):
        """ calculates the overlap of two gaussians """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        gamma = alphaA + alphaB
        return ( math.exp ( -alphaA
                            * alphaB
                            * self.disp_squared(orbA['R'], orbB['R']) * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
                            / gamma )
                * self.s_func(orbA, orbB, gaussA_i, gaussB_i, 0)   # Sx
                * self.s_func(orbA, orbB, gaussA_i, gaussB_i, 1)   # Sy
                * self.s_func(orbA, orbB, gaussA_i, gaussB_i, 2) ) # Sz

    def calc_kinetic_energy(self):
        """ calculates kinetic energy integrals"""
        self.kinetic = np.zeros((self.norb,self.norb))
        for orbA_i, orbB_i in self.orbital_pairs:
            for gaussA_i, gaussB_i in self.gaussian_pairs:
                self.kinetic[ orbA_i, orbB_i ] += self.normalized_gaussian_kinetic(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i)

    def normalized_gaussian_kinetic(self, orbA, orbB, gaussA_i, gaussB_i):
        """ calculates gaussian contribution to the kinetic energy integral """
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
                     + self.gaussian_overlap(orbA, self.clone_ao(orbB, 'n', -2), gaussA_i, gaussB_i) * -0.5 * nb * (nb-1) ) )
    
    def calc_nuclear_attraction(self):
        """ calc nuclear attraction integrals"""
        self.nuclear = np.zeros((self.norb,self.norb))
        for ind, atom in enumerate(self.molecule):
            nuc_coords = atom[1]
            nuc_charge = self.molecule.charges[ind]
            for orbA_i, orbB_i in self.orbital_pairs:
                for gaussA_i, gaussB_i in self.gaussian_pairs:
                    self.nuclear[ orbA_i, orbB_i ] += self.normalized_gaussian_nuclear(self.sto3g[ orbA_i ], self.sto3g[ orbB_i ], gaussA_i, gaussB_i, nuc_coords, nuc_charge)

    def normalized_gaussian_nuclear(self, orbA, orbB, gaussA_i, gaussB_i, nuc_coords, nuc_charge):
        """ calculates gaussian contribution to the nuclear attraction integral """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        gamma = alphaA + alphaB
        
        return ( orbA['d'][gaussA_i]
                * orbB['d'][gaussB_i]
                * self.normalization(orbA, gaussA_i)
                * self.normalization(orbB, gaussB_i)
                * (-1.0 * nuc_charge)
                * ( 2 * math.pi / gamma )
                * math.exp ( -alphaA
                             * alphaB
                             * self.disp_squared(orbA['R'], orbB['R']) * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
                             / gamma )
                * self.nuclear_summation(orbA, orbB, gaussA_i, gaussB_i, nuc_coords))
    

    def calc_electron_repulsion(self):
        """ calculates electron repulsion integral"""
        self.twoelec = np.zeros((self.norb,self.norb,self.norb,self.norb))
        for orbA_i, orbB_i, orbC_i, orbD_i in self.orbital_quartets:
            for gaussA_i, gaussB_i, gaussC_i, gaussD_i in self.gaussian_quartets:
                orbA, orbB, orbC, orbD = self.sto3g[orbA_i], self.sto3g[orbB_i], self.sto3g[orbC_i], self.sto3g[orbD_i]
                self.twoelec[ orbA_i, orbB_i , orbC_i, orbD_i] += self.normalized_gaussian_electron(orbA, orbB, orbC, orbD, gaussA_i, gaussB_i, gaussC_i, gaussD_i)
    
    def normalized_gaussian_electron(self, orbA, orbB, orbC, orbD, gaussA_i, gaussB_i, gaussC_i, gaussD_i):
        """ calculates gaussian contribution to the electron repulsion integral """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        alphaC = orbC['a'][gaussC_i]
        alphaD = orbD['a'][gaussD_i]
        gammaAB = alphaA + alphaB
        gammaCD = alphaC + alphaD
        
        return ( orbA['d'][gaussA_i]
                * orbB['d'][gaussB_i]
                * orbC['d'][gaussC_i]
                * orbD['d'][gaussD_i]
                * self.normalization(orbA, gaussA_i)
                * self.normalization(orbB, gaussB_i)
                * self.normalization(orbC, gaussC_i)
                * self.normalization(orbD, gaussD_i)
                * ( 2 * (math.pi ** 2) / (gammaAB*gammaCD) )
                *  (( math.pi / (gammaAB+gammaCD) ) ** 0.5 )
                * math.exp ( -alphaA
                            * alphaB
                            * self.disp_squared(orbA['R'], orbB['R']) * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
                            / gammaAB )
                * math.exp ( -alphaC
                            * alphaD
                            * self.disp_squared(orbC['R'], orbD['R']) * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
                            / gammaCD )
                * self.twoelec_summation(orbA, orbB, orbC, orbD, gaussA_i, gaussB_i, gaussC_i, gaussD_i))
    

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

        return math.sqrt( ((8*a)**(l+m+n)) # better form
                         * ( math.factorial(l) * math.factorial(m) * math.factorial(n) )
                         * ( (2*a/math.pi) ** 1.5 )
                         / ( math.factorial(2*l) * math.factorial(2*m) * math.factorial(2*n) ) )


    def s_func(self, orbA, orbB, gaussA_i, gaussB_i, coord):
        """ Useful function for gaussian overlap """
        gamma = orbA['a'][gaussA_i] + orbB['a'][gaussB_i]
        P = self.com( orbA, orbB, gaussA_i, gaussB_i )
        PA_coord = ( P[coord] - orbA['R'][coord] ) * ANGSTROM_TO_BOHR
        PB_coord = ( P[coord] - orbB['R'][coord] ) * ANGSTROM_TO_BOHR
        char = ANG_COORDS[coord]

        j_max_exc = int( (orbA[char] + orbB[char]) / 2 ) + 1
        sum = 0.0
        for j in range(j_max_exc):
            sum += ( self.f_func(2*j, orbA[char], orbB[char], PA_coord, PB_coord)
                     * misc.factorial2( 2*j - 1 )
                     / ( (2*gamma) ** j) )
        return math.sqrt( math.pi / gamma ) * sum
    
    def f_func( self, j, l, m, a, b ):
        """ Usefule function for (?) """
        sum_min = max(0, j-m)
        sum_max_exc = min(j, l) + 1
        ans = 0.0
        for k in range(sum_min, sum_max_exc):
            ans += ( special.binom(m, j-k)
                     * special.binom(l, k)
                     * ( a ** (l - k) )
                     * ( b ** (m + k - j) ) )
        return ans

    def nuclear_summation(self, orbA, orbB, gaussA_i, gaussB_i, nuc_coords):
        """ performs all 9 loops required to evaluate a single nuclear term """
        P = self.com(orbA, orbB, gaussA_i, gaussB_i) # The 'center of mass' between atomic orbitals A and B
        PC2 = ((P[0]-nuc_coords[0])**2 + (P[1]-nuc_coords[1])**2 + (P[2]-nuc_coords[2])**2) * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
        gamma = orbA['a'][gaussA_i] + orbB['a'][gaussB_i]
        PA = (P - orbA['R'])  * ANGSTROM_TO_BOHR
        PB = (P - orbB['R'])  * ANGSTROM_TO_BOHR
        PC = (P - nuc_coords) * ANGSTROM_TO_BOHR
        outer_sum = 0.0
        for l, r, i in self.triple_sum(orbA, orbB, 0):
            middle_sum = 0.0
            for m, s, j in self.triple_sum(orbA, orbB, 1):
                inner_sum = 0.0
                for n, t, k in self.triple_sum(orbA, orbB, 2):
                    lmn = l + m + n
                    rst = r + s + t
                    ijk = i + j + k
                    inner_sum += ( self.boys_func(lmn - 2*rst - ijk, gamma * PC2)
                                   * self.v_func(n, t, k, orbA['n'], orbB['n'], gamma, PA[2], PB[2], PC[2]) )
                middle_sum += inner_sum * self.v_func(m, s, j, orbA['m'], orbB['m'], gamma, PA[1], PB[1], PC[1])
            outer_sum += middle_sum * self.v_func(l, r, i, orbA['l'], orbB['l'], gamma, PA[0], PB[0], PC[0])
        return outer_sum


    def twoelec_summation(self, orbA, orbB, orbC, orbD, gaussA_i, gaussB_i, gaussC_i, gaussD_i):
        """ performs all 15 loops required to evaluate a single two-electron term"""
        P = self.com(orbA, orbB, gaussA_i, gaussB_i) # The 'center of mass' between atomic orbitals A and B
        Q = self.com(orbC, orbD, gaussC_i, gaussD_i) # The 'center of mass' between atomic orbitals C and D
        PQ2 = self.disp_squared(P,Q) * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR
        gammaAB = orbA['a'][gaussA_i] + orbB['a'][gaussB_i]
        gammaCD = orbC['a'][gaussC_i] + orbD['a'][gaussD_i]
        delta = 1/(4*gammaAB) + 1/(4*gammaCD)
        PA = (P - orbA['R'])  * ANGSTROM_TO_BOHR
        PB = (P - orbB['R'])  * ANGSTROM_TO_BOHR
        QC = (Q - orbC['R'])  * ANGSTROM_TO_BOHR
        QD = (Q - orbD['R'])  * ANGSTROM_TO_BOHR
        PQ = (P-Q) * ANGSTROM_TO_BOHR
        
        outer_sum = 0.0
        for lAB, rAB, lCD, rCD, i in self.quintuple_sum(orbA, orbB, orbC, orbD, 0):
            middle_sum = 0.0
            for mAB, sAB, mCD, sCD, j in self.quintuple_sum(orbA, orbB, orbC, orbD, 1):
                inner_sum = 0.0
                for nAB, tAB, nCD, tCD, k in self.quintuple_sum(orbA, orbB, orbC, orbD, 2):
                    v = lAB + lCD + mAB + mCD + nAB + nCD -2*(rAB + rCD + sAB + sCD + tAB + tCD) - (i + j + k)
                    inner_sum += ( self.boys_func(v, PQ2 / (4*delta))
                                  * self.g_func(nAB,nCD,tAB,tCD,k,orbA['n'],orbB['n'],PA[2],PB[2],gammaAB,orbC['n'],orbD['n'],QC[2],QD[2],gammaCD,PQ[2]) )
                middle_sum += inner_sum * self.g_func(mAB,mCD,sAB,sCD,j,orbA['m'],orbB['m'],PA[1],PB[1],gammaAB,orbC['m'],orbD['m'],QC[1],QD[1],gammaCD,PQ[1])
            outer_sum += middle_sum * self.g_func(lAB,lCD,rAB,rCD,i,orbA['l'],orbB['l'],PA[0],PB[0],gammaAB,orbC['l'],orbD['l'],QC[0],QD[0],gammaCD,PQ[0])
        return outer_sum

    def v_func(self, lmn, rst, ijk, ang_A, ang_B, gamma, PA, PB, PC) :
        """ Useful function for nuclear attraction integral """
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
        """ Analytic solution to  """
        if x < 10e-6:
            return  1/(2*v+1) - x/(2*v+3)
        else :
            return ( 0.5 * ( x ** (-v - 0.5))
                     * special.gammainc(v + 0.5, x)
                     * special.gamma(v + 0.5) )

    def theta_func(self, lmn, ang_A, ang_B, a, b, rst, gamma):
        """ Useful function for electron repulsion integral """
        return (self.f_func(lmn, ang_A, ang_B, a, b)
                * math.factorial(lmn)
                * (gamma ** (rst-lmn))
                / math.factorial(rst)
                / math.factorial(lmn - 2*rst) )

    def g_func( self, lmnAB, lmnCD, rstAB, rstCD, ijk, ang_A, ang_B, PA, PB, gammaAB, ang_C, ang_D, QC, QD, gammaCD, PQ):
        """ Useful function for electron repulsion integral """
        delta = 1/(4.0*gammaAB) + 1/(4.0*gammaCD)
        
        return ( ((-1) ** lmnAB)
                 * self.theta_func(lmnAB,ang_A,ang_B,PA,PB,rstAB,gammaAB)
                 * self.theta_func(lmnCD,ang_C,ang_D,QC,QD,rstCD,gammaCD)
                 * ((-1) ** ijk)
                 * ((2*delta) ** (2*(rstAB + rstCD)))
                 * math.factorial(lmnAB + lmnCD - 2*rstAB - 2*rstCD)
                 * (delta ** ijk)
                 * (PQ ** (lmnAB + lmnCD - 2*(rstAB + rstCD + ijk)))
                 / ((4*delta) ** (lmnAB + lmnCD))
                 / math.factorial(ijk)
                 / math.factorial(lmnAB + lmnCD - 2*(rstAB + rstCD + ijk)) )

    #
    # Vector functions
    #
    
    def com( self, orbA, orbB , gaussA_i, gaussB_i ):
        """ The 'center of mass' between orbitals A and B """
        alphaA = orbA['a'][gaussA_i]
        alphaB = orbB['a'][gaussB_i]
        return (alphaA*orbA['R'] + alphaB*orbB['R'])/(alphaA + alphaB)

    def disp_squared( self, vec1, vec2 ):
        """ The squared displacement separating the two vectors """
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

    def triple_sum(self, orbA, orbB, coord):
        """ this returns a list of tuples for evaluating the triple summations in the nuclear atttraction integral"""
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

    def quintuple_sum(self, orbA, orbB, orbC, orbD, coord):
        """ this returns a list of tuples for evaluating the quintuple summations in the nuclear atttraction integral"""
        ang_char = ANG_COORDS[coord] # [0,1,2] -> ['l','m','n']
        lmnAB_max_exc = orbA[ang_char] + orbB[ang_char] + 1 # for summing to (lA+lB) or (mA+mB) or (nA+nB)
        lmnCD_max_exc = orbC[ang_char] + orbD[ang_char] + 1 # for summing to (lC+lD) or (mC+mD) or (nC+nD)
        lmnAB_rstAB_lmnCD_rstCD_ijk = [] # tuples of (lAB,rAB,lAB,rAB,i) or (mAB,sAB,mAB,sAB,j) or (nAB,tAB,nAB,tAB,k)
        for lmnAB in range(lmnAB_max_exc):
            rstAB_max_exc = int(lmnAB/2.0)+1
            for rstAB in range(rstAB_max_exc):
                for lmnCD in range(lmnCD_max_exc):
                    rstCD_max_exc = int(lmnCD/2.0)+1
                    for rstCD in range(rstCD_max_exc):
                        ijk_max_exc = int((lmnAB+lmnCD-2*rstAB-2*rstCD)/2.0)+1
                        for ijk in range(ijk_max_exc):
#                            print((lmnAB,rstAB,lmnCD,rstCD,ijk,ijk_max_exc))
                            lmnAB_rstAB_lmnCD_rstCD_ijk.append((lmnAB,rstAB,lmnCD,rstCD,ijk)) # adding a tuple of valid coordinates to the list
        return lmnAB_rstAB_lmnCD_rstCD_ijk


if __name__ == '__main__':
#    molecule_string = open('../../../extra-files/molecule.xyz','r').read()
    molecule_string = open('../../../extra-files/molecule.xyz','r').read()
    molecule = Molecule(molecule_string)
    ints = Integrals('sto3g.basis', molecule)
    print('Integrals Computed (import this class to use)')

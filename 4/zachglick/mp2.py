import sys
sys.path.append('../../3/zachglick')

import numpy as np

from rhf import RHF

rhf = RHF('Config.ini', printResults=False)
twoelec_MO = np.zeros(rhf.twoelec.shape)

# Total, Occupied, and Virtual orbital counts
norb = rhf.norb
nocc = rhf.nocc
nvirt = norb-nocc

# Naive transform implementation, θ(?) complexity

for I in range(nocc):
    for J in range(nocc):
        for A in range(nocc, norb):
            for B in range(nocc, norb):
                
                for u in range(norb):
                    for v in range(norb):
                        for p in range(norb):
                            for s in range(norb):

                                twoelec_MO[I,J,A,B] += rhf.twoelec[u,p,v,s] * rhf.C[u, I] * rhf.C[v, J] * rhf.C[p, A] * rhf.C[s, B]

# Better transform implementation, θ(?) complexity

#pass


#                  SUM         |< I J || A B >|^2
#  energy_corr =   OVER        ------------------
#                  I,J,A,B         eI+eJ-eA-eB

orbital_energies = rhf.energies
energy_corr = 0.0

for I in range(nocc):
    for J in range(nocc):
        for A in range(nocc, norb):
            for B in range(nocc, norb):
                denominator = orbital_energies[I]+orbital_energies[J]-orbital_energies[A]-orbital_energies[B]
                numerator = twoelec_MO[I,J,A,B]*(2*twoelec_MO[I,J,A,B] - twoelec_MO[I,J,B,A])
                energy_corr += numerator/denominator
energy_total = rhf.energy_SCF+energy_corr

print('SCF Energy:       %.12e' % (rhf.energy_SCF))
print('MP2 Correlation:  %.12e' % (energy_corr))
print('Total Energy:     %.12e' % (energy_total))

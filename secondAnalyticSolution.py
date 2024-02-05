# plotting code for one of the analytic solutions in the paper

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from numpy.linalg import norm
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import fsolve, minimize
from scipy.special import i0, k0
from scipy.integrate import quad

from plotsOfFieldLines import psi, chi, Bz, Br, BMag

NGrid = 50
NBigGrid = 1000
NATerms = 10 # for the vacuum solution
rI = 0.1 # inner limit on plasma region
rF = 0.11 # outer limit on plasma region
constantCoeff = .2
psiI = psi(rI, 0.) # psi coordinate corresonding to rI at midplane
psiF = psi(rF, 0.) # psi coordinate corresponding to rF at midplane
print("psiI = ", psiI)
print("psiF = ", psiF)
zLimit = 0.5
rLimit = 1.5*rF
chiLimit = chi(0., 1.05*zLimit)
phiCutoff = False

# invert magnetic coordinates
def invertCoords(chiVal, psiVal):
    return fsolve(lambda X: [psi(X[0],X[1])-psiVal,chi(X[0],X[1])-chiVal], [.1,.1])

def phiBoundary(chi):
    return np.exp(-chi*chi/.1) #* 2e-5

def plasmaPhi(chiVal, psiVal):
    sgn = -1
    prefactor = sgn * 100

    # 4e2 and .7e2 are two interesting values

    firstTerm = np.sqrt(phiBoundary(chiVal))
    secondTerm = np.zeros_like(chiVal)

    for idx, value in np.ndenumerate(chiVal):
        secondTerm[idx] = quad(lambda p: 1./np.abs(invertCoords(value, p)[0]), psiI, psiVal[idx])[0]

    return (firstTerm + prefactor * secondTerm)**2

def omegaRatio(chiVal, psiVal, ZVal, prefactor):
    firstTerm = np.sqrt(phiBoundary(chiVal))
    secondTerm = np.zeros_like(chiVal)
    rVal = np.zeros_like(chiVal)
    zVal = np.zeros_like(chiVal)

    for idx, value in np.ndenumerate(chiVal):
        secondTerm[idx] = quad(lambda p: 1./np.abs(invertCoords(value, p)[0]), psiI, psiVal[idx])[0]
        rVal[idx], zVal[idx] = np.abs(invertCoords(chiVal[idx], psiVal[idx]))

    return (1./(ZVal*rVal*BMag(rVal, zVal)*prefactor)) * np.sqrt(ZVal+1) * (firstTerm + prefactor*np.sqrt(ZVal+1)*secondTerm)

# vacuum series solution for phi
# inOut distinguishes between the inner and outer vacuum regions
def vacuumPhi(r, z, A, inOut):
    phiVal = np.zeros_like(r)
    for i in range(len(A)):
        if inOut == 'in':
            phiVal += A[i] * np.cos(2. * pi * i * z) * i0(2. * pi * i * r)
        elif inOut == 'out':
            if i == 0:
                phiVal += A[i]
            else:
                phiVal += A[i] * np.cos(2. * pi * i * z) * k0(2. * pi * i * np.abs(r))
    return phiVal

chis1D = np.linspace(-chiLimit, chiLimit, NGrid)
psis1D = np.linspace(psiI, psiF, NGrid)

chis, psis = np.meshgrid(chis1D, psis1D)
phis = plasmaPhi(chis, psis)
omegaRatios = omegaRatio(chis, psis, 1., -1e2)
interp = RegularGridInterpolator((chis1D, psis1D), np.transpose(phis))

fig, ax = plt.subplots()
a = ax.contour(chis, psis, phis, levels=20)
fig.colorbar(a)
ax.set_xlabel('$\chi$')
ax.set_ylabel('$\psi$')
#plt.savefig('phis2Slice.pdf')
plt.show()

rs = np.zeros_like(chis)
for i in range(len(chis)):
    for j in range(len(chis[0])):
        rs[i,j] = invertCoords(chis[i,j], psis[i,j])[0]
fig, ax = plt.subplots()
a = ax.contour(chis, psis, np.abs(rs), levels=20)
fig.colorbar(a)
plt.show()

plt.plot(psis1D, plasmaPhi(0.*np.ones_like(psis1D), psis1D), label='$\chi = 0$')
plt.plot(psis1D, plasmaPhi(.2*np.ones_like(psis1D), psis1D), label='$\chi = .2$')
plt.plot(psis1D, plasmaPhi(.4*np.ones_like(psis1D), psis1D), label='$\chi = .4$')
plt.legend()
plt.show()

# now we do the full solution

Z = np.linspace(-zLimit,zLimit,NBigGrid)
R = np.linspace(-rLimit,rLimit,NBigGrid)
Z, R = np.meshgrid(Z, R)

# calculate phi = 0 split
if phiCutoff == True:
    z1D = np.linspace(-zLimit,zLimit,NGrid)
    rSplit = fsolve(lambda r: plasmaPhi(chi(r,z1D), psi(r,z1D)), np.ones(NGrid)*rI)
    rSplitInterp = interp1d(z1D, rSplit)

# calculate lower boundary coefficients
NATerms = 10
zs = np.linspace(-zLimit, zLimit, NGrid)
rs = fsolve(lambda r: psi(r, zs) - psiI, .1*np.ones_like(zs))
boundaryPhi = phiBoundary(chi(rs, zs))
A0 = [ 8.57427358e-01, -5.73297616e-01, -1.22779454e-01, -2.03598234e-02, -2.20491911e-03, -4.18257320e-05,  3.97484010e-05,  9.34010188e-06, 1.14773712e-06,  2.13507537e-08]
#A0 = [.8, -.6, -.1, -.2]
A = minimize(lambda A: norm(vacuumPhi(rs, zs, A, 'in') - boundaryPhi), A0)
plt.plot(zs, boundaryPhi, label='desired boundary')
plt.plot(zs, vacuumPhi(rs, zs, A.x, 'in'), linestyle='--', label='numerical boundary')
plt.legend()
plt.show()

# calculate upper boundary coefficients
if phiCutoff == False:
    rs = fsolve(lambda r: psi(r, zs) - psiF, .1*np.ones_like(zs))
    boundaryPhi = plasmaPhi(chi(rs, zs), psi(rs, zs))
    A0 = [ 1.75769562, -1, -0.60598674, -0.50857991, -0.43610693, -0.38379535, -0.34203612, -0.2984088,  -0.23224787, -0.11821791]
    AUpper = minimize(lambda A: norm(vacuumPhi(rs, zs, A, 'out') - boundaryPhi), A0)
    plt.plot(zs, boundaryPhi, label='desired boundary')
    plt.plot(zs, vacuumPhi(rs, zs, AUpper.x, 'out'), linestyle='--', label='numerical boundary')
    plt.legend()
    plt.show()

# CALCULATE PHI

PHI = np.zeros_like(Z)
#pindx = np.where((psi(R,Z) >= psiI) & (psi(R,Z) <= psiF))
if phiCutoff == True:
    pindx = np.where((psi(R,Z) >= psiI) & (np.abs(R) <= rSplitInterp(Z)) & (psi(R,Z) <= psiF))
else:
    pindx = np.where((psi(R,Z) >= psiI) & (psi(R,Z) <= psiF))
PHI[pindx] = interp((np.abs(chi(R[pindx], Z[pindx])), psi(R[pindx], Z[pindx])))
innerIndx = np.where(psi(R,Z) < psiI)
PHI[innerIndx] = vacuumPhi(R[innerIndx], Z[innerIndx], A.x, 'in')
if phiCutoff == False:
    outerIndx = np.where(psi(R,Z) > psiF)
    PHI[outerIndx] = vacuumPhi(R[outerIndx], Z[outerIndx], AUpper.x, 'out')

extendedR = -R
fig, ax = plt.subplots()
CS = ax.contour(Z, R, psi(R,Z), levels=25, linestyles='--', colors='black', alpha=.2)
CS2 = ax.contour(Z, R, chi(R,Z), linestyles='--', colors='black', alpha=.2)
CS3 = ax.contour(Z, R, PHI, levels=25)
cbar = fig.colorbar(CS3, label='$\phi$ / $\phi_0$')
ax.set_xlabel('z / L')
ax.set_ylabel('r / L')
#plt.savefig('phis2WithLaplace_fullFluxSurfaces.png', dpi=500)
plt.show()

fig, ax = plt.subplots()
c = ax.pcolormesh(Z, R, PHI, shading='gouraud')
fig.colorbar(c, ax=ax, label='$\phi$ / $\phi_0$')
ax.set_xlabel('z / L')
ax.set_ylabel('r / L')
#plt.savefig('phis2WithLaplace_colormap_fullFluxSurfaces.png', dpi=500)
plt.show()

zs1D = np.linspace(-zLimit,zLimit,NBigGrid)
rs1D = np.linspace(-rLimit,rLimit,NBigGrid)
interpFull = RegularGridInterpolator((zs1D, rs1D), np.transpose(PHI))

ZD, RD = np.meshgrid(zs1D[1:-1], rs1D[1:-1])
dr = rLimit / NBigGrid
dz = zLimit / NBigGrid
PHID = interpFull((ZD, RD))

#fig, ax = plt.subplots()
#c = ax.pcolormesh(ZD, RD, PHID, shading='gouraud')
#plt.show()

ER = - (interpFull((ZD, RD+dr)) - interpFull((ZD, RD-dr))) / (2. * dr)

#fig, ax = plt.subplots()
#c = ax.pcolormesh(ZD, RD, ER, shading='gouraud')
#fig.colorbar(c, ax=ax, label='$E_r$')
#plt.show()

EZ = - (interpFull((ZD+dz, RD)) - interpFull((ZD-dz, RD))) / (2. * dz)

#fig, ax = plt.subplots()
#c = ax.pcolormesh(ZD, RD, EZ, shading='gouraud')
#fig.colorbar(c, ax=ax, label='$E_z$')
#plt.show()

EMag = np.sqrt(ER*ER+EZ*EZ)
fig, ax = plt.subplots()
c = ax.pcolormesh(ZD, RD, EMag, shading='gouraud', cmap='plasma')
fig.colorbar(c, ax=ax, label='|E L / $\phi_0$|')
ax.set_xlabel('z / L')
ax.set_ylabel('r / L')
#plt.savefig('E_fullFluxSurfaces_plasmacmap.png', dpi=500)
plt.show()


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
fig, ax = plt.subplots()
c = ax.pcolormesh(ZD, RD, EMag, shading='gouraud', cmap='plasma')
fig.colorbar(c, ax=ax, label='|E L / $\phi_0$|')
ax.set_xlabel('z / L')
ax.set_ylabel('r / L')

axins = zoomed_inset_axes(ax, 6, loc='center right')
c = axins.pcolormesh(ZD, RD, EMag, shading='gouraud', cmap='plasma')
axins.set_xlim(.45, .495)
axins.set_ylim(0.054, .065)
plt.xticks(visible=False)
plt.yticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
#plt.savefig('E_fullFluxSurfaces_plasmacmap_inset.png', dpi=500)
plt.show()

BR = np.zeros_like(ER)
BZ = np.zeros_like(EZ)
for i in range(NBigGrid-2):
    for j in range(NBigGrid-2):
        BR[i,j] = Br(RD[i,j], ZD[i,j])
        BZ[i,j] = Bz(RD[i,j], ZD[i,j])

fig, ax = plt.subplots(2)
c = ax[0].pcolormesh(ZD, RD, BZ, shading='gouraud', cmap='plasma')
fig.colorbar(c, ax=ax[0], label='$B_z / B_0$')
ax[0].set_xlabel('z / L')
ax[0].set_ylabel('r / L')
#ax[0].set_title('$B_z / B_0$')

c = ax[1].pcolormesh(ZD, RD, BR, shading='gouraud', cmap='plasma')
fig.colorbar(c, ax=ax[1], label='$B_r / B_0$')
ax[1].set_xlabel('z / L')
ax[1].set_ylabel('r / L')
#ax[1].set_title('$B_r / B_0$')
plt.show()

ExB = np.zeros_like(ER)
for i in range(NBigGrid-2):
    for j in range(NBigGrid-2):
        ExB[i,j] = -ER[i,j] * BZ[i,j] + EZ[i,j] * BR[i,j]

toDisplay = np.zeros_like(ER)
for i in range(NBigGrid-2):
    for j in range(NBigGrid-2):
        BrMag = np.sqrt(BR[i,j]*BR[i,j] + BZ[i,j]*BZ[i,j])
        toDisplay[i,j] = ExB[i,j] / ((BrMag**3) * RD[i,j])

# code to produce plots of field lines, with utility functions for 
# field configuration

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.special import i0, i1
from scipy.interpolate import interp1d

# parameters controlling field setup
alpha = 0.5
zLimit = .6
rLimit = .2
mu = 1e-3

# plotting/grid parameter
NGrid = 100

# flux function
def psi(r, z):
    u = 2. * pi * z
    return .5 * r * r * (1. - (alpha/pi) * np.cos(u) * (1./r) * i1(2.*pi*r))

# field-length-like coordinate
def chi(r, z):
    return z - (alpha / (2.*pi)) * np.sin(2.*pi*z) * i0(2.*pi*r)

# z component of B field
def Bz(r, z):
    return 1. - alpha * np.cos(2.*pi*z) * i0(2.*pi*r)

# r component of B field
def Br(r, z):
    return - alpha * np.sin(2.*pi*z) * i1(2.*pi*r)

# |B|
def BMag(r, z):
    BrVal = Br(r, z)
    BzVal = Bz(r, z)
    return np.sqrt(BrVal * BrVal + BzVal * BzVal)

# parallel component of phi decomposition
def phiPar(r, z):
    chiVal = chi(r,z)
    chiVal0 = chi(0,.5)
    return -(chiVal/chiVal0)**2

# perpendicular component of phi decomposition
def phiPerp(r, z):
    psiVal = psi(r,z)
    psiVal0 = psi(.1,0)
    return -psiVal/psiVal0

# parallel component of electric field
def EPar(r, z):
    dPhiDChi = 2. * chi(r, z)
    return - np.array([dPhiDChi * Br(r, z), np.zeros_like(Br(r,z)), dPhiDChi * Bz(r, z)])

# perpendicular component of electric field
def EPerp(r, z):
    dPhiDPsi = 20.
    return - dPhiDPsi * np.array([np.abs(r) * Bz(r, z), np.zeros_like(Br(r,z)), -np.abs(r) * Br(r, z)])

def runTests():
    Z = np.linspace(-zLimit,zLimit,NGrid)
    R = np.linspace(-rLimit,rLimit,NGrid)
    Z, R = np.meshgrid(Z, R)

    EParVal = EPar(R, Z)
    BVal = np.array([Br(R, Z), np.zeros_like(Br(R, Z)), Bz(R, Z)])
    EPerpVal = EPerp(R, Z)
    VExBVal = (EPerpVal[2]*BVal[0] - EPerpVal[0]*BVal[2]) / (BVal[0]*BVal[0] + BVal[1]*BVal[1] + BVal[2]*BVal[2])
    OmegaExBVal = VExBVal / R
    fig, ax = plt.subplots()
    ax.contour(Z, R, VExBVal)
    plt.show()

    rdc = 5
    ETot = EParVal + EPerpVal
    J = EParVal + mu * EPerpVal
    JE = J[0]*ETot[0] + J[1]*ETot[1] + J[2]*ETot[2]
    plt.quiver(Z[::rdc,::rdc], R[::rdc,::rdc], ETot[2][::rdc,::rdc], ETot[0][::rdc,::rdc])
    plt.show()

    plt.quiver(Z[::rdc,::rdc], R[::rdc,::rdc], J[2][::rdc,::rdc], J[0][::rdc,::rdc])
    plt.title('Current')
    plt.xlabel('z / L')
    plt.ylabel('r / L')
    #plt.savefig('JQuiver.pdf')
    plt.show()

    fig, ax = plt.subplots()
    dissipation = ax.pcolormesh(Z, R, np.log(JE), shading='gouraud')
    dissipation = ax.pcolormesh(Z, R, np.log(JE), shading='gouraud')
    ax.set_xlabel('z / L')
    ax.set_ylabel('r / L')
    cbar = fig.colorbar(dissipation, label = '$\log( \, \mathbf{j} \cdot \mathbf{E})$')
    #plt.savefig('dissipation.pdf')
    plt.show()

    psiLevels = 21
    fig, ax = plt.subplots()
    CS = ax.contour(Z, R, psi(R,Z), levels=psiLevels)
    CS2 = ax.contour(Z, R, chi(R,Z), linestyles='--', colors='black', alpha=.2)
    ax.set_xlabel('z / L')
    ax.set_ylabel('r / L')
    cbar = fig.colorbar(CS, label='$\psi$  /  $L^2 B_0$')
    #plt.savefig('./psiContours.pdf')
    plt.show()

    fig, ax = plt.subplots()
    CSUnfancy = ax.contour(Z, R, psi(R,Z), levels=psiLevels, linestyles='--', colors='black', alpha=.2)
    CS2Fancy = ax.contour(Z, R, chi(R,Z), levels=15)
    ax.set_xlabel('z / L')
    ax.set_ylabel('r / L')
    cbar = fig.colorbar(CS2Fancy, label='$\chi$  /  $L B_0$')
    #plt.savefig('./chiContours.pdf')
    plt.show()

    phiLevels = 20
    fig, ax = plt.subplots()
    #CSUnfancy = ax.contour(Z, R, psi(R,Z), levels=psiLevels, linestyles='--', colors='black', alpha=.4)
    #CS2Unfancy = ax.contour(Z, R, chi(R,Z), linestyles='--', colors='black', alpha=.2)
    phiCS = ax.contour(Z, R, phiPar(R,Z) + phiPerp(R,Z), levels=phiLevels)
    cbar = fig.colorbar(phiCS, label='Isopotential surfaces ($\phi$ / $\phi_0$)')
    ax.set_xlabel('z / L')
    ax.set_ylabel('r / L')
    #plt.savefig('./phiContours.pdf')
    plt.show()

    phiLevels = 16
    fig, ax = plt.subplots()
    #CSUnfancy = ax.contour(Z, R, psi(R,Z), levels=psiLevels, linestyles='--', colors='black', alpha=.2)
    #CS2Unfancy = ax.contour(Z, R, chi(R,Z), linestyles='--', colors='black', alpha=.2)
    phiCS = ax.contour(Z, R, phiPerp(R,Z), levels=phiLevels)
    cbar = fig.colorbar(phiCS, label='Isopotential surfaces ($\phi$ / $\phi_0$)')
    ax.set_xlabel('z / L')
    ax.set_ylabel('r / L')
    #plt.savefig('./phiContoursWithPhiPar.pdf')
    plt.show()

    # now we want to try plotting just a particular region
    NLevels = 10
    r0 = 0.05
    r1 = 0.0535
    psiLevels = psi(np.linspace(r0, r1, NLevels), np.zeros(NLevels))
    fig, ax = plt.subplots()
    CS = ax.contour(Z, R, psi(R,Z), levels=psiLevels)
    CS2 = ax.contour(Z, R, chi(R,Z), linestyles='--', colors='black', alpha=.2)
    ax.set_xlabel('z / L')
    ax.set_ylabel('r / L')
    ax.set_ylim(-.075,.075)
    cbar = fig.colorbar(CS, label='$\psi$  /  $L^2 B_0$')
    #plt.savefig('./narrowRegion.pdf')
    plt.show()

    fig, ax = plt.subplots()
    BMagPlot = ax.contour(Z, R, BMag(R, Z), levels=50)
    cbar = fig.colorbar(BMagPlot)
    plt.xlabel('z / L')
    plt.ylabel('z / L')
    ax.set_ylim(-.075, .075)
    plt.show()

# makes a cartoon picture of a field geometry
def cartoonDiverter():
    XY = [
    (-1,  3),
    (-.5, .5),
    (0,   1),
    (.5, .5),
    (1,  3),
    ]

    X = [xy[0] for xy in XY]
    Y = [xy[1] for xy in XY]

    a = interp1d(X, Y, kind='quadratic')
    XFine = np.linspace(np.min(X), np.max(X), 100)

    fig, ax = plt.subplots()
    ax.axis('off')
    for scaling in np.linspace(0, 1, 5)[0:]:
        ax.plot(XFine, scaling*a(XFine), label='$\phi$ / $\phi_0$ = '+str(scaling), color='black', alpha=.1+scaling*.9)
        ax.plot(XFine, -scaling*a(XFine), color='black', alpha=.1+scaling*.9)
    ax.set_xlim(-1,1)
    # plt.savefig('cartoonDiverter.pdf')
    plt.show()

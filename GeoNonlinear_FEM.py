from typing import Self

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import ldl
from scipy.linalg import eig


def B2D_LR(ue, EA, EI, GAq, l):
    """
    2D beam element for large rotations
        - Timoshenko theory
        - 2 nodes
        - reduced (one point) integration to avoid shear locking

    ue              ... nodal displacements
    EA, EI, GAq,    ... axial, bending and shear stiffness
    l               ... element length
    """

    u1, w1, phi1 = ue[0], ue[1], ue[2]
    u2, w2, phi2 = ue[3], ue[4], ue[5]

    # internal force vector
    t1 = 0.1e1 / l
    t2 = u1 * t1
    t3 = u2 * t1
    t4 = t2 - t3
    t6 = t1 * t4 + t1
    t8 = t4**2
    t12 = w1 * t1 - w2 * t1
    t13 = t12**2
    t15 = t8 / 2 + t13 / 2 + t2 - t3
    t17 = phi1 + phi2
    t18 = np.sin(t17 / 2)
    t19 = t18 * t1
    t21 = np.cos(t17 / 2)
    t23 = t21 * t12 + t18 * t4 + t18
    t24 = t23 * GAq
    t25 = t24 * t19
    t26 = t1 * t21
    t29 = phi1 * t1 - phi2 * t1
    t30 = t29**2
    t34 = -t12 * t18 + t4 * t21 + t21
    t35 = t34 * EI * t30
    t36 = t35 * t26
    t44 = t15 * EA * t1 * t12 - t35 * t19 + t24 * t26
    t47 = t23 * GAq * t34 / 2
    t48 = -t29 * t23 / 2
    t49 = t1 * t34
    t52 = t29 * t34

    fin = np.zeros(6)
    fin[0] = (t15 * EA * t6 + t25 + t36) * l
    fin[1] = t44 * l
    fin[2] = (t47 + t52 * EI * (t48 + t49)) * l
    fin[3] = (-t15 * EA * t6 - t25 - t36) * l
    fin[4] = -t44 * l
    fin[5] = (t47 + t52 * EI * (t48 - t49)) * l

    # element stiffness matrix
    t1 = 0.1e1 / l
    t2 = t1 * (u1 - u2)
    t3 = t1 * (w1 - w2)
    t4 = t3**2
    t5 = 0.1e1 / 0.2e1
    t6 = t5 * (t2**2 + t4) + t2
    t2 = t2 + 1
    t7 = t1 * t2
    t8 = t5 * (phi1 + phi2)
    t9 = np.sin(t8)
    t8 = np.cos(t8)
    t10 = t1 * (phi1 - phi2)
    t11 = t8**2
    t12 = t9**2
    t13 = t10**2 * EI
    t14 = t1**2
    t15 = (EA * t6 + GAq * t12 + t13 * t11) * t14
    t16 = t1 * t9
    t17 = t16 * t8 * (-t13 + GAq)
    t18 = t7 * EA
    t19 = t1 * (t18 * t3 + t17)
    t20 = t9 * t2 + t3 * t8
    t2 = t8 * t2 - t3 * t9
    t21 = t9 * t2
    t22 = t9 * t2
    t23 = t8 * t20
    t24 = t8 * t2
    t25 = 2 * t24 * t14 * t10 * EI
    t26 = t5 * t1 * (GAq * (t23 + t22) + t13 * (-t8 * t20 - t21))
    t27 = t25 + t26
    t25 = -t25 + t26
    t26 = l * t19
    t18 = l * (t18 * t7 + t15)
    t19 = l * t19
    t4 = t14 * ((t4 + t6) * EA + GAq * t11 + t13 * t12)
    t6 = t9 * t20
    t11 = t8 * t2
    t12 = 2 * t21 * t14 * t10 * EI
    t14 = t5 * t1 * (GAq * (-t11 + t6) + t13 * (-t9 * t20 + t24))
    t21 = -t12 - t14
    t3 = t1 * (EA * t3 * t7 + t17)
    t12 = t12 - t14
    t14 = l * t4
    t17 = l * t3
    t4 = l * t4
    t9 = t5 * t9 * t10
    t24 = t1 * t8
    t28 = t1 * (-t9 + t24)
    t29 = t5 * t20 * t10
    t30 = t1 * t2
    t31 = -t29 + t30
    t32 = t31 * t1
    t33 = t32 * t8
    t34 = EI * t10
    t35 = t1 * GAq
    t22 = t5 * t35 * (t23 + t22)
    t23 = t5 * t8 * t10
    t36 = -t1 * (t23 + t16)
    t37 = t16 * t31
    t5 = t5 * t35 * (-t11 + t6)
    t6 = -t2 * t10 / 4
    t11 = t20 * t1
    t35 = EI * t2
    t20 = t20**2
    t38 = -GAq * (-(t2**2) + t20) / 4
    t13 = -t2 * (-GAq * t2 + t13 * t2) / 4 - t20 * GAq / 4
    t3 = l * t3
    t9 = -t1 * (t9 + t24)
    t20 = -t29 - t30
    t24 = t20 * t1
    t8 = t24 * t8
    t1 = t1 * (-t23 + t16)
    t16 = t16 * t20
    t20 = t20 * EI
    k = np.array([[l * (EA * t7 ** 2 + t15),t26,l * t27,-t18,-t19,l * t25],[t26,t14,l * t21,-t17,-t4,l * t12],[l * (t34 * (t2 * t28 + t33) + t22),l * (t34 * (t2 * t36 - t37) - t5),l * (t35 * (t10 * (-t11 + t6) + t32) + t38 - t29 * t31 * EI),l * (t34 * (-t2 * t28 - t33) - t22),l * (t34 * (-t2 * t36 + t37) + t5),l * (t13 - t31 * EI * (t29 + t30))],[-t18,-t17,-l * t27,l * (EA * t7 ** 2 + t15),t3,-l * t25],[-t19,-t4,-l * t21,t3,t14,-l * t12],[l * (t34 * (t9 * t2 + t8) + t22),l * (t34 * (t1 * t2 - t16) - t5),l * (t13 + t20 * (-t29 + t30)),l * (t34 * (-t9 * t2 - t8) - t22),l * (t34 * (-t1 * t2 + t16) + t5),l * (t35 * (t10 * (t6 + t11) - t24) + t38 - t20 * t29)]])

    # geometrix stiffness matrix
    t1 = 0.1e1 / l
    t2 = t1 * (u1 - u2)
    t3 = t1 * (w1 - w2)
    t4 = 0.1e1 / 0.2e1
    t5 = t4 * (phi1 + phi2)
    t6 = np.cos(t5)
    t5 = np.sin(t5)
    t7 = t2 + 1
    t8 = t3 * t6 + t5 * t7
    t9 = t1 * (phi1 - phi2)
    t7 = -t3 * t5 + t6 * t7
    t10 = t9**2
    t11 = t6 * GAq * t8
    t12 = t1**2
    t13 = t4 * t1 * (-t5 * t10 * EI * t7 + t11)
    t14 = t6 * t12 * t9 * EI * t7
    t15 = t14 + t13
    t13 = -t14 + t13
    t2 = t1 * EA * (t4 * (t2**2 + t3**2) + t2)
    t3 = t5 * GAq * t8
    t14 = t4 * t1 * (t6 * t10 * EI * t7 + t3)
    t12 = t5 * t12 * t9 * EI * t7
    t16 = -t14 - t12
    t12 = -t14 + t12
    t14 = t4 * t5 * t9
    t17 = t1 * t6
    t18 = t1 * (t17 - t14)
    t11 = t11 * t4 * t1
    t6 = t4 * t6 * t9
    t5 = t1 * t5
    t19 = -t1 * (t6 + t5)
    t3 = t3 * t4 * t1
    t4 = -t7 * t9 / 4
    t20 = t8 * t1
    t21 = -(t8**2) * GAq / 4
    t8 = -l * (t7**2 * t10 * EI + t8**2 * GAq) / 4
    t10 = -t1 * (t17 + t14)
    t1 = t1 * (-t6 + t5)
    kg = np.array([[t2,0,l * t15,-t2,0,l * t13],[0,t2,l * t16,0,-t2,l * t12],[l * (t18 * EI * t7 * t9 + t11),l * (t19 * EI * t7 * t9 - t3),l * (t21 + (t4 - t20) * EI * t7 * t9),l * (-t18 * EI * t7 * t9 - t11),l * (-t19 * EI * t7 * t9 + t3),t8],[-t2,0,-l * t15,t2,0,-l * t13],[0,-t2,-l * t16,0,t2,-l * t12],[l * (t7 * t10 * EI * t9 + t11),l * (t1 * EI * t7 * t9 - t3),t8,l * (-t7 * t10 * EI * t9 - t11),l * (-t1 * EI * t7 * t9 + t3),l * (t21 + (t4 + t20) * EI * t7 * t9)]])

    return fin, k, kg


def B2D_SR(ue, EA, EI, GAq, l):
    """
    2D beam element for large rotations
        - Bernoulli theory
        - 2 nodes

    ue              ... nodal displacements
    EA, EI, GAq,    ... axial, bending and shear stiffness (not needed)
    l               ... element length
    """

    u1, w1, phi1 = ue[0], ue[1], ue[2]
    u2, w2, phi2 = ue[3], ue[4], ue[5]

    fine = np.zeros((6, 1), dtype="float")
    kme = np.zeros((6, 6), dtype="float")
    kge = np.zeros((6, 6), dtype="float")

    # xI = [(0.5 - 1/np.sqrt(3))*l, (0.5 + 1/np.sqrt(3))*l]
    # w  = [l/2, l/2]

    xI = [(1 - np.sqrt(3 / 5)) * l / 2, 0.5 * l, (1 + np.sqrt(3 / 5)) * l / 2]
    wI = [5 / 9 * 0.5 * l, 8 / 9 * 0.5 * l, 5 / 9 * 0.5 * l]

    D = np.diag([EA, EI])

    xx = 0

    for i in range(0, len(wI)):
        x = xI[i]
        w = wI[i]

        # derivative of shape functions
        # first
        dN1dx = 1 / l
        dN2dx = 6 * x * (l - x) / l**3
        dN3dx = -(-2 * l * x + 3 * x**2) / l**2
        dN4dx = -1 / l
        dN5dx = -6 * x * (l - x) / l**3
        dN6dx = -(l - x) * (l - 3 * x) / l**2
        # second
        dN2dxx = (6 * l - 12 * x) / l**3
        dN3dxx = -(-2 * l + 6 * x) / l**2
        dN5dxx = (12 * x - 6 * l) / l**3
        dN6dxx = -(-4 * l + 6 * x) / l**2

        # strains
        dudx = dN1dx * u1 + dN4dx * u2
        dwdx = dN2dx * w1 + dN3dx * phi1 + dN5dx * w2 + dN6dx * phi2
        dwdxx = dN2dxx * w1 + dN3dxx * phi1 + dN5dxx * w2 + dN6dxx * phi2
        E0 = dudx + 0.5 * (dudx**2 + dwdx**2)
        Kb = dwdxx

        # cross section forces
        N = EA * E0
        M = EI * Kb
        CSF = np.array([[N, M]])

        # B-matrix
        B = np.array([[(1+dudx)*dN1dx,  dwdx*dN2dx,   dwdx*dN3dx, (1+dudx)*dN4dx, dwdx*dN5dx, dwdx*dN6dx],
                      [             0,      dN2dxx,       dN3dxx,              0,     dN5dxx,     dN6dxx]])
        
        # internal force vector
        fine += B.T @ CSF.T * w

        # material stiffness matrix
        kme += B.T @ D @ B * w

        # geometric stiffness matrix
        kge += N*np.array([[dN1dx*dN1dx,           0,           0, dN1dx*dN4dx,           0,           0],
                           [          0, dN2dx*dN2dx, dN2dx*dN3dx,           0, dN2dx*dN5dx, dN2dx*dN6dx],
                           [          0, dN3dx*dN2dx, dN3dx*dN3dx,           0, dN3dx*dN5dx, dN3dx*dN6dx],
                           [dN4dx*dN1dx,           0,           0, dN4dx*dN4dx,           0,           0],
                           [          0, dN5dx*dN2dx, dN5dx*dN3dx,           0, dN5dx*dN5dx, dN5dx*dN6dx],
                           [          0, dN6dx*dN2dx, dN6dx*dN3dx,           0, dN6dx*dN5dx, dN6dx*dN6dx]])*w
    
    return fine.flatten(), kme+kge, kge

def assemble(N, E, e_Type, u, EA, EI, GAq):
    # number of DOF
    numDOF = N.shape[0] * 3
    # number of elements
    numE = E.shape[0]

    # intialize arrays
    fin = np.zeros(numDOF)
    K = np.zeros((numDOF, numDOF))
    Kg = np.zeros((numDOF, numDOF))

    # direct stiffness method
    for i in range(0, numE):
        # node indices
        dof_e = np.concatenate((3*(E[i,0]-1) + np.array([0,1,2]), 3*(E[i,1]-1) + np.array([0,1,2]))).astype(int)

        # length
        ind = E[i, :] - 1
        dx = N[ind[0], 0] - N[ind[1], 0]
        dy = N[ind[0], 1] - N[ind[1], 1]
        l = np.sqrt(dx**2 + dy**2)

        # TRANSFORMATION MATRIX
        alpha = np.arctan2(dy, dx)
        T = np.eye(3)
        T[0, 0], T[0, 1] = np.cos(alpha), np.sin(alpha)
        T[1, 0], T[1, 1] = -np.sin(alpha), np.cos(alpha)

        # transform element nodal displacements
        ue = u[dof_e]
        ue[0:3] = T @ ue[0:3]
        ue[3:6] = T @ ue[3:6]

        # element arrays
        if e_Type == "Beam2D_LR":
            fine, ke, kg = B2D_LR(ue, EA, EI, GAq, l)
        else:
            fine, ke, kg = B2D_SR(ue, EA, EI, GAq, l)

        # TRANSFORMATION:
        # internal force vector
        fine[0:3] = T.T @ fine[0:3]
        fine[3:6] = T.T @ fine[3:6]

        # element stiffness matrix
        ke[np.ix_(range(0,3),range(0,3))] = T.T @ ke[np.ix_(range(0,3),range(0,3))] @ T
        ke[np.ix_(range(0,3),range(3,6))] = T.T @ ke[np.ix_(range(0,3),range(3,6))] @ T
        ke[np.ix_(range(3,6),range(0,3))] = T.T @ ke[np.ix_(range(3,6),range(0,3))] @ T
        ke[np.ix_(range(3,6),range(3,6))] = T.T @ ke[np.ix_(range(3,6),range(3,6))] @ T

        # element geometric stiffness matrix
        kg[np.ix_(range(0,3),range(0,3))] = T.T @ kg[np.ix_(range(0,3),range(0,3))] @ T
        kg[np.ix_(range(0,3),range(3,6))] = T.T @ kg[np.ix_(range(0,3),range(3,6))] @ T
        kg[np.ix_(range(3,6),range(0,3))] = T.T @ kg[np.ix_(range(3,6),range(0,3))] @ T
        kg[np.ix_(range(3,6),range(3,6))] = T.T @ kg[np.ix_(range(3,6),range(3,6))] @ T

        # ASSEMBLING
        # internal force vector
        fin[dof_e] += fine

        # global stiffness matrix
        K[np.ix_(dof_e, dof_e)] += ke

        # global geometric stiffness matrix
        Kg[np.ix_(dof_e, dof_e)] += kg

    return fin, K, Kg

class SectionProperty:

    def __init__(self):
        self.A: float | None = None
        self.I: float | None = None
        self.Aq: float | None = None

    @classmethod
    def rectangular(cls, width, height):
        inst = cls()
        inst.A = width * height
        inst.I = width * height**3 / 12
        inst.Aq = 5 / 6 * inst.A
        return inst

class MaterialProperty:

    def __init__(self):
        self.E: float | None = None
        self.nu: float | None = None
        self.G: float | None = None

    @classmethod
    def isotropic(cls, E: float, nu: float) -> Self:
        inst = cls()
        inst.E = E
        inst.nu = nu
        inst.G = E / (2 * (1 + nu))
        return inst

def example(n):
    # Blattfeder
    if n == 1:
        # blattfeder
        L = 5000
        n = 20
        beta = 0 / 180 * np.pi
        N = np.zeros((n + 1, 2))
        N[:, 0] = np.linspace(0, L * np.cos(beta), n + 1)
        N[:, 1] = np.linspace(0, L * np.sin(beta), n + 1)

        E = np.zeros((n, 2), dtype=int)
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1], [1, 2], [1, 3]])

        rect = SectionProperty.rectangular(width = 100, height = 150)
        steel = MaterialProperty.isotropic(E = 210000, nu = 0.3)

        EA = steel.E * rect.A
        EI = steel.E * rect.I
        GAq = steel.G * rect.Aq

        F = np.array([[n + 1, 3, 2 * EI * np.pi / L]])

        monitor_DOF = [n + 1, 1]

    if n == 2:
        # deep arch
        R = 100
        n = 40
        phi0 = 215
        phi = np.linspace(-17.5 / 180 * np.pi, (180 + 17.5) / 180 * np.pi, n + 1)
        N = np.zeros((n + 1, 2))
        N[:, 0] = R * np.cos(phi)
        N[:, 1] = R * np.sin(phi)

        E = np.zeros((n, 2), dtype=int)
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1],
                    [1, 2],
                    [1, 3],
                    [n+1, 1],
                    [n+1, 2]], dtype=int)

        F = np.array([[n / 2 + 1, 2, -10]])

        b = 1
        h = 1
        EA = 210000 * b * h
        EI = 210000 * b * h**3 / 12
        GAq = 5 / 6 * 80760 * b * h

        monitor_DOF = [n / 2 + 1, 2]

    if n == 3:
        # Kragträger
        L = 5000
        n = 20
        beta = 0 / 180 * np.pi
        N = np.zeros((n + 1, 2))
        N[:, 0] = np.linspace(0, L * np.cos(beta), n + 1)
        N[:, 1] = np.linspace(0, L * np.sin(beta), n + 1)

        E = np.zeros((n, 2), dtype="int")
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1],
                    [1, 2],
                    [1, 3]])

        b = 100
        h = 150
        EA = 210000 * b * h
        EI = 210000 * b * h**3 / 12
        GAq = 5 / 6 * 80760 * b * h

        F = np.array([[n + 1, 2, -1e6]])

        monitor_DOF = [n + 1, 1]

    if n == 4:
        # einhüfiger Rahmen
        N = np.zeros((41, 2))
        N[0:21, 1] = np.linspace(0, 120, 21)
        N[20:42, 0] = np.linspace(0, 120, 21)
        N[20:42, 1] = 120

        E = np.zeros((40, 2), dtype="int")
        E[:, 0] = np.linspace(2, 41, 40)
        E[:, 1] = np.linspace(1, 40, 40)

        BC = np.array([[1, 1],
                       [1, 2],
                       [41, 1],
                       [41, 2]])
        
        EA = 7.2E6*6
        EI = 7.2E6*2
        GAq = 5/6*(7.2E6/(2*(1+0.3)))*6

        F = np.array([[23, 2, -10e3]])

        monitor_DOF = [23, 2]

    if n == 5:
        # Euler Stab
        L = 5000
        n = 20
        beta = 0 / 180 * np.pi
        N = np.zeros((n + 1, 2))
        N[:, 0] = np.linspace(0, L * np.cos(beta), n + 1)
        N[:, 1] = np.linspace(0, L * np.sin(beta), n + 1)

        E = np.zeros((n, 2), dtype="int")
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1],
                       [1, 2],
                       [n+1, 2]])

        b = 100
        h = 100
        EA = 210000 * b * h
        EI = 210000 * b * h**3 / 12
        GAq = 5 / 6 * 80760 * b * h

        F = np.array([[n + 1, 1, -100e3]])

        monitor_DOF = [n + 1, 1]

    if n == 6:
        # shallow arch: Länge in m
        R = 70.72
        n = 30
        phi0 = 2 * 16.427
        phi = np.linspace(-16.427 / 180 * np.pi, 16.427 / 180 * np.pi, n + 1)
        N = np.zeros((n + 1, 2))
        N[:, 0] = R * np.sin(phi)
        N[:, 1] = R * np.cos(phi)

        E = np.zeros((n, 2), dtype="int")
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1],
                       [1, 2],
                       [n+1, 1],
                       [n+1, 2]], dtype="int")
        
        b = 0.05
        h = 0.20
        EA = 210000e6 * b * h
        EI = 210000e6 * b * h**3 / 12
        GAq = 5 / 6 * 80760e6 * b * h

        F = np.array([[n / 2 + 1, 2, -10e3]])

        monitor_DOF = [n / 2 + 1, 2]

    if n == 7:
        # shallow arch - radial pressure
        R = 40
        n = 30
        phi0 = 20 / 180 * np.pi
        phi = np.linspace(-phi0, phi0, n + 1)
        N = np.zeros((n + 1, 2))
        N[:, 0] = R * np.sin(phi)
        N[:, 1] = R * np.cos(phi)

        E = np.zeros((n, 2), dtype="int")
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1],
                       [1, 2],
                       [n+1, 1],
                       [n+1, 2]], dtype="int")
        
        b = 0.05
        h = 0.20
        EA = 210000e6 * b * h
        EI = 210000e6 * b * h**3 / 12
        GAq = 5 / 6 * 80760e6 * b * h

        F = np.zeros((2 * (n - 1), 3))
        p0 = 1000
        dphi = 2 * phi0 / n
        e = 0
        for i in range(2, n + 1):
            F[e, :] = [i, 1, -R * dphi * p0 * np.sin(phi[i - 1])]
            F[e + 1, :] = [i, 2, -R * dphi * p0 * np.cos(phi[i - 1])]
            e += 2

        monitor_DOF = [n / 2 + 1, 2]

    # shallow arch - point load
    if n == 8:
        R = 40
        n = 70
        phi0 = 20 / 180 * np.pi
        phi = np.linspace(-phi0, phi0, n + 1)
        N = np.zeros((n + 1, 2))
        N[:, 0] = R * np.sin(phi)
        N[:, 1] = R * np.cos(phi)

        E = np.zeros((n, 2), dtype="int")
        E[:, 0] = np.linspace(2, n + 1, n)
        E[:, 1] = np.linspace(1, n, n)

        BC = np.array([[1, 1],
                       [1, 2],
                       [n+1, 1],
                       [n+1, 2]], dtype="int")
        
        b = 0.05
        h = 0.20
        EA = 210000e6 * b * h
        EI = 210000e6 * b * h**3 / 12
        GAq = 5 / 6 * 80760e6 * b * h

        F = np.array([[n / 2 + 1, 2, -100e3]])

        monitor_DOF = [n / 2 + 1, 2]

    return N, E, BC, F, EA, EI, GAq, monitor_DOF


def plot_BucklingModes(N, u):
    # --- Settings ---
    scale = 1  # deformation magnification
    mode_colors = ["b", "0.75", "0.85"]  # 1st: blue, 2nd: dark gray, 3rd: light gray
    mode_labels = ["1st buckling mode", "2nd buckling mode", "3rd buckling mode"]
    node_marker_sizes = [6, 4, 4]
    node_fill_colors = ["yellow", "lightgray", "lightgray"]  # De-emphasize higher modes

    # --- Undeformed coordinates ---
    x_orig = N[:, 0]
    y_orig = N[:, 1]

    # --- Plot ---
    plt.figure(figsize=(10, 4))

    # Plot undeformed shape
    plt.plot(x_orig, y_orig, "k--", label="Undeformed", linewidth=1)

    # Plot each buckling mode
    for i in range(min(3, u.shape[1])):
        u_mode = u[:, i]
        x_def = x_orig + scale * u_mode[0::3]
        y_def = y_orig + scale * u_mode[1::3]

        # Line for mode shape
        plt.plot(x_def, y_def, color=mode_colors[i], label=mode_labels[i], linewidth=2)

        # Nodes for mode shape
        plt.plot(
            x_def,
            y_def,
            "o",
            markerfacecolor=node_fill_colors[i],
            markeredgecolor=mode_colors[i],  # Match line color
            markersize=node_marker_sizes[i],
        )

    # --- Final layout ---
    # plt.grid(True)
    plt.axis("equal")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("Undeformed vs Buckling Mode Shapes")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_Deformation(N, u):
    # --- Settings ---
    scale = 1  # deformation magnification
    mode_color = "b"
    mode_label = "Deformed"
    node_marker_size = 6
    node_fill_color = "yellow"

    # --- Undeformed coordinates ---
    x_orig = N[:, 0]
    y_orig = N[:, 1]

    # --- Deformed coordinates ---
    x_def = x_orig + scale * u[0::3]
    y_def = y_orig + scale * u[1::3]

    # --- Plot ---
    plt.figure(figsize=(10, 4))

    # Plot undeformed shape
    plt.plot(x_orig, y_orig, "k--", label="Undeformed", linewidth=1)

    # Plot deformed shape
    plt.plot(x_def, y_def, color=mode_color, label=mode_label, linewidth=2)
    plt.plot(
        x_def,
        y_def,
        "o",
        markerfacecolor=node_fill_color,
        markeredgecolor=mode_color,
        markersize=node_marker_size,
    )

    # --- Final layout ---
    plt.axis("equal")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Undeformed vs Deformed Shape")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_monitorDOF(plot_monitor):
    # *** MONITOR WINDOW ***
    # Or just use original order
    x = plot_monitor[:, 1]
    y = plot_monitor[:, 0]

    # Plot blue line through all points
    plt.plot(x, y, color="blue", linewidth=2)

    # Masks for coloring dots
    green_mask = plot_monitor[:, 2] == 0
    red_mask = plot_monitor[:, 2] > 0

    # Plot dots with colors
    plt.plot(plot_monitor[green_mask, 1], plot_monitor[green_mask, 0],
            'o', markerfacecolor='green', markeredgecolor='black', label='stable')

    plt.plot(plot_monitor[red_mask, 1], plot_monitor[red_mask, 0],
            'o', markerfacecolor='red', markeredgecolor='black', label='unstable')

    plt.grid(True)
    plt.ylabel("load factor")
    plt.xlabel("monitor DOF")
    plt.legend()
    plt.show()


# examples:
# 1 ... Dünne Blattfeder
# 2 ... Deep arch
# 3 ... Kragträger
# 4 ... Einhüftiger Rahmen
# 5 ... Euler Stab
# 6 ... Shallow arch
# 7 ... Shallow arch - radial pressure
# 8 ... Shallow arch - point load
N, E, BC, F, EA, EI, GAq, monitor_DOF = example(1)

# *** ANALYSE TYPE ***
# arc-length method (Risk's method)
arc_Length = False
intial_Load_Factor = 0.5
# buckling
lin_Buckling = False
# imperfection
imp = False
scal_BM = 0.0005
# number of load increments
num_Inc = 20
# element type
e_Type = "Beam2D_LR"
# stop incremental loading
inc_loading = False

# apply geometric imperfection
if imp:
    buckling_modes = np.load("bucklingModes.npy")
    N[:, 0] += scal_BM * buckling_modes[0::3, 0]
    N[:, 1] += scal_BM * buckling_modes[1::3, 0]

# *** CHARCTERISTIC VALUES AND PRELOCATE MEMORY ***
# number of nodes
numN = N.shape[0]
# number of DOF
numDOF = 3 * numN
# nodal displacements
u = np.zeros(numDOF)
# out-of-balance vector
g = np.zeros(numDOF)
# external force vector
fex = np.zeros(numDOF, dtype="float")
for i in range(0, F.shape[0]):
    fex[int(3 * (F[i, 0] - 1) + F[i, 1] - 1)] += F[i, 2]

# monitor DOF
dof_monitor = int(3 * (monitor_DOF[0] - 1) + monitor_DOF[1] - 1)
plot_monitor = np.zeros((num_Inc + 1, 3))
count_inc = int(1)

# load factor
load_factor = 0.0

if arc_Length:
    # arc-length method
    F = np.zeros((numDOF, 2))
    F[:, 1] = fex
    DL_fac = 0
    Du_old = np.zeros(numDOF)
else:
    # load control
    fac = np.arange(1 / num_Inc, 1 + 1 / num_Inc, 1 / num_Inc)

# START INCREMENTAL LOADING
for j in range(0, num_Inc):
    # print increment
    print("***")
    print(f"Increment:  {j:2.0f}")

    # START NEWTON ITERATION
    for k in range(0, 20):
        if arc_Length == False:
            load_factor = fac[j]

        # DIRECT STIFFNESS METHOD
        fin, K, Kg = assemble(N, E, e_Type, u, EA, EI, GAq)

        # OUT-OF-BALANCE VECTOR
        g = fin - load_factor * fex

        # APPLY BOUNDARY CONDITIONS
        constrDOF = 3 * (BC[:, 0] - 1) + BC[:, 1] - 1
        for i in constrDOF:
            g[i] = 0
            K[i, :], Kg[i, :] = 0, 0
            K[:, i], Kg[:, i] = 0, 0
            K[i, i], Kg[i, i] = 1, 1

        # determine and print relative error
        err = np.linalg.norm(g) / np.linalg.norm(fex)
        print(f"{k + 1:2d}     {err:.3e}")

        # CONVERGENCE CHECK
        if k > 0:
            if err < 1e-6:
                # stability check
                _, D, _ = ldl(K)
                diag_Elements = np.diag(D)

                # monitor DOF
                plot_monitor[j+1,:] = [load_factor, u[dof_monitor], np.sum(diag_Elements < 1E-6) ]
                count_inc += 1

                # print load factor and number of negative diagonal entries
                print(f"Load factor: {load_factor:4.3f}")
                print(f"Negative diagonal: {plot_monitor[j + 1, 2]:2.0f}")

                # stop iteration
                break

        # SOLVE LINEARIZED SYSTEM
        if arc_Length:
            # ARC-LENGTH METHOD
            # Riks method
            if k == 0:
                # Prediction
                Du = np.linalg.solve(K, fex)

                # sign of loading increment
                if j == 0:
                    # first increment
                    lfac_sign = 1
                    # determine arc length
                    arc_Length = np.linalg.norm(intial_Load_Factor * Du)
                else:
                    lfac_sign = np.sign(Du_old.T @ Du + DL_fac)
                    DL_fac = 0
                    Du_old[:] = 0

                # determine load increment according to arc-length
                DL_fac0 = lfac_sign * arc_Length / np.sqrt(Du.T @ Du + 1)
                Du0 = DL_fac0 * Du

                # update displacements and load factor
                u += Du0
                load_factor += DL_fac0

            else:
                # correction
                # displacements for g and fex
                F[:, 0] = -g
                du = np.linalg.solve(K, F)

                # change of load factor according to constrained (plane - Riks method)
                dl_fac = (-Du0.T @ du[:, 0]) / (Du0.T @ du[:, 1] + DL_fac0)

                # update
                load_factor += dl_fac
                DL_fac += dl_fac
                # displacements
                u_inc = du[:, 0] + dl_fac * du[:, 1]
                Du_old += u_inc
                u += u_inc
        else:
            # LOAD CONTROLLED
            u += np.linalg.solve(K, -g)

    # reach maximum number of iterations
    if k == 19:
        inc_loading = True

    # stop incremental loading
    if inc_loading:
        break

# print computation end status
if inc_loading:
    plot_monitor = plot_monitor[:-2, :]
    print("***\n   Iteration does not converge!  \n***")
else:
    print("***\n   Computation completed sucessfully!  \n***")

# plot monitor DOF
plot_monitorDOF(plot_monitor)

# plot deformation
plot_Deformation(N, u)


# *** LINEARIZED BUCKLING ANALYSIS ***
if lin_Buckling:
    # Solve generalized eigenvalue problem: K u = λ (-Kg) u
    eigvals, eigvecs = eig(K, -Kg)

    # Scale each eigenvector so its max absolute entry is 1
    eigvecs_scaled = np.zeros_like(eigvecs)
    for i in range(eigvecs.shape[1]):
        vec = eigvecs[:, i]
        max_val = np.max(np.abs(vec))
        eigvecs_scaled[:, i] = vec / max_val

    # Take real parts
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs_scaled)

    # Define tolerances
    tol_near_neg1 = 1e-3  # Exclude eigenvalues near -1
    threshold_near_0 = 10  # Keep eigenvalues near 0

    # Filter:
    # - Near zero: |λ| < threshold
    # - Not near -1: |λ + 1| > tol
    near_zero_mask = (np.abs(eigvals) < threshold_near_0) & (
        np.abs(eigvals + 1.0) > tol_near_neg1
    )

    # Apply mask
    near_zero_vals = eigvals[near_zero_mask]
    near_zero_vecs = eigvecs_scaled[:, near_zero_mask]

    # Sort by proximity to zero
    sorted_indices = np.argsort(np.abs(near_zero_vals))
    critical_loads = near_zero_vals[sorted_indices]
    buckling_modes = near_zero_vecs[:, sorted_indices]

    # --- Print ---
    print("LINEARIZED BUCKLING ANALYSIS\n")
    print("Critical loads:")
    for val in critical_loads[:5]:
        print(f"  {val:10.4e}")

    # Get mode shape for smallest positive eigenvalue (≠ 1)
    smallest_val = critical_loads[0]

    x_coords = N[:, 0]
    y_coords = N[:, 1]

    # Bounding box size
    Lx = x_coords.max() - x_coords.min()
    Ly = y_coords.max() - y_coords.min()

    # Characteristic dimension (diagonal of bounding box) - scaling of buckling mode
    L_char = np.sqrt(Lx**2 + Ly**2)
    uMax = np.max(np.abs(buckling_modes[:, 0]))
    scal = L_char / 10 / uMax
    # buckling mode
    u[:] = 0
    u = buckling_modes[:, 0:3] * scal

    # save modes
    np.save("bucklingModes.npy", buckling_modes)

    # plot modes
    plot_BucklingModes(N, u)

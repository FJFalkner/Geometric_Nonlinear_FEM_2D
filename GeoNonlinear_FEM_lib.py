from typing_extensions import Self
from typing import Optional
from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

from ansys.mapdl.core import launch_mapdl

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

    xI = [(1 - np.sqrt(3 / 5)) * l / 2, 0.5 * l, (1 + np.sqrt(3 / 5)) * l / 2]
    wI = [5 / 9 * 0.5 * l, 8 / 9 * 0.5 * l, 5 / 9 * 0.5 * l]

    D = np.diag([EA, EI])
    
    for x, w in zip(xI, wI):

        # derivative of shape functions
        # first
        dN1dx =  1 / l
        dN2dx = -6*x**2/l**3 + 6*x/l**2
        dN3dx = -3*x**2/l**2 + 2*x/l
        dN4dx = -1 / l
        dN5dx =  6*x**2/l**3 - 6*x/l**2
        dN6dx = -3*x**2/l**2 + 4*x/l - 1
        # second
        dN2dxx = -12*x/l**3 + 6/l**2
        dN3dxx = -6*x/l**2 + 2/l
        dN5dxx =  12*x/l**3 - 6/l**2
        dN6dxx = -6*x/l**2 + 4/l

        # strains
        dudx =  dN1dx * u1 + dN4dx * u2
        dwdx =  dN2dx * w1 + dN3dx * phi1 + dN5dx * w2 + dN6dx * phi2
        dwdxx = dN2dxx * w1 + dN3dxx * phi1 + dN5dxx * w2 + dN6dxx * phi2
        E0 = dudx  + 0.5 * dwdx**2
        Kb = -dwdxx

        # cross section forces
        N = EA * E0
        M = EI * Kb
        CSF = np.array([[N, M]])

        # B-matrix
        B = np.array([[         dN1dx,  dwdx*dN2dx,   dwdx*dN3dx,          dN4dx, dwdx*dN5dx, dwdx*dN6dx],
                      [             0,      -dN2dxx,       -dN3dxx,              0,     -dN5dxx,     -dN6dxx]])
        
        # internal force vector
        fine += B.T @ CSF.T * w

        # material stiffness matrix
        kme += B.T @ D @ B * w

        # geometric stiffness matrix
        kge += N*np.array([[          0,           0,           0,           0,           0,           0],
                           [          0, dN2dx*dN2dx, dN2dx*dN3dx,           0, dN2dx*dN5dx, dN2dx*dN6dx],
                           [          0, dN3dx*dN2dx, dN3dx*dN3dx,           0, dN3dx*dN5dx, dN3dx*dN6dx],
                           [          0,           0,           0,           0,           0,           0],
                           [          0, dN5dx*dN2dx, dN5dx*dN3dx,           0, dN5dx*dN5dx, dN5dx*dN6dx],
                           [          0, dN6dx*dN2dx, dN6dx*dN3dx,           0, dN6dx*dN5dx, dN6dx*dN6dx]])*w
    
    return fine.flatten(), kme+kge, kge

def B2D_SR_ML(ue, EA, EI, GAq, l):
    """
    2D beam element for moderate displacements and small rotations rotations
        - Bernoulli theory
        - 2 nodes
        - shallow arch assumption
        - membrane locking avoided by mean strain

    ue              ... nodal displacements
    EA, EI, GAq,    ... axial, bending and shear stiffness (not needed)
    l               ... element length
    """

    # nodal displacements
    u1, w1, phi1 = ue[0], ue[1], ue[2]  # right node
    u2, w2, phi2 = ue[3], ue[4], ue[5]  # left node

    fine = np.zeros(6, dtype="float")       # internal force vector
    ke   = np.zeros((6, 6), dtype="float")  # element stiffness matrix
    kge  = np.zeros((6, 6), dtype="float")  # geometric striffness matrix

    # internal force vector
    t1 = phi1 ** 2
    t4 = phi2 ** 2
    t7 = l ** 2
    t9 = w1 - w2
    t16 = t9 ** 2
    t19 = (t7 * (-phi1 * phi2 + 2 * t1 + 2 * t4) + l * (3 * phi1 * t9 + 3 * phi2 * t9 + 30 * u1 - 30 * u2) + 18 * t16) * EA
    t20 = 1 / t7
    t22 = t20 * t19 / 30
    t24 = 1 / t7 / l
    t33 = 6 * phi1 * t20 + 6 * phi2 * t20 + 12 * w1 * t24 - 12 * w2 * t24
    t35 = 4 * t33 * EI
    t36 = EI * t20
    t42 = 1 / l
    t49 = -2 * phi1 * t42 - 4 * phi2 * t42 - 6 * w1 * t20 + 6 * w2 * t20
    t52 = 12 * t49 * EI * t24 - 6 * t33 * t36
    t55 = phi1 + phi2
    t57 = 36 * w1
    t58 = 36 * w2
    t63 = EI * t42
    t65 = 6 * t49 * t63
    t69 = 2 * t33 * EI * l
    t70 = t33 * t63
    t73 = 6 * t49 * t36
    t80 = 3 * l * t9
    t85 = t49 * EI
    fine[0] = t22
    fine[1] = t35 + t7 * t52 / 2 + t19 * t24 * (3 * l * t55 + t57 - t58) / 900 - t65
    fine[2] = t69 + t7 * (-2 * t70 + t73) / 2 + t19 * t24 * (t7 * (4 * phi1 - phi2) + t80) / 900 - 2 * t85
    fine[3] = -t22
    fine[4] = -t35 - t7 * t52 / 2 + t19 * t24 * (-3 * l * t55 - t57 + t58) / 900 + t65
    fine[5] = t69 + t7 * (-4 * t70 + t73) / 2 + t19 * t24 * (t7 * (-phi1 + 4 * phi2) + t80) / 900 - 4 * t85

    # stiffness matrix
    t1 = phi1 + phi2
    t2 = -w1 + w2
    t3 = 4 * phi1
    t4 = 3 * t2
    t5 = (-phi2 + t3) * l - t4
    t6 = 1 / l
    t7 = t6 ** 2
    t8 = t6 * t7
    t9 = (l * t1 / 10 - 0.6e1 / 0.5e1 * t2) * EA * t7
    t10 = EA * t6
    t11 = t10 / 30
    t12 = t11 * t5
    t4 = t11 * ((phi1 - 4 * phi2) * l + t4)
    t13 = phi1 ** 2
    t14 = phi2 ** 2 + t13
    t15 = 9
    t16 = 2 * phi1
    t17 = -t16 * phi2 + t14 * t15
    t1 = t2 * t1
    t18 = u2 - u1
    t19 = -3 * t1 - 10 * t18
    t20 = t2 ** 2
    t21 = 216 * t20
    t22 = 1200 * EI
    t23 = (phi2 + t16) * phi2 + 6 * t13
    t24 = 5 * t18
    t25 = t2 * (-phi1 * t15 + phi2) - t24
    t26 = 54 * t20
    t27 = 1800 * EI
    t28 = 6 * phi2
    t16 = (t28 + t16) * phi2 + t13
    t15 = t2 * (-phi2 * t15 + phi1) - t24
    t29 = 0.1e1 / 0.100e3
    t30 = 0.1e1 / 0.300e3
    t31 = t29 * (((l * t17 + 12 * t19) * l + t21) * EA + t22) * t8
    t32 = t30 * (((l * t23 + 6 * t25) * l + t26) * EA + t27) * t7
    t8 = t29 * (((-l * t17 - 12 * t19) * l - t21) * EA - t22) * t8
    t17 = t30 * (((l * t16 + 6 * t15) * l + t26) * EA + t27) * t7
    t19 = 8
    t21 = 3 * phi2
    t18 = -20 * t18
    t29 = 27 * t20
    t1 = t30 * ((2 * l * ((t21 * phi1 - t14) * l - t1 + t24) - 3 * t20) * EA + 600 * EI) * t6
    t5 = -t11 * t5
    t11 = t30 * (((-l * t23 - 6 * t25) * l - t26) * EA - t27) * t7
    t7 = t30 * (((-l * t16 - 6 * t15) * l - t26) * EA - t27) * t7
    ke = np.array([[t10,t9,t12,-t10,-t9,-t4],[t9,t31,t32,-t9,t8,t17],[t12,t32,t30 * (((((-t3 + t21) * phi2 + t13 * t19) * l - 2 * t2 * (6 * phi1 + phi2) + 2 * t18) * l + t29) * EA + t22) * t6,t5,t11,t1],[-t10,-t9,t5,t10,t9,t4],[-t9,t8,t11,t9,t31,t7],[-t4,t17,t1,t4,t7,t30 * (((((phi2 * t19 - t3) * phi2 + 3 * t13) * l - 2 * t2 * (phi1 + t28) + 2 * t18) * l + t29) * EA + t22) * t6]])

    # geometric stiffness matrix
    kge = np.array([[0,0,0,0,0,0],[0,0.2e1 / 0.25e2 * EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 3,EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150,0,-0.2e1 / 0.25e2 * EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 3,EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150],[0,EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150,0.2e1 / 0.225e3 * EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l,0,-EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150,-EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l / 450],[0,0,0,0,0,0],[0,-0.2e1 / 0.25e2 * EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 3,-EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150,0,0.2e1 / 0.25e2 * EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 3,-EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150],[0,EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150,-EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l / 450,0,-EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l ** 2 / 150,0.2e1 / 0.225e3 * EA * ((phi1 ** 2 - phi1 * phi2 / 2 + phi2 ** 2) * l ** 2 + ((0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi1 + (0.3e1 / 0.2e1 * w1 - 0.3e1 / 0.2e1 * w2) * phi2 + 15 * u1 - 15 * u2) * l + 9 * (w1 - w2) ** 2) / l]])

    return fine, ke, kge


# Section stiffness   
class SectionProperty:

    def __init__(self, b: float, h: float, E: float = 2.1E11, nu: float = 0.3, rho: float = 7850):

        # initial material steel in kg, N, m
        self.EI   = E*b*h**3/12
        self.EA   = E*b*h
        self.GAq  = 5/6*b*h*E/(2*(1+nu))
        self.rhoA = rho*b*h

# Geometric nonlinear examples
class GNLexamples:

    def __init__(self):

        self.N: float
        self.E: float
        self.BC: float
        self.F: float
        self.monDOF: float
        self.elType: Callable
        self.sec = None 

    @classmethod
    def leafSpring(cls, b: float, h: float, M: float, l: float = 0.5, n: int = 10, elType: Callable = B2D_SR_ML, E: float = 2.1E11, nu: float = 0.3):
        """
        Construct a simple 2D leaf-spring beam problem.
        b:      width
        h:      thickness
        M:      applied end moment
        l:      total length (default 0.5 m)
        n:      number of elements (default 10)
        elType: element type (default B2D_SR_ML)
        E:      Younds modulus
        nu:     Poisson ration
        """
        
        inst = cls()
        # nodes
        inst.N = np.zeros((n + 1, 2))
        inst.N[:, 0] = np.linspace(0, l, n + 1)
        # elements
        inst.E = np.zeros((n, 2), dtype=int)
        inst.E[:, 0] = np.linspace(2, n + 1, n)
        inst.E[:, 1] = np.linspace(1, n, n)
        # boundary conditions
        inst.BC = np.array([[1, 1], [1, 2], [1, 3]])
        # external force
        inst.F = np.array([[n + 1, 3, M]])
        # monitor DOF
        inst.monDOF = [n + 1, 1]
        # section properties
        inst.sec = SectionProperty(b, h, E = E, nu = nu)
        # element type
        inst.elType = elType

        return inst
    
    @classmethod
    def shallowArch(cls, loadCase: str = "pressure", R: float = 40, alpha: float = 20, F: float = 1E6, b: float = 0.01, h: float = 0.2, n: int = 40, elType: Callable = B2D_SR_ML):
        """
        Construct a shallow arch under a point load at the center and pinned supports at the ends.
        loadCase: "pressure" (default) or "force"
        R:      radius of the arch
        alpha:  half of the opening angle in rad
        F:      magnitude of the loading
        b:      width of the cross section
        h:      height of the cross section
        n:      number of elements used
        elType: element type
        """
        inst = cls()
        # nodes
        phi0 = alpha
        phi = np.linspace(-phi0, phi0, n + 1)
        inst.N = np.zeros((n + 1, 2))
        inst.N[:, 0] = R * np.sin(phi)
        inst.N[:, 1] = R * np.cos(phi)
        # elements
        inst.E = np.zeros((n, 2), dtype="int")
        inst.E[:, 0] = np.linspace(2, n + 1, n)
        inst.E[:, 1] = np.linspace(1, n, n)
        # boundary conditions
        inst.BC = np.array([[1, 1],
                            [1, 2],
                            [n+1, 1],
                            [n+1, 2]], dtype="int")
        # force
        if loadCase == "force":
            inst.F = np.array([[n / 2 + 1, 2, -F]])
        # pressure
        elif loadCase == "pressure":
            inst.F = np.zeros((2*(n-1),3), dtype = float)
            dp = F*R*2*phi0 / n
            e = 0
            for i, alpha in enumerate(phi[1:-1], start=2):
                inst.F[e,:]   = [i, 1,  dp*np.sin(alpha)]
                inst.F[e+1,:] = [i, 2, -dp*np.cos(alpha)]
                e += 2
        else:
            raise ValueError("Unknown loadCase. Use 'pressure' or 'force'.")
        
        inst.monDOF = [n / 2 + 1, 2]
        # section
        inst.sec = SectionProperty(b, h)
        # element type
        inst.elType = elType

        return inst
    
    @classmethod
    def deepArch(cls, R: float = 100, phi0: float = 215, F: float = 1E6, b: float = 0.01, h: float = 0.2, n: int = 40, elType: Callable = B2D_SR):
        """
        Construct a deep arch under a point load at the center and pinned supports at the ends.
        R:      radius of the arch
        alpha:  half of the opening angle in degrees
        F:      magnitude of the vertical force
        b:      width of the cross section
        h:      height of the cross section
        n:      number of elements used
        """
        inst = cls()
        # deep arch
        xx = phi0/2
        phi = np.linspace(-xx / 180 * np.pi, xx / 180 * np.pi, n + 1)
        inst.N = np.zeros((n + 1, 2))
        inst.N[:, 0] = R * np.sin(phi)
        inst.N[:, 1] = R * np.cos(phi)

        inst.E = np.zeros((n, 2), dtype=int)
        inst.E[:, 0] = np.linspace(2, n + 1, n)
        inst.E[:, 1] = np.linspace(1, n, n)

        inst.BC = np.array([[1, 1],
                            [1, 2],
                            [n+1, 1],
                            [n+1, 2],
                            [n+1, 3]], dtype=int)

        inst.F = np.array([[n / 2 + 1, 2, -F]])

        # section
        inst.sec = SectionProperty(b, h)
        # monitor DOF
        inst.monDOF = [n / 2 + 1, 2]
        # element type
        inst.elType = elType

        return inst
    
    def plotMesh(self):
        """
        Plot mesh of finite element discretization with custom background and axes.
        The coordinate system origin is always at the bottom left of the plot window.
        """
        fig, ax = plt.subplots()
        # Set light gray background
        ax.set_facecolor('#f0f0f0')

        # Plot mesh nodes and elements
        ax.plot(self.N[:, 0], self.N[:, 1], 'k-o', markerfacecolor='yellow', markeredgecolor='k', markersize=5)
        indBC = (self.BC[:, 0] - 1).astype(int)
        ax.plot(self.N[indBC, 0], self.N[indBC, 1], 'o', markerfacecolor='red', markeredgecolor='k', markersize=6)
        indF = (self.F[:, 0] - 1).astype(int)
        ax.plot(self.N[indF, 0], self.N[indF, 1], 'o', markerfacecolor='green', markeredgecolor='k', markersize=6)

        # Remove axes and grid
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Find plot limits
        x_min, x_max = np.min(self.N[:, 0]), np.max(self.N[:, 0])
        z_min, z_max = np.min(self.N[:, 1]), np.max(self.N[:, 1])
        dx = (x_max - x_min) * 0.1
        dz = (z_max - z_min) * 0.1

        # If dx or dz is close to zero, use the other dimension
        if np.isclose(dx, 0):
            dx = dz if not np.isclose(dz, 0) else 0.5
        if np.isclose(dz, 0):
            dz = dx if not np.isclose(dx, 0) else 0.5

        # Set plot limits with margin
        ax.set_xlim(x_min - dx, x_max + dx)
        # Avoid identical y limits (which causes matplotlib warning)
        if z_max - z_min == 0:
            ax.set_ylim(z_min - 0.5, z_max + 0.5)
        else:
            ax.set_ylim(z_min - dz, z_max + dz)

        # Coordinate system origin at bottom left of plot window
        base_x = x_min - dx
        base_z = z_min - dz

        # Arrow length
        arr_len = max(dx, dz) * 0.7 if max(dx, dz) > 0 else 0.5

        # Draw x-axis (red, horizontal)
        ax.arrow(base_x, base_z, arr_len, 0, head_width=arr_len*0.15, head_length=arr_len*0.15,
                 fc='red', ec='red', length_includes_head=True)
        ax.text(base_x + arr_len * 1.1, base_z, 'x', color='red', fontsize=12, va='center', ha='left')

        # Draw z-axis (blue, vertical)
        ax.arrow(base_x, base_z, 0, arr_len, head_width=arr_len*0.15, head_length=arr_len*0.15,
                 fc='blue', ec='blue', length_includes_head=True)
        ax.text(base_x, base_z + arr_len * 1.1, 'z', color='blue', fontsize=12, va='bottom', ha='center')

        plt.show()

    def imperfection(self, imp: np.ndarray, scale: float):

        # modify x and y coordinates
        self.N[:,0] += scale*imp[0::3]
        self.N[:,1] += scale*imp[1::3]

    def plotDisplacement(self, U, scale: float = 1.0, labels=None, title: str = None):
        """
        Plot deformed structure for one or several displacement vectors.
        If multiple displacement states are given, highlight the node
        with maximum displacement difference.

        Parameters
        ----------
        U : np.ndarray or tuple/list of np.ndarray
            Displacement vectors (length 3*nnodes each).
        scale : float
            Scaling factor for displacements.
        labels : list of str, optional
            Legend entries for each displacement vector.
            If None, defaults to "u[0]", "u[1]", ...
        title : str, optional
            Plot title. If None, no title is shown.
        """

        fig, ax = plt.subplots()
        ax.set_facecolor('#f0f0f0')

        # Normalize input to tuple
        if isinstance(U, np.ndarray):
            U = (U,)
        elif isinstance(U, list):
            U = tuple(U)

        n_nodes = self.N.shape[0]

        # Undeformed mesh
        if len(U) > 1:
            ax.plot(self.N[:, 0], self.N[:, 1], 'k--', label="undeformed")
        else:
            ax.plot(self.N[:, 0], self.N[:, 1], 'k--')

        # Colors
        colors = ['yellow', 'skyblue', 'cyan', 'orange', 'lime', 'red', 'violet']

        # Handle legend labels only if multiple displacement vectors
        if len(U) > 1:
            if labels is None:
                labels = [f"u[{i}]" for i in range(len(U))]
            elif len(labels) != len(U):
                raise ValueError("Length of labels must match number of displacement vectors in U.")
        else:
            labels = [None]

        # Store deformed positions
        deformed_positions = []

        handles = []
        for i, (u, lbl) in enumerate(zip(U, labels)):
            x_def = self.N[:, 0] + scale * u[0::3]
            y_def = self.N[:, 1] + scale * u[1::3]
            deformed_positions.append(np.column_stack([x_def, y_def]))

            fc = colors[i % len(colors)]
            if lbl is not None:
                h, = ax.plot(
                    x_def, y_def, 'k-o',
                    markerfacecolor=fc,
                    markeredgecolor='k',
                    markersize=5,
                    label=lbl
                )
            else:
                h, = ax.plot(
                    x_def, y_def, 'k-o',
                    markerfacecolor=fc,
                    markeredgecolor='k',
                    markersize=5
                )
            handles.append(h)

        # --- Highlight maximum displacement difference ---
        if len(deformed_positions) > 1:
            # Compare first vs. last state
            diff = np.linalg.norm(deformed_positions[-1] - deformed_positions[0], axis=1)
            i_max = np.argmax(diff)

            p1 = deformed_positions[0][i_max]
            p2 = deformed_positions[-1][i_max]

            # Draw dotted line
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b:', linewidth=2)

            # Annotate difference value
            dval = diff[i_max]
            ax.text(p2[0], p2[1], f"{dval:.4e}", color='b', fontsize=9,
                    ha='left', va='bottom')

        # Remove axes and grid
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Find plot limits including scaled displacement
        x_all = np.concatenate([self.N[:, 0]] + [self.N[:, 0] + scale * u[0::3] for u in U])
        z_all = np.concatenate([self.N[:, 1]] + [self.N[:, 1] + scale * u[1::3] for u in U])
        x_min, x_max = np.min(x_all), np.max(x_all)
        z_min, z_max = np.min(z_all), np.max(z_all)
        dx = (x_max - x_min) * 0.1
        dz = (z_max - z_min) * 0.1

        # Set plot limits with margin
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(z_min - dz, z_max + dz)

        # Coordinate system origin at bottom left of plot window
        base_x = x_min - dx
        base_z = z_min - dz

        # Arrow length
        arr_len = max(dx, dz) * 0.7 if max(dx, dz) > 0 else 0.5

        # Draw x-axis (red, horizontal)
        ax.arrow(base_x, base_z, arr_len, 0, head_width=arr_len*0.15, head_length=arr_len*0.15,
                 fc='red', ec='red', length_includes_head=True)
        ax.text(base_x + arr_len * 1.1, base_z, 'x', color='red', fontsize=12, va='center', ha='left')

        # Draw z-axis (blue, vertical)
        ax.arrow(base_x, base_z, 0, arr_len, head_width=arr_len*0.15, head_length=arr_len*0.15,
                 fc='blue', ec='blue', length_includes_head=True)
        ax.text(base_x, base_z + arr_len * 1.1, 'z', color='blue', fontsize=12, va='bottom', ha='center')

        # Only show legend if multiple displacement vectors
        if len(U) > 1:
            ax.legend()

        # Plot title if provided
        if title is not None:
            ax.set_title(title)

        plt.show()

        return tuple(handles)
        
# Finite element solvers
class FEMsolve:

    def __init__(self):
        self.numDOF: int = 0
        self.u: Optional[np.ndarray] = None
        self.monVal: Optional[np.ndarray] = None

    @classmethod
    def LoadCon(cls, mesh, numInc: int, maxIter: int = 15, maxErr: float = 1E-6, pOut: bool = False):
        """
        Load-controlled nonlinear finite element analysis
            - full newton method
            - constant incremental loading 

        mesh:    finite element discretization
        numInc:  number of load increments
        maxIter: maximum number of iterations (default: 15)
        maxErr:  maximum relative error in equilibrium (default: 1E-6)
        pOut:    print output to console (default: False)
        """

        inst = cls()

        # monitor dof
        dofM = int(3*(mesh.monDOF[0] - 1) + mesh.monDOF[1] -1)
        inst.monVal = np.array([[0, 0, 0]])

        # degrees of freedom
        inst.numDOF = 3*mesh.N.shape[0]

        # nodal displacements
        inst.u = np.zeros(inst.numDOF)

        # out-of-balance vector
        g = np.zeros(inst.numDOF)

        # external force vector
        fex = np.zeros(inst.numDOF, dtype="float")
        for i in range(0, mesh.F.shape[0]):
            fex[int(3 * (mesh.F[i, 0] - 1) + mesh.F[i, 1] - 1)] += mesh.F[i, 2]

        # constrained dof
        constrDOF = np.array([3 * (bc[0] - 1) + bc[1] - 1 for bc in mesh.BC])

        # dicrete load steps
        LambdaD = np.arange(1/numInc, 1+1/numInc, 1/numInc)

        # *** START INCREMENTAL LOADING ***
        for Lambda in LambdaD:
            # print load step
            if pOut:
                print(f"Lambda = {Lambda:5.4f}")

            # START NEWTON ITERATION
            for iter in range(0,maxIter):

                # global arrays
                fin, K, _ = inst.assemble(mesh, inst.u, mesh.elType)

                # out-of-balance vector
                g = fin - Lambda*fex

                # apply boundary conditions
                inst.applyBD(K, g, constrDOF = constrDOF)

                # convergence check
                err = np.linalg.norm(g)/(Lambda*np.linalg.norm(fex))
                if pOut and iter > 0:
                    print(f"{iter+1:3d}    {err:12.4e}")

                # solution converged
                if err < maxErr:
                    # stability check
                    eigVal = np.linalg.eigvals(K)
                    numNegEig = np.sum(eigVal<1E-8)
                    
                    # get monitor dof
                    inst.monVal = np.append(inst.monVal, np.array([[Lambda, inst.u[dofM], numNegEig]]), axis = 0)

                    # print stability
                    if pOut:
                        print(f"-------------------\nNeg. EigVal = {numNegEig:2d}\n*******************")

                    # stop equilibrium iteration
                    break

                # maximum number of iterations reached
                if iter == maxIter-1:

                    # error message
                    print(f"*******************\n ITERATION FAILED!\n*******************")

                    return inst

                # solve and update displacements
                inst.u += np.linalg.solve(K, -g)

        return inst
    
    @classmethod
    def arcL(cls, mesh, numInc: int, maxIter: int = 15, Lambda0: float = 0.1, maxErr: float = 1E-6, pOut: bool = False):
        """
        Arc-Length Method with classical Riks constrain surface
        mesh:    finite element discretization
        numInc:  number of increments
        maxIter: maximal number of interations (default = 15) 
        Lambda0: inital load increment for arc-length (default = 0.1)
        maxErr:  maximal relative error (default = 1E-6)
        pOut:    print output to console (default: False)
        """

        inst = cls()

        # monitor dof
        dofM = int(3*(mesh.monDOF[0] - 1) + mesh.monDOF[1] - 1)
        inst.monVal = np.array([[0, 0, 0]])

        # degrees of freedom
        inst.numDOF = 3*mesh.N.shape[0]

        # nodal displacements
        inst.u = np.zeros(inst.numDOF)
        # displacement increment in current load increment
        DuC = np.zeros(inst.numDOF)
        # displacement increment in previous load increment
        DuO = np.zeros(inst.numDOF) 
        du = np.zeros((inst.numDOF,2))  

        # out-of-balance vector and external force vector
        F = np.zeros((inst.numDOF,2))
        for i in range(0, mesh.F.shape[0]):
            F[int(3 * (mesh.F[i, 0] - 1) + mesh.F[i, 1] - 1),1] += mesh.F[i, 2]

        # constrained dof
        constrDOF = np.array([3 * (bc[0] - 1) + bc[1] - 1 for bc in mesh.BC])

        # load factor
        Lambda = 0
        DLambda = 0

        # *** START INCREMENTAL LOADING ***
        for i in range(0,numInc):

            # print load step
            if pOut:
                print(f"Increment = {i:3d}")

            # START NEWTON ITERATION
            for iter in range(0,maxIter):

                # global arrays
                fin, K, _ = inst.assemble(mesh, inst.u, mesh.elType)

                # out-of-balance vector
                F[:,0] = -(fin - Lambda*F[:,1])

                # apply boundary conditions
                inst.applyBD(K, F, constrDOF = constrDOF)

                # convergence check
                if iter > 0:
                    err = np.linalg.norm(F[:,0])/(np.linalg.norm(F[:,1]))
                    if pOut:
                        print(f"{iter+1:3d}    {err:12.4e}")

                    # solution converged
                    if err < maxErr:
                        # stability check
                        eigVal = np.linalg.eigvals(K)
                        numNegEig = np.sum(eigVal<1E-8)
                        
                        # get monitor dof
                        inst.monVal = np.append(inst.monVal, np.array([[Lambda, inst.u[dofM], numNegEig]]), axis = 0)

                        # print stability
                        if pOut:
                            print(f"-------------------\nNegEig = {numNegEig:1d}")
                            print(f"Lambda = {Lambda:5.4f}")
                            print(f"*******************")

                        #
                        DuO = DuC

                        # stop equilibrium iteration
                        break

                # maximum number of iterations reached
                if iter == maxIter-1:

                    # error message
                    print(f"*******************\n ITERATION FAILED!\n*******************")

                    return inst

                # ***  ARC-LENGTH ***
                # PREDICTOR
                if iter == 0:
                    # displacement increment
                    DuC = np.linalg.solve(K,F[:,1])

                    # first load increment
                    if i == 0:
                        # sign of increment load factor
                        DLambdaSign = 1
                        # determine arc-length
                        arcL0 = np.linalg.norm(Lambda0 * DuC)
                    # general load increment
                    else:
                        # sign of increment load factor
                        DLambdaSign = np.sign(DuO.T @ DuC + DLambda)

                        DuO[:] = 0
                        DLamabda = 0

                    # load increment according to arc-length
                    DLambda0 = DLambdaSign * arcL0 / np.sqrt(DuC.T @ DuC + 1)
                    Du0 = DLambda0 * DuC
                    DuC = Du0

                    # update displacements and load factor
                    inst.u += Du0
                    Lambda += DLambda0
                    DLambda = DLambda0

                # CORRECTOR
                else:
        
                    # displacements for g and fex
                    U = np.linalg.solve(K, F)
                
                    # interative change of load factor
                    dLambda = (-Du0.T @ U[:,0]) /(Du0.T @ U[:,1] + DLambda0)

                    # update
                    Lambda += dLambda
                    DLambda += dLambda
                
                    du = U[:,0] + dLambda*U[:,1]
                    DuC += du
                    inst.u += du

        return inst

    @classmethod
    def linBuckling(cls, mesh, u:np.ndarray, numModes: int = 3,  pOut: bool = False):
        """
        Linearized Buckling Analysis
        mesh:     finite element discretization
        u:        displacements at reference point
        numModes: number of modes (default = 3)
        """
        inst = cls()

        # degrees of freedom
        inst.numDOF = 3*mesh.N.shape[0]

        # constrained dof
        constrDOF = np.array([3 * (bc[0] - 1) + bc[1] - 1 for bc in mesh.BC])
         # external force vector
        fex = np.zeros(inst.numDOF, dtype="float")
        for i in range(0, mesh.F.shape[0]):
            fex[int(3 * (mesh.F[i, 0] - 1) + mesh.F[i, 1] - 1)] += mesh.F[i, 2]

        # assemble stiffness matrix at reference
        _, K, _ = inst.assemble(mesh, u, mesh.elType)

        # apply boundary conditions
        inst.applyBD(K, fex, constrDOF = constrDOF)

        # determine tangential displacement
        ut = np.linalg.solve(K,fex)

        # assemble stiffness matrix at reference
        _, _, Kg = inst.assemble(mesh, u, mesh.elType)

        # apply boundary conditions
        inst.applyBD(Kg, constrDOF = constrDOF)

        # solve generalized eigenvalue problem
        eigVals, eigVec = eig(K, -Kg)

        # remove eigenvalues close to -1 (boundary conditions)
        mask = np.abs(eigVals.real + 1) > 1E-6
        eigValsMod = eigVals[mask]
        eigVecMod = eigVec[:,mask]

        # sort eigenvalues
        ind = np.argsort(np.abs(eigValsMod))[:numModes]
        inst.LambdaCrit = eigValsMod[ind].real + 1
        inst.buckModes = eigVecMod[:,ind]

        # scaling of buckling modes
        max_per_col = np.max(np.abs(inst.buckModes), axis=0)  
        inst.buckModes = inst.buckModes / max_per_col

        # print
        if pOut:
            print(f"LINEARIZED BUCKLING ANALYSIS\nCritical load factors:")
            for val in inst.LambdaCrit:
                print(f"{val:6.2f}")

        return inst

    
    def assemble(self, mesh: GNLexamples, u:  np.ndarray, elem_func: Callable[[np.ndarray, float, float, float, float], tuple]):

        # number of DOF
        numDOF = mesh.N.shape[0] * 3

        # intialize arrays
        fin = np.zeros(numDOF)
        K   = np.zeros((numDOF, numDOF))
        Kg  = np.zeros((numDOF, numDOF))

        for x in mesh.E:

            # indices of nodes (python)
            indN = x - 1

            # indices of element nodal displacements
            dofE = np.concatenate([3*indN[0]+ np.array([0,1,2]), 3*indN[1] + np.array([0,1,2])])

            # element nodal displacements
            ue = u[dofE]

            # length of element
            dx = mesh.N[indN[0], 0] - mesh.N[indN[1], 0]
            dy = mesh.N[indN[0], 1] - mesh.N[indN[1], 1]
            l = np.sqrt(dx**2 + dy**2)

            # TRANSFORMATION MATRIX
            alpha = np.arctan2(dy, dx)
            T = np.eye(3)
            T[0, 0], T[0, 1] =  np.cos(alpha), np.sin(alpha)
            T[1, 0], T[1, 1] = -np.sin(alpha), np.cos(alpha)

            # transform element nodal displacements
            ue[0:3] = T @ ue[0:3]
            ue[3:6] = T @ ue[3:6]

            # element arrays
            fine, ke, kge = elem_func(ue, mesh.sec.EA, mesh.sec.EI, mesh.sec.GAq, l)

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
            kge[np.ix_(range(0,3),range(0,3))] = T.T @ kge[np.ix_(range(0,3),range(0,3))] @ T
            kge[np.ix_(range(0,3),range(3,6))] = T.T @ kge[np.ix_(range(0,3),range(3,6))] @ T
            kge[np.ix_(range(3,6),range(0,3))] = T.T @ kge[np.ix_(range(3,6),range(0,3))] @ T
            kge[np.ix_(range(3,6),range(3,6))] = T.T @ kge[np.ix_(range(3,6),range(3,6))] @ T

            # ASSEMBLING
            # internal force vector
            fin[dofE] += fine

            # global stiffness matrix
            K[np.ix_(dofE, dofE)] += ke

            # global geometric stiffness matrix
            Kg[np.ix_(dofE, dofE)] += kge

        return fin, K, Kg

    @staticmethod
    def applyBD(*arrays: np.ndarray, constrDOF: np.ndarray):
        """
        Apply boundary conditions to one or more arrays.
        
        Parameters
        ----------
        *arrays : np.ndarray
            One or more arrays (matrices/vectors) to modify in-place.
        constrDOF : np.ndarray
            List/array of constrained DOFs.
        """
        for A in arrays:
            if A.ndim == 1:  # vector
                A[constrDOF] = 0

            elif A.ndim == 2:  # matrix
                if A.shape[0] == A.shape[1]:  # square
                    for dof in constrDOF:
                        A[:, dof] = 0
                        A[dof, :] = 0
                        A[dof, dof] = 1
                else:  # non-square
                    for dof in constrDOF:
                        A[dof, :] = 0

    def plotDisplacement(self, mesh: GNLexamples, u: np.ndarray, scale: float = 1.0):
        """
        Plot deformed structure
        mesh:  finite element discretization
        u:     nodal displacements
        scale: scaling factor of displacements (default = 1.0)
        """
        plt.plot(mesh.N [:,0], mesh.N[:, 1], 'k--')
        plt.plot(mesh.N [:,0] + scale*u[0::3], mesh.N[:, 1] + scale*u[1::3], 'k-o', markerfacecolor = 'yellow', markeredgecolor='k', markersize = 5)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(which = 'major', linestyle = ':')
        plt.show()

    def plotMonitor(self):
        """
        plot load-displacement curve of monitor DOF
        """
        # *** MONITOR WINDOW ***
        # Or just use original order
        x = self.monVal[:, 1]
        y = self.monVal[:, 0]

        # Plot blue line through all points
        plt.plot(x, y, color="blue", linewidth=2)

        # Masks for coloring dots
        green_mask = self.monVal[:, 2] == 0
        red_mask = self.monVal[:, 2] > 0

        # Plot dots with colors
        plt.plot(self.monVal[green_mask, 1], self.monVal[green_mask, 0],
                'o', markerfacecolor='green', markeredgecolor='black', label='stable')

        plt.plot(self.monVal[red_mask, 1], self.monVal[red_mask, 0],
                'o', markerfacecolor='red', markeredgecolor='black', label='unstable')

        plt.grid(True)
        plt.ylabel("$\lambda$")
        plt.xlabel("$u_{mon}$")
        plt.legend()
        plt.show()

# ANSYS
class ANSYS:

    def __init__(self):
        self.numDOF: int = 0
        self.u: Optional[np.ndarray] = None
        self.monVal: Optional[np.ndarray] = None

    @classmethod
    def LoadCon(cls, mesh, numInc: int = 5, nlgeom: str = "on"):

        inst = cls()

        # pre-processing
        ansys = inst.toANSYS(mesh)

        # static structural analysis
        ansys.run("/solu")
        ansys.antype("static")              # static analysis
        ansys.autots("off")                 # automatic time stepping off
        ansys.nlgeom(nlgeom)                # activate geometric nonlinearities
        ansys.nsubst(numInc)                # number of increments (substeps)
        out = ansys.solve()                 # solve
        ansys.finish()

        # post-processing
        ansys.post1()
        ansys.set("LAST")
        inst.u = np.zeros(mesh.N.shape[0]*3, dtype = "float")
        inst.u[0::3] = ansys.post_processing.nodal_displacement("X")
        inst.u[1::3] = ansys.post_processing.nodal_displacement("Y")

        ansys.exit()

        return inst

    def toANSYS(self, mesh):

        # MAPDL starten
        mapdl = launch_mapdl(nproc=1)

        # Start mapdl and clear it.
        mapdl.clear()
        mapdl.verify()

        # Enter verification example mode and the pre-processing routine.
        mapdl.prep7()

        # Element type: BEAM188.
        mapdl.et(1, "188")
        # 2D analysis (uy, ux and rotz)
        # *** element y axis coincide with global -Z axis ***
        mapdl.keyopt(1, 5, 1)   

        # Material type: linear elastic
        E = 1E10
        nu = 5*mesh.sec.EA/(12*mesh.sec.GAq) - 1
        mapdl.mp("EX", 1, E)                # Youngs modulus N/m^2
        mapdl.mp("PRXY", 1, nu)             # Poisson ratio

        h = np.sqrt(12*mesh.sec.EI/mesh.sec.EA)
        b = mesh.sec.EA/E/h
        #print(h, b, E, nu)

        # cross section of beam
        mapdl.sectype(1, "Beam", "RECT", "my_sec")  # rectangular cross section
        mapdl.secdata(b, h)                         # b (in element y) and h (in element z) in m

        # nodes
        for i, val in enumerate(mesh.N, start=1):
            mapdl.n(i, val[0], val[1], 0)

        # elements
        for val in mesh.E:
            mapdl.e(val[0], val[1])

        # boundary conditions:
        # Map BC codes to MAPDL DOF labels
        bc_map = {1: "UX", 2: "UY", 3: "ROTZ",}

        # Apply boundary conditions
        for node, bc_code in mesh.BC:
            dof = bc_map.get(bc_code)   # look up in dict
            if dof:                     # only apply if valid
                mapdl.d(node, dof)
            
        # loads
        f_map = {1:("FX", 1), 2:("FY", 1), 3:("MZ", -1)}
        for node, loadLabel, value in mesh.F:
            if loadLabel in f_map:
                fdof, sign = f_map[loadLabel]
                mapdl.f(node, fdof, sign*value)

        mapdl.finish()
        
        return mapdl
    


from typing_extensions import Self
from typing import Optional
from collections.abc import Callable

import numpy as np
import matplotlib.pyplot as plt
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

    xI = [(1 - np.sqrt(3 / 5)) * l / 2, 0.5 * l, (1 + np.sqrt(3 / 5)) * l / 2]
    wI = [5 / 9 * 0.5 * l, 8 / 9 * 0.5 * l, 5 / 9 * 0.5 * l]

    D = np.diag([EA, EI])
    
    print("-")
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
        #E0 = dudx + 0.5 * (dudx**2 + dwdx**2)
        E0 = dudx # + 0.5 * dwdx**2
        Kb = -dwdxx

        # cross section forces
        N = EA * E0
        M = EI * Kb
        CSF = np.array([[N, M]])
        print(N)

        # B-matrix
        dwdx = 0
        #B = np.array([[(1+dudx)*dN1dx,  dwdx*dN2dx,   dwdx*dN3dx, (1+dudx)*dN4dx, dwdx*dN5dx, dwdx*dN6dx],
        #              [             0,      dN2dxx,       dN3dxx,              0,     dN5dxx,     dN6dxx]])
        B = np.array([[         dN1dx,  dwdx*dN2dx,   dwdx*dN3dx,          dN4dx, dwdx*dN5dx, dwdx*dN6dx],
                      [             0,      -dN2dxx,       -dN3dxx,              0,     -dN5dxx,     -dN6dxx]])
        
        # internal force vector
        fine += B.T @ CSF.T * w

        # material stiffness matrix
        kme += B.T @ D @ B * w

        N = 0
        # geometric stiffness matrix
        # kge += N*np.array([[dN1dx*dN1dx,           0,           0, dN1dx*dN4dx,           0,           0],
        #                    [          0, dN2dx*dN2dx, dN2dx*dN3dx,           0, dN2dx*dN5dx, dN2dx*dN6dx],
        #                    [          0, dN3dx*dN2dx, dN3dx*dN3dx,           0, dN3dx*dN5dx, dN3dx*dN6dx],
        #                    [dN4dx*dN1dx,           0,           0, dN4dx*dN4dx,           0,           0],
        #                    [          0, dN5dx*dN2dx, dN5dx*dN3dx,           0, dN5dx*dN5dx, dN5dx*dN6dx],
        #                    [          0, dN6dx*dN2dx, dN6dx*dN3dx,           0, dN6dx*dN5dx, dN6dx*dN6dx]])*w

        kge += N*np.array([[          0,           0,           0,           0,           0,           0],
                           [          0, dN2dx*dN2dx, dN2dx*dN3dx,           0, dN2dx*dN5dx, dN2dx*dN6dx],
                           [          0, dN3dx*dN2dx, dN3dx*dN3dx,           0, dN3dx*dN5dx, dN3dx*dN6dx],
                           [          0,           0,           0,           0,           0,           0],
                           [          0, dN5dx*dN2dx, dN5dx*dN3dx,           0, dN5dx*dN5dx, dN5dx*dN6dx],
                           [          0, dN6dx*dN2dx, dN6dx*dN3dx,           0, dN6dx*dN5dx, dN6dx*dN6dx]])*w
    
    return fine.flatten(), kme+kge, kge

def B2D_SRA(ue, EA, EI, GAq, l):

    u1, w1, phi1 = ue[0], ue[1], ue[2]
    u2, w2, phi2 = ue[3], ue[4], ue[5]

    fine = np.zeros(6, dtype="float")
    ke = np.zeros((6, 6), dtype="float")
    kge = np.zeros((6, 6), dtype="float")

    t1 = l ** 2
    t2 = t1 ** 2
    t4 = t1 * l
    t5 = 0.1e1 / t4
    t6 = w1 * t5
    t8 = 0.1e1 / t1
    t9 = phi1 * t8
    t11 = w2 * t5
    t13 = phi2 * t8
    t15 = -6 * t6 - 3 * t9 + 6 * t11 - 3 * t13
    t16 = t15 ** 2
    t22 = 0.1e1 / l
    t27 = phi2 * t22
    t29 = 2 * phi1 * t22 + 6 * w1 * t8 - 6 * w2 * t8 + 4 * t27
    t30 = t15 * t29
    t35 = t29 ** 2
    t37 = -t15 * phi2 + t35 / 2
    t46 = phi2 ** 2
    t48 = u1 * t22 - u2 * t22 + t46 / 2
    t49 = t48 * EA
    t50 = t16 * EA * t2 / 10 + t30 * EA * t4 / 4 + t37 * EA * t1 / 3 - t29 * phi2 * l * EA / 2 + t49
    t51 = phi1 + phi2
    t52 = 3 * l * t51
    t53 = 6 * w1
    t54 = 6 * w2
    t55 = -t52 - t53 + t54
    t57 = t16 * EA
    t59 = t57 * t4 * t55 / 3
    t60 = -t55 * l
    t62 = phi1 + 2 * phi2
    t66 = l * t62 + 3 * w1 - 3 * w2
    t67 = t66 * l
    t70 = t2 * t1
    t71 = 0.1e1 / t70
    t72 = t71 * (t60 + 2 * t67)
    t75 = t71 * t55
    t76 = t29 * EA
    t77 = t15 * t76
    t80 = -3 * t57 * t72 - 6 * t77 * t75
    t81 = t2 ** 2
    t85 = 2 * t66 * t1
    t86 = phi2 * t4
    t88 = t71 * (-t85 - t86)
    t93 = t37 * EA
    t96 = -3 * t57 * t88 - 6 * t77 * t72 - 6 * t93 * t75
    t97 = t2 * t4
    t106 = EA * phi2
    t107 = t29 * t106
    t110 = 6 * t107 * t75 - 3 * t57 * t13 - 6 * t93 * t72 - 6 * t77 * t88
    t117 = t107 * t72 - t77 * t13 - t49 * t75 - t93 * t88
    t118 = t2 * l
    t124 = t107 * t88 - t93 * t13 - t49 * t72
    t132 = EI * t5
    t137 = 12 * t6 + 6 * t9 - 12 * t11 + 6 * t13
    t140 = 6 * t76 * t46 * t8 + 12 * t137 * t132 - 6 * t49 * t88
    t145 = EI * t8
    t147 = 6 * t137 * t145
    t150 = -6 * t49 * t13 - 12 * t29 * t132 - t147
    t153 = EI * t22
    t155 = -6 * t29 * t153
    t160 = -0.3e1 / 0.2e1 * t52 - 9 * w1 + 9 * w2
    t166 = 0.1e1 / t118
    t167 = t166 * (t60 + 3 * t67)
    t169 = t166 * t160
    t177 = t166 * (-t85 - 0.3e1 / 0.2e1 * t86)
    t219 = t137 * t153
    t222 = -6 * t29 * t145
    t226 = -t29 * EI
    t244 = l * t51
    t248 = 9 * t244 + 18 * w1 - 18 * w2
    t253 = 3 * t244 + t53 - t54
    t256 = t1 * t62
    t259 = 2 * l * (-w1 + w2)
    t262 = t166 * (-4 * t253 * l - 6 * t256 + 9 * t259)
    t265 = t166 * t248
    t273 = -2 * t256 + 3 * t259
    t278 = t166 * (-4 * t273 * l + t253 * t1 + 3 * t86)
    t290 = t166 * (-4 * phi2 * t2 + t273 * t1)
    fine[0] = t50
    fine[1] = -t59 + t81 * t80 / 8 + t97 * t96 / 7 + t70 * t110 / 6 + 0.6e1 / 0.5e1 * t118 * t117 + 0.3e1 / 0.2e1 * t2 * t124 + t4 * t140 / 3 + t1 * t150 / 2 - t155
    fine[2] = -t57 * t2 * t160 / 9 + t81 * (-t57 * t167 - 2 * t77 * t169) / 8 + t97 * (-2 * t77 * t167 - 2 * t93 * t169 - t57 * t177) / 7 + t70 * (2 * t107 * t169 - 2 * t93 * t167 - 2 * t77 * t177 - t57 * t27) / 6 + 0.2e1 / 0.5e1 * t118 * (t107 * t167 - t49 * t169 - t93 * t177 - t77 * t27) + t2 * (t107 * t177 - t49 * t167 - t93 * t27) / 2 + t4 * (2 * t76 * t46 * t22 - 2 * t49 * t177 + t147) / 3 + t1 * (-2 * t49 * t27 - 2 * t219 + t222) / 2 - 2 * t226
    fine[3] = -t50
    fine[4] = t59 - t81 * t80 / 8 - t97 * t96 / 7 - t70 * t110 / 6 - 0.6e1 / 0.5e1 * t118 * t117 - 0.3e1 / 0.2e1 * t2 * t124 - t4 * t140 / 3 - t1 * t150 / 2 + t155
    fine[5] = t57 * t2 * t248 / 18 + t81 * (t57 * t262 / 2 + t77 * t265) / 8 + t97 * (t57 * t278 / 2 + t77 * t262 + t93 * t265) / 7 + t70 * (t57 * t290 / 2 + t77 * t278 + t93 * t262 - t107 * t265) / 6 + t118 * (t16 * t106 / 2 + t77 * t290 + t93 * t278 - t107 * t262 + t49 * t265) / 5 + t2 * (t30 * t106 - t107 * t278 + t49 * t262 + t93 * t290) / 4 + t4 * (t37 * t106 - t107 * t290 + t49 * t278 + t147) / 3 + t1 * (-t29 * EA * t46 + t49 * t290 - 4 * t219 + t222) / 2 + l * t48 * t106 - 4 * t226

    t1 = phi1 + phi2
    t2 = w1 - w2
    t3 = 4
    t4 = 3
    t5 = t4 * t2
    t6 = 0.1e1 / l
    t7 = t6 ** 2
    t8 = t6 * t7
    t9 = EA * t6
    t10 = t9 / 30
    t11 = t10 * ((-phi2 * t3 + phi1) * l - t5)
    t3 = t10 * ((phi1 * t3 - phi2) * l + t5)
    t5 = EA * (l * t1 + 12 * t2) * t7 / 10
    t10 = phi2 ** 2
    t12 = phi1 ** 2
    t13 = t12 + t10
    t14 = u1 - u2
    t1 = t2 * t1
    t15 = 54 * t1 + 84 * t14
    t16 = t2 ** 2
    t17 = 9 * t13 * l
    t18 = 216 * t16
    t19 = 840 * EI
    t20 = 6 * phi1 * phi2
    t10 = t4 * (t12 - t10)
    t21 = -t10 + t20
    t22 = 28 * t14
    t23 = 72 * t2
    t24 = t23 * phi1 + t22
    t25 = 108 * t16
    t26 = 1680 * EI
    t10 = t10 + t20
    t20 = t23 * phi2 + t22
    t22 = 0.1e1 / 0.70e2
    t23 = 0.1e1 / 0.280e3
    t27 = t22 * (((t17 + t15) * l + t18) * EA + t19) * t8
    t28 = t23 * ((t25 + l * (l * t21 + t24)) * EA + t26) * t7
    t8 = t22 * (((-t17 - t15) * l - t18) * EA - t19) * t8
    t15 = t23 * ((t25 + l * (l * t10 + t20)) * EA + t26) * t7
    t17 = 36
    t18 = 9 * phi1
    t2 = 9 * t2 * (phi1 - phi2)
    t19 = 56 * t14
    t16 = 54 * t16
    t22 = 0.1e1 / 0.420e3
    t29 = 0.1e1 / 0.840e3
    t1 = t29 * (t26 - EA * l * (-18 * t1 + 28 * t14 + (-12 * phi1 * phi2 + 9 * t13) * l)) * t6
    t13 = t23 * ((-t25 + l * (-l * t21 - t24)) * EA - t26) * t7
    t7 = t23 * ((-t25 + l * (-l * t10 - t20)) * EA - t26) * t7
    ke = np.array([[t9,t5,t3,-t9,-t5,-t11],[t5,t27,t28,-t5,t8,t15],[t3,t28,t22 * ((t16 + l * (((phi2 * t4 - t18) * phi2 + t12 * t17) * l + t19 - t2)) * EA + t26) * t6,-t3,t13,t1],[-t9,-t5,-t3,t9,t5,t11],[-t5,t8,t13,t5,t27,t7],[-t11,t15,t1,t11,t7,t22 * ((t16 + l * (((phi2 * t17 - t18) * phi2 + t12 * t4) * l + t19 + t2)) * EA + t26) * t6]])

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
    def leafSpring(cls, b: float, h: float, M: float, L: float = 0.5, n: int = 10, elType: Callable = B2D_SR):
        """
        Construct a simple 2D leaf-spring beam problem.
        b: width
        h: thickness
        M: applied end moment
        L: total length (default 0.5 m)
        n: number of elements (default 10)
        elType: element type (default B2D_SR)
        """
        inst = cls()
        # nodes
        inst.N = np.zeros((n + 1, 2))
        inst.N[:, 0] = np.linspace(0, L, n + 1)
        # elements
        inst.E = np.zeros((n, 2), dtype=int)
        inst.E[:, 0] = np.linspace(2, n + 1, n)
        inst.E[:, 1] = np.linspace(1, n, n)
        # boundary conditions
        inst.BC = np.array([[1, 1], [1, 2], [1, 3]])
        # external force
        inst.F = np.array([[n + 1, 2, M]])
        # monitor DOF
        inst.monDOF = [n + 1, 1]
        # section properties
        inst.sec = SectionProperty(b, h)
        # element type
        inst.elType = elType

        return inst
    
    @classmethod
    def shallowArch(cls, R: float = 40, alpha: float = 20, F: float = 1E6, b: float = 0.01, h: float = 0.2, n: int = 40, elType: Callable = B2D_SR):
        """
        Construct a shallow arch under a point load at the center and pinned supports at the ends.
        R:      radius of the arch
        alpha:  half of the opening angle in degrees
        F:      magnitude of the vertical force
        b:      width of the cross section
        h:      height of the cross section
        n:      number of elements used
        elType: element type
        """
        inst = cls()
        # nodes
        phi0 = alpha / 180 * np.pi
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
        # force at center
        inst.F = np.array([[n / 2 + 1, 2, -F]])
        # section
        inst.sec = SectionProperty(b, h)
        # monitor DOF
        inst.monDOF = [n / 2 + 1, 1]
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
        xx = 180 - phi0
        phi = np.linspace(-xx / 180 * np.pi, (180 + xx) / 180 * np.pi, n + 1)
        inst.N = np.zeros((n + 1, 2))
        inst.N[:, 0] = R * np.cos(phi)
        inst.N[:, 1] = R * np.sin(phi)

        inst.E = np.zeros((n, 2), dtype=int)
        inst.E[:, 0] = np.linspace(2, n + 1, n)
        inst.E[:, 1] = np.linspace(1, n, n)

        inst.BC = np.array([[1, 1],
                            [1, 2],
                            [n+1, 1],
                            [n+1, 2]], dtype=int)

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
        plot mesh of finite element discretization
        """
        plt.plot(self.N [:,0], self.N[:, 1], 'k-o', markerfacecolor = 'yellow', markeredgecolor='k', markersize = 5)
        indBC = (self.BC[:,0] - 1).astype(int)
        plt.plot(self.N [indBC,0], self.N[indBC, 1], 'o', markerfacecolor = 'red', markeredgecolor='k', markersize = 6)
        indF = (self.F[:,0] - 1).astype(int)
        plt.plot(self.N [indF,0], self.N[indF, 1], 'o', markerfacecolor = 'green', markeredgecolor='k', markersize = 6)
        plt.axis('equal')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(which = 'major', linestyle = ':')
        plt.show()

    def imperfection(self, imp: np.ndarray, scale: float):

        self.N[:,0] += scale*imp[0::3]
        self.N[:,1] += scale*imp[1::3]

# Finite element solvers
class FEMsolve:

    def __init__(self):
        self.numDOF: int = 0
        self.u: Optional[np.ndarray] = None
        self.monVal: Optional[np.ndarray] = None

    @classmethod
    def LoadCon(cls, mesh, numInc: int, maxIter: int = 15, maxErr: float = 1E-6):
        """
        Load-controlled nonlinear finite element analysis
            - full newton method
            - constant incremental loading 

        mesh:    finite element discretization
        numInc:  number of load increments
        maxIter: maximum number of iterations (default: 15)
        maxErr:  maximum relative error in equilibrium (default: 1E-6)
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
                print(f"{iter+1:3d}    {err:12.4e}")

                # solution converged
                if err < maxErr:
                    # stability check
                    eigVal = np.linalg.eigvals(K)
                    numNegEig = np.sum(eigVal<1E-8)
                    
                    # get monitor dof
                    inst.monVal = np.append(inst.monVal, np.array([[Lambda, inst.u[dofM], numNegEig]]), axis = 0)

                    # print stability
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
    def arcL(cls, mesh, numInc: int, maxIter: int = 15, Lambda0: float = 0.1, maxErr: float = 1E-6):
        """
        Arc-Length Method with classical Riks constrain surface
        mesh:    finite element discretization
        numInc:  number of increments
        maxIter: maximal number of interations (default = 15) 
        Lambda0: inital load increment for arc-length (default = 0.1)
        maxErr:  maximal relative error (default = 1E-6)
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
                    print(f"{iter+1:3d}    {err:12.4e}")

                    # solution converged
                    if err < maxErr:
                        # stability check
                        eigVal = np.linalg.eigvals(K)
                        numNegEig = np.sum(eigVal<1E-8)
                        
                        # get monitor dof
                        inst.monVal = np.append(inst.monVal, np.array([[Lambda, inst.u[dofM], numNegEig]]), axis = 0)

                        # print stability
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
    def linBuckling(cls, mesh, u:np.ndarray, numModes: int = 3):
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
        plt.ylabel("load factor")
        plt.xlabel("monitor DOF")
        plt.legend()
        plt.show()


# get discretizatin of example
mesh = GNLexamples.leafSpring(b = 0.05, h = 0.001, M = -0.1, n = 3, elType = B2D_SRA)
#mesh = GNLexamples.shallowArch(n = 20, F = 5.5E4)
#mesh = GNLexamples.deepArch(F = 0.1, n = 10, phi0 = 180)
#mesh.elType = B2D_LR
# plot mesh
mesh.plotMesh()

# nonlinear analysis
sol = FEMsolve.LoadCon(mesh, numInc = 1)
#sol = FEMsolve.arcL(mesh, numInc = 15, Lambda0 = 0.4)
sol.plotMonitor()
sol.plotDisplacement(mesh, sol.u)

print(sol.u[0::3])
print(sol.u[1::3])
print(sol.u[2::3])

# stability analysis
#stabAnalysis = FEMsolve.linBuckling(mesh, sol.u)
#stabAnalysis.plotDisplacement(mesh, stabAnalysis.buckModes[:,2], scale = 10)

#mesh.imperfection(stabAnalysis.buckModes[:,0], scale = 0.01)
#postBuckling = FEMsolve.arcL(mesh, numInc = 40, Lambda0 = 0.2)
#postBuckling.plotMonitor()
#postBuckling.plotDisplacement(mesh, postBuckling.u)

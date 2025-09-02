import numpy as np
#import matplotlib.pyplot as plt

def B2D_SR(ue,EA,EI,GAq,l):

    '''
    2D beam element for large rotations 
        - Bernoulli theory
        - 2 nodes

    ue              ... nodal displacements
    EA, EI, GAq,    ... axial, bending and shear stiffness (not needed)
    l               ... element length
    '''

    u1,w1,phi1 = ue[0],ue[1],ue[2]
    u2,w2,phi2 = ue[3],ue[4],ue[5]

    fine = np.zeros((6,1), dtype="float")
    kme  = np.zeros((6,6), dtype="float")
    kge  = np.zeros((6,6), dtype="float")


    #xI = [(0.5 - 1/np.sqrt(3))*l, (0.5 + 1/np.sqrt(3))*l]
    #w  = [l/2, l/2]

    xI = [(1 - np.sqrt(3/5))*l/2, 0.5*l, ((1 + np.sqrt(3/5)))*l/2]
    wI = [5/9*0.5*l, 8/9*0.5*l, 5/9*0.5*l]

    D = np.diag([EA, EI])

    xx = 0

    for i in range(0, len(wI)):

        x = xI[i]
        w = wI[i]

        # derivative of shape functions
        # first
        dN1dx =  1/l
        dN2dx =  6*x*(l - x)/l**3
        dN3dx =  -(-2*l*x + 3*x**2)/l**2
        dN4dx = -1/l
        dN5dx = -6*x*(l - x)/l**3
        dN6dx =  -(l - x)*(l - 3*x)/l**2
        # second
        dN2dxx = (6*l - 12*x)/l**3
        dN3dxx = -(-2*l + 6*x)/l**2
        dN5dxx = (12*x - 6*l)/l**3
        dN6dxx = -(-4*l + 6*x)/l**2

        # strains
        dudx  = dN1dx*u1 + dN4dx*u2
        dwdx  = dN2dx*w1 + dN3dx*phi1 + dN5dx*w2 + dN6dx*phi2
        dwdxx = -(dN2dxx*w1 + dN3dxx*phi1 + dN5dxx*w2 + dN6dxx*phi2)
        E0 = dudx + 0.5*(dudx**2 + dwdx**2)
        Kb = dwdxx

        # cross section forces
        N = EA*E0
        M = EI*Kb
        CSF = np.array([[N, M]])
 
        # B-matrix
        B = np.array([[(1+dudx)*dN1dx,  dwdx*dN2dx,   dwdx*dN3dx, (1+dudx)*dN4dx, dwdx*dN5dx, dwdx*dN6dx],
                      [             0,     -dN2dxx,      -dN3dxx,              0,    -dN5dxx,    -dN6dxx]])
        
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


def B2D_lr(ue,EA,EI,GAq,l):

    u1,w1,phi1 = ue[0],ue[1],ue[2]
    u2,w2,phi2 = ue[3],ue[4],ue[5]

    # one point (midpoint) integration

    # displacements
    u = 0.5*(u1 + u2)
    w = 0.5*(w1 + w2)
    phi = 0.5*(phi1 + phi2)

    # derivatives
    dudx = (u1 - u2)/l
    dwdx = (w1 - w2)/l
    dphidx = (phi1 - phi2)/l

    # deformations
    E0 = dudx + 0.5*(dudx**2 + dwdx**2)
    Lambda = (1 + dudx)*np.cos(phi) - dwdx*np.sin(phi)
    Kb = Lambda*dphidx
    Gamma = (1 + dudx)*np.sin(phi) + dwdx*np.cos(phi)

    # cross sectional forces
    CSF = np.array([EA*E0, GAq*Gamma, EI*Kb]).T

    # B-matrix
    B = np.array([[(1 + dudx)/l,            dwdx/l,                 0,                              -(1 + dudx)/l,          -dwdx/l,                0                               ],
                  [np.sin(phi)/l,           np.cos(phi)/l,          0.5*Lambda,                     -np.sin(phi)/l,         -np.cos(phi)/l,         0.5*Lambda                      ],
                  [dphidx*np.cos(phi)/l,    -dphidx*np.sin(phi)/l,  (Lambda/l - Gamma*dphidx*0.5),  -dphidx*np.cos(phi)/l,  dphidx*np.sin(phi)/l,   (-Lambda/l - Gamma*dphidx*0.5)  ]])
    
    # internal force vector
    fin = np.zeros((6,1), dtype="float")
    fin = B.T @ CSF * l

    # material
    D = np.diag([EA, GAq, EI])

    # material stiffness matrix
    km = np.zeros((6,6), dtype="float")
    km = B.T @ D @ B * l

    # geometric stiffness matrix
    kg = np.zeros((6,6), dtype="float")

    G1 =  CSF[1]*np.cos(phi) - CSF[2]*dphidx*np.sin(phi)
    G2 = -CSF[1]*np.sin(phi) - CSF[2]*dphidx*np.cos(phi)
    G3 = -CSF[1]*Gamma - CSF[2]*dphidx*Lambda

    A1 = np.array([[CSF[0], 0, CSF[2]*np.cos(phi)],[0, CSF[0], -CSF[2]*np.sin(phi)],[CSF[2]*np.cos(phi), -CSF[2]*np.sin(phi), 0]])
    A2 = np.diag([0, 0, G3])
    A3 = np.array([[0, 0, G1],[0, 0, G2],[0, 0, -CSF[2]*Gamma]])
    A4 = A3.T

    kg[np.ix_(range(0,3),range(0,3))] = 1/l*A1*1/l + 0.5*A2*0.5 + 1/l*A3*0.5 + 0.5*A4*1/l
    kg[np.ix_(range(0,3),range(3,6))] = 1/l*A1*(-1/l) + 0.5*A2*0.5 + 1/l*A3*0.5 + 0.5*A4*(-1/l)
    kg[np.ix_(range(3,6),range(0,3))] = kg[np.ix_(range(0,3),range(3,6))].T
    kg[np.ix_(range(3,6),range(3,6))] = 1/l*A1*1/l + 0.5*A2*0.5 + (-1/l)*A3*0.5 + 0.5*A4*(-1/l)

    return fin, km+kg, kg

def B2D_LR(ue,EA,EI,GAq,l):

    u1,w1,phi1 = ue[0],ue[1],ue[2]
    u2,w2,phi2 = ue[3],ue[4],ue[5]

    # internal force vector
    t1 = 0.1e1 / l
    t2 = u1 * t1
    t3 = u2 * t1
    t4 = t2 - t3
    t6 = t1 * t4 + t1
    t8 = t4 ** 2
    t12 = w1 * t1 - w2 * t1
    t13 = t12 ** 2
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
    t30 = t29 ** 2
    t34 = -t12 * t18 + t4 * t21 + t21
    t35 = t34 * EI * t30
    t36 = t35 * t26
    t44 = t15 * EA * t1 * t12 - t35 * t19 + t24 * t26
    t47 = t23 * GAq * t34 / 2
    t48 = -t29 * t23 / 2
    t49 = t1 * t34
    t52 = t29 * t34

    fin = np.zeros((6,1))
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
    t4 = t3 ** 2
    t5 = 0.1e1 / 0.2e1
    t6 = t5 * (t2 ** 2 + t4) + t2
    t2 = t2 + 1
    t7 = t1 * t2
    t8 = t5 * (phi1 + phi2)
    t9 = np.sin(t8)
    t8 = np.cos(t8)
    t10 = t1 * (phi1 - phi2)
    t11 = t8 ** 2
    t12 = t9 ** 2
    t13 = t10 ** 2 * EI
    t14 = t1 ** 2
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
    t20 = t20 ** 2
    t38 = -GAq * (-t2 ** 2 + t20) / 4
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


    #geometrix stiffness matrix
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
    t10 = t9 ** 2
    t11 = t6 * GAq * t8
    t12 = t1 ** 2
    t13 = t4 * t1 * (-t5 * t10 * EI * t7 + t11)
    t14 = t6 * t12 * t9 * EI * t7
    t15 = t14 + t13
    t13 = -t14 + t13
    t2 = t1 * EA * (t4 * (t2 ** 2 + t3 ** 2) + t2)
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
    t21 = -t8 ** 2 * GAq / 4
    t8 = -l * (t7 ** 2 * t10 * EI + t8 ** 2 * GAq) / 4
    t10 = -t1 * (t17 + t14)
    t1 = t1 * (-t6 + t5)
    kg = np.array([[t2,0,l * t15,-t2,0,l * t13],[0,t2,l * t16,0,-t2,l * t12],[l * (t18 * EI * t7 * t9 + t11),l * (t19 * EI * t7 * t9 - t3),l * (t21 + (t4 - t20) * EI * t7 * t9),l * (-t18 * EI * t7 * t9 - t11),l * (-t19 * EI * t7 * t9 + t3),t8],[-t2,0,-l * t15,t2,0,-l * t13],[0,-t2,-l * t16,0,t2,-l * t12],[l * (t7 * t10 * EI * t9 + t11),l * (t1 * EI * t7 * t9 - t3),t8,l * (-t7 * t10 * EI * t9 - t11),l * (-t1 * EI * t7 * t9 + t3),l * (t21 + (t4 + t20) * EI * t7 * t9)]])

    return fin, k, kg



def timoshenko_beam_2d_stiffness(EA, EI, GAq, L):
    """
    Compute the 6x6 local stiffness matrix for a 2D Timoshenko beam element.

    Parameters:
    E      : Young's modulus
    A      : Cross-sectional area
    I      : Second moment of area (about z-axis)
    G      : Shear modulus
    kappa  : Shear correction factor
    L      : Length of the beam element

    Returns:
    K      : 6x6 numpy array (element stiffness matrix)
    """
    # Shear rigidity
    kGA = GAq

    # Shear deformation parameter
    phi = (12 * EI) / (kGA * L**2)

    # Common bending-shear stiffness factor
    f = EI / ((1 + phi) * L**3)

    # Axial stiffness
    EA_L = EA / L

    # Build stiffness matrix
    K = np.array([
        [ EA_L,     0,          0,     -EA_L,     0,          0],
        [ 0,        12*f,       6*f*L,  0,        -12*f,      6*f*L],
        [ 0,        6*f*L,      (4 + phi)*f*L**2, 0, -6*f*L,   (2 - phi)*f*L**2],
        [-EA_L,     0,          0,      EA_L,     0,          0],
        [ 0,       -12*f,      -6*f*L,  0,         12*f,     -6*f*L],
        [ 0,        6*f*L,      (2 - phi)*f*L**2, 0, -6*f*L,   (4 + phi)*f*L**2]
    ])

    return K

def B2D_FJ(ue,EA,EI,GAq,l):

    u1,w1,phi1 = ue[0],ue[1],ue[2]
    u2,w2,phi2 = ue[3],ue[4],ue[5]

    fine = np.zeros((6,1), dtype="float")
    kme   = np.zeros((6,6), dtype="float")
    kge  = np.zeros((6,6), dtype="float")


    #xI = [(0.5 - 1/np.sqrt(3))*l, (0.5 + 1/np.sqrt(3))*l]
    #w  = [l/2, l/2]

    xI = [(0.5 - np.sqrt(3/5))*l, 0.5*l, (0.5 + np.sqrt(3/5))*l]
    wI = [5/9*0.5*l, 8/9*0.5*l, 5/9*0.5*l]

    D = np.diag([EA, EI])

    for i in range(0, len(wI)):

        x = xI[i]
        w = wI[i]

        # derivative of shape functions
        # first
        dN1dx =  1/l
        dN2dx =  6*x*(l - x)/l**3
        dN3dx =  (-2*l*x + 3*x**2)/l**2
        dN4dx = -1/l
        dN5dx = -6*x*(l - x)/l**3
        dN6dx =  (l - x)*(l - 3*x)/l**2
        # second
        dN2dxx = (6*l - 12*x)/l**3
        dN3dxx = (-2*l + 6*x)/l**2
        dN5dxx = (12*x - 6*l)/l**3
        dN6dxx = (-4*l + 6*x)/l**2

        # strains
        dudx  = dN1dx*u1 + dN4dx*u2
        dwdx  = dN2dx*w1 + dN3dx*phi1 + dN5dx*w2 + dN6dx*phi2
        dwdxx = -(dN2dxx*w1 + dN3dxx*phi1 + dN5dxx*w2 + dN6dxx*phi2)
        E0 = dudx + 0.5*(dudx**2 + dwdx**2)
        Kb = dwdxx

        # cross section forces
        N = EA*E0
        M = EI*Kb
        CSF = np.array([[N, M]])
 
        # B-matrix
        B = np.array([[(1+dudx)*dN1dx,  dwdx*dN2dx,   dwdx*dN3dx, (1+dudx)*dN4dx, dwdx*dN5dx, dwdx*dN6dx],
                      [             0,     -dN2dxx,      -dN3dxx,              0,    -dN5dxx,    -dN6dxx]])
        
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


#u = np.array([0, 0, 0, 0, 0, 0], dtype='float')
u = 100*np.random.randn(6)
u[2], u[5] = u[2]*0.05, u[5]*0.1

#u = np.array([-41.99969484,  36.71836306,  25.92069098, -15.92506805,  78.73847371, -69.3048176 ])

EA = 100
EI = 120
GAq = 150
l = 50

#fine0, ke0, kg = B2D_LR(u,EA,EI,GAq,l)
#fine0, ke0, kg = B2D_LR(u,EA,EI,GAq,l)
fine0, ke0, kg = B2D_SR(u,EA,EI,GAq,l)
#fine0, ke0, kg = B2D_FJ(u,EA,EI,GAq,l)

# B_num = np.zeros((6), dtype="float")
# for i in range(0,6):
#          uP = u.astype(float)
#          uP[i] += 1E-8
#          fineP, rr, ss, Bp, strp = B2D_lr(uP,EA,EI,GAq,l)
         
#          B_num[i] = (strp - str0)/1E-8

# print(B_num)
# print(B0[2,:])

k_num = np.zeros((6,6), dtype="float")
h = 1E-8
for i in range(0,6):
    for j in range(0,6):
        uP = u.astype(float)
        uP[j] += h
        #fineP, rr, ss = B2D_FJ(uP,EA,EI,GAq,l)
        fineP, rr, ss = B2D_SR(uP,EA,EI,GAq,l)
        #fineP, rr, ss = B2D_LR(uP,EA,EI,GAq,l)

        k_num[i,j] = (fineP[i] - fine0[i])/h

err = (k_num - ke0)/ke0*1000

#kt = timoshenko_beam_2d_stiffness(EA, EI, GAq, l)

print(u)
for row in err:
    print('  '.join(f"{val:10.3f}" for val in row))

#print('/n')
#for row in kt:
#   print('  '.join(f"{val:10.2f}" for val in row))
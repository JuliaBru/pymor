__author__ = 'j_brun16'
import numpy as np
from pymordemos.rb_to_fp import rb_solutions


def legpol(V,m):
    L=np.array(np.zeros([np.shape(V)[0],m+1]))
    L[:,0] = 1.
    if m>=1:
        L[:,1] = V
        if m>=2:
            L[:,2] = 1./2.*(3.*V**2.-1.)
            if m>=3:
                for i in range(3,m+1):
                    L[:,i] = 1./i*((2.*i-1.)*V*L[:,i-1]-(i-1.)*L[:,i-2])
                for i in range(m+1):
                    L[:,i]*=np.sqrt((2.*i+1.)/2.)
    return L

def Godunov(A):
    assert A.ndim==2
    assert A.shape[0]==A.shape[1]
    W,V=np.linalg.eig(A)
    WV=[W,V]
    DW=DW.sort(axis=0)


def myl2prod(U,V,h):
    assert np.shape(U)==np.shape(V)
    #h=(domain[1]-domain[0])/np.shape(U)[0]
    return U.dot(V)*h


def syscond(V,C,h,m): #V=V-Stuetzstellen (Grid), C=Vektor mit Auswertungen auf Grid, m=dim
    Leg=legpol(V,m)
    cond=np.array(np.zeros([m]))
    for i in range(m):
        cond[i]=myl2prod(Leg[:,i],C,h)
    return cond

def sysmatr(V,h,m):
    Leg=legpol(V,m)
    Matr=np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            Matr[i,j]=myl2prod(Leg[:,i]*V,Leg[:,j],h)
    return Matr

def massmatr(V,h,m):
    Leg=legpol(V,m)
    Matr=np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            Matr[i,j]=myl2prod(Leg[:,i],Leg[:,j],h)
    return Matr


def dirichletfunc(V,h): #nicht analytisch... delta-Distr analytisch def?
    #Domain [-1,1], Dl=100*delta(1-v), Dr=100*delta(1+v)
    n=np.shape(V)[0]
    Dl=np.array(np.zeros([n]))
    Dr=np.array(np.zeros([n]))
    Dl[-1]=1./h #Dirac
    Dr[0]=1./h #Dirac
    Dl*=100.
    Dr*=100.
    return Dl,Dr

def VGrid(n):
    discr=rb_solutions(0,n,(-1.,1.))
    V=discr.products['l2'].grid.quadrature_points(1)[:,0,0]
    h=discr.products['l2'].grid.volumes(0)
    return V,h




def Sysdirichlet(n,m):
    V,h=VGrid(n)
    h=np.max(h)
    Dl,Dr=dirichletfunc(V,h)
    Leg=legpol(V,m)
    return syscond(V,Dl,h,m),syscond(V,Dr,h,m)

def SysInitial(n,m):
    V,h=VGrid(n)
    nv=np.shape(V)[0]
    h=np.max(h)
    DI=10**(-4)*np.ones(nv)
    return syscond(V,DI,h,m)

def Sysmatrix(n,m):
    V,h=VGrid(n)
    h=np.max(h)
    return sysmatr(V,h,m)

def MassMatrix(n,m):
    V,h=VGrid(n)
    h=np.max(h)
    return massmatr(V,h,m)



b=Sysdirichlet(200,3)
A=Sysmatrix(200,3)
B=MassMatrix(200,3)
I=SysInitial(200,3)

#D=Sysdirichlet(200,5)
#print(D)


#d=VGrid(20)
#s=syscond(V,dr,10)




#print(s)









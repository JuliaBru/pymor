__author__ = 'j_brun16'
import numpy as np
#from pymordemos.rb_to_fp import rb_solutions
from pymor.operators import cg
from pymor.la import NumpyVectorArray
from pymor.functions import GenericFunction


def legpol(V,m):
    L=np.array(np.zeros([np.shape(V)[0],m]))
    L[:,0] = 1.
    if m>=2:
        L[:,1] = V
        if m>=3:
            L[:,2] = 1./2.*(3.*V**2.-1.)
            if m>=4:
                for i in range(3,m):
                    L[:,i] = 1./i*((2.*i-1.)*V*L[:,i-1]-(i-1.)*L[:,i-2])
    for i in range(m):
        L[:,i]*=np.sqrt((2.*i+1.)/2.)
    return L

def legpolchar(V,m):
    L=np.array(np.zeros((m,)+np.shape(V)))
    L[0,...] = 1.
    if m>=2:
        L[1,...] = V
        if m>=3:
            L[2,...] = 1./2.*(3.*V**2.-1.)
            if m>=4:
                for i in range(3,m):
                    L[i,...] = 1./i*((2.*i-1.)*V*L[i-1,...]-(i-1.)*L[i-2,...])
    for i in range(m):
        L[i,...]*=np.sqrt((2.*i+1.)/2.)*(V>=-1)*(V<=1)
    return NumpyVectorArray(L)




#discr=rb_solutions(0,1000,[-1,1])
#grid=discr.visualizer.grid
#a=grid.quadrature_points(1,order=2)
#L=legpolchar(grid.quadrature_points(1,order=2)[:,0,0],5)
#discr.visualize(L)
#prod=discr.l2_product
#M=prod.apply2(L,L,False)
#diff=discr.h1_product
#S=diff.apply2(L,L,False)
#absorb=discr.absorb_product
#A=absorb.apply2(L,L,False)
#print('M:')
#print(M)
#print('S:')
#print(S)
#print('A:')
#print(A)


def LegendreMatrices(n,m):
    discr=rb_solutions(0,n,[-1,1])
    grid=discr.visualizer.grid
    L=legpolchar(grid.quadrature_points(1,order=2)[:,0,0],m)
    mprod=discr.l2_product
    M=mprod.apply2(L,L,False)
    dprod=discr.absorb_product
    D=dprod.apply2(L,L,False)
    sprod=discr.h1_product
    S=sprod.apply2(L,L,False)
    return M,D,S


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


def dirichletfunc(V,h): #nicht analytisch... delta-Distr analytisch def?
    #Domain [-1,1], Dl=delta(1-v), Dr=delta(1+v) beim drueberintegrieren Faktor 0.5 weil Delta am Rand
    n=np.shape(V)[0]
    Dl=np.array(np.zeros([n]))
    Dr=np.array(np.zeros([n]))
    Dl[-1]=1./h #Dirac
    Dr[0]=1./h #Dirac
    return 0.5*Dl,0.5*Dr

def VGrid(n):
    discr=rb_solutions(0,n,(-1.,1.))
    V=discr.products['l2'].grid.quadrature_points(1)[:,0,0]
    h=discr.products['l2'].grid.volumes(0)
    return V,h




def Sysdirichlet(n,m):
    V,h=VGrid(n)
    h=np.max(h)
    Dl,Dr=dirichletfunc(V,h)
    return syscond(V,Dl,h,m),syscond(V,Dr,h,m)





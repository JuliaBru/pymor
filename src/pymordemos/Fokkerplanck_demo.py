__author__ = 'j_brun16'

from pymordemos.rb_to_fp import pod_from_snapshots
from pymordemos.fokkerplanck import fp_system
from greedy_fp_neu import greedy_fp
import time
import numpy as np
import csv
from pymor.la import NumpyVectorArray
import pickle

from pymor.la.pod import pod
tic=time.time()

#POD from random snapshots
if False:
    rb,sw,v_discr=pod_from_snapshots(snapshots=10,rb_size=1)
    fpsol,x_discr=fp_system(m=1,basis_type='RB',basis_pl_discr=(rb,v_discr))
    x_discr.visualize(fpsol)

#Legendre
if True:
    fpsol,x_discr=fp_system(m=6,basis_type='Leg',problem_name='2Beams',n_grid=100,save_csv=True,save_time=True, save_pickled=False, CFL_type='Auto')
    x_discr.visualize(fpsol)







#Greedy Variante 1
#In jeder Stufe Modellordnung neue Snapshots
#Jeweils erst Sample viele Random, dann fuer i<imax jeweils Sample viele in gerinstem Abstand zum bisher besten.
if False:
    fpsol,x_discr=greedy_fp(MaxOrdn=2,imax=1,sample=5,test_grid=50,seed=10)
    x_discr.visualize(fpsol)


def fperror(V,FPLoes):
    (nt,nx)=V.data.shape
    if nt < 1000 or nx < 500:
        fpmitt=np.zeros((nt,nx))
        for i in range(nx):
            for j in range(nt):
                fpmitt[j,i]=np.mean(FPLoes[j*1000/nt:(j+1)*1000/nt,i*500/nx:(i+1)*500/nx ])
    else:
        fpmitt=FPLoes
    fehl=np.sum(np.abs(fpmitt-V.data))/np.sum(np.abs(fpmitt))
    return fehl


toc=time.time()

#Berechnung mit vorher gepickleter Basis
#if True:

Compute_Error=False

if Compute_Error == True:

    FPLoes=np.zeros((1000,500))
    with open('fploes.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in reader:
            for j in range(500):
                FPLoes[i,j]=float(row[j])
            i+=1


    print('Error of reduced solution is {}'.format(fperror(fpsol,FPLoes)))
    print('Solving took {} s'.format(toc-tic))
    x_discr.visualize(NumpyVectorArray(FPLoes))
# -*- coding: utf-8 -*-


import pickle

from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymordemos.fokkerplanck import fp_system
from pymor.analyticalproblems.fokkerplanck_rb import Fokkerplanck_V
from pymor.la.pod import pod
import numpy as np
import csv
from pymor.la import NumpyVectorArray
from datetime import datetime as date
import time
from pymor.functions import GenericFunction
from pymor.la.gram_schmidt import gram_schmidt


problem=Fokkerplanck_V(problem='SourceBeam', delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                       dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))
# v_discr=dict.fromkeys(range(6))
n=100.
# #grid=discretization.visualizer.grid
# v_discr[0], _ = discretize_elliptic_cg_plus(problem, diameter=1. / n)
# n=250.
# v_discr[1],_=discretize_elliptic_cg_plus(problem, diameter=1. / n)
# v_discr[2]=v_discr[1]
# v_discr[3]=v_discr[1]
# v_discr[4]=v_discr[1]
# v_discr[5]=v_discr[1]
v_discr,_=discretize_elliptic_cg_plus(problem, diameter=1. / n)

def deltae1(x):
    return np.exp(-(x-1)**2/0.1)
def deltae2(x):
    return np.exp(-(x-1)**2/0.01)
def deltae3(x):
    return np.exp(-(x-1)**2/0.001)
def deltae4(x):
    return np.exp(-(x-1)**2/0.0001)
def deltakrass(x):
    return (x==1)



Deltae1=GenericFunction(deltae1, shape_range=(1,))
Deltae2=GenericFunction(deltae2, shape_range=(1,))
Deltae3=GenericFunction(deltae3, shape_range=(1,))
Deltae4=GenericFunction(deltae4, shape_range=(1,))
Deltaek=GenericFunction(deltakrass, shape_range=(1,))


dirich1=Deltae1.evaluate(v_discr.visualizer.grid.centers(1))[:,0]
dirich1=dirich1/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirich1),NumpyVectorArray(dirich1),pairwise=False)))
dirich2=Deltae2.evaluate(v_discr.visualizer.grid.centers(1))[:,0]
dirich2=dirich2/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirich2),NumpyVectorArray(dirich2),pairwise=False)))
dirich3=Deltae3.evaluate(v_discr.visualizer.grid.centers(1))[:,0]
dirich3=dirich3/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirich3),NumpyVectorArray(dirich3),pairwise=False)))
dirich4=Deltae4.evaluate(v_discr.visualizer.grid.centers(1))[:,0]
dirich4n=Deltae4.evaluate(v_discr.visualizer.grid.centers(1))[:,0]
dirich4n*=(dirich4n > 0.0001)
dirich4=dirich4/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirich4),NumpyVectorArray(dirich4),pairwise=False)))
dirich4n=dirich4n/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirich4n),NumpyVectorArray(dirich4n),pairwise=False)))

dirichk=Deltaek.evaluate(v_discr.visualizer.grid.centers(1))[:,0]
dirichk=dirichk/(np.sqrt(v_discr.products['l2'].apply2(NumpyVectorArray(dirichk),NumpyVectorArray(dirichk),pairwise=False)))


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

FPLoes=np.zeros((1000,500))
with open('fploes.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i=0
    for row in reader:
        for j in range(500):
            FPLoes[i,j]=float(row[j])
        i+=1



####################################################################################### 20 20 20 20 20 20 20
V=pickle.load(open("rb-daten 20000, Nr. 0 15-01-29 16:57:02.p",'rb'))
POD=dict.fromkeys(range(10))
RB=dict.fromkeys(range(5))
#
#
#
#
#
#
POD[0],_=pod(NumpyVectorArray(V.data[2000:2020,:]),modes=16)
#
#
#
# FT=np.ones((10,6))
# T=np.ones((10,6))
#
# for i in (range(10)):
#     for j in range(9,15):
#         print('i={}, m={}'.format(i,j+1))
#         if POD[i]._len < j+1:
#             FT[i,j-9]=-1.
#             T[i,j-9]=-1
#         else:
#             tic = time.time()
#             redsol,x_discr = fp_system(m=j+1,basis_type='RB',basis_pl_discr=(NumpyVectorArray(POD[i].data[0:j+1,:]),v_discr))
#             FT[i,j-9]=fperror(redsol,FPLoes=FPLoes)
#             T[i,j-9]=time.time()-tic
#
#
# d=date.now()
# with open('Tabelle POD Loes Fehler m=10-15 {} je 20 snapshots.csv'.format(d.strftime("%y-%m-%d %H:%M:%S")),'w') as csvfile:
#     writer=csv.writer(csvfile)
#     writer.writerow([' ','m','Nr.0', 'Nr.1','Nr.2', 'Nr.3','Nr.4', 'Nr.5','Nr.6','Nr.7','Nr.8','Nr.9'])
#     for j in range(9,15):
#         writer.writerow(['Sol',j+1,FT[0,j-9],FT[1,j-9],FT[2,j-9],FT[3,j-9],FT[4,j-9],FT[5,j-9],FT[6,j-9],FT[7,j-9],FT[8,j-9],FT[9,j-9]])
#         writer.writerow(['Time',j+1,T[0,j-9],T[1,j-9],T[2,j-9],T[3,j-9],T[4,j-9],T[5,j-9],T[6,j-9],T[7,j-9],T[8,j-9],T[9,j-9]])
# #
# #
# #
# POD=dict.fromkeys(range(10))
# #
# #
#
#
#
# #
#POD[0],_=pod(NumpyVectorArray(V.data[0:200,:]),modes=16)
# POD[1],_=pod(NumpyVectorArray(V.data[200:400,:]),modes=16)
# POD[2],_=pod(NumpyVectorArray(V.data[400:600,:]),modes=16)
# POD[3],_=pod(NumpyVectorArray(V.data[600:800,:]),modes=16)
# POD[4],_=pod(NumpyVectorArray(V.data[800:1000,:]),modes=16)
# POD[5],_=pod(NumpyVectorArray(V.data[1000:1200,:]),modes=16)
# POD[6],_=pod(NumpyVectorArray(V.data[1200:1400,:]),modes=16)
# POD[7],_=pod(NumpyVectorArray(V.data[1400:1600,:]),modes=16)
# POD[8],_=pod(NumpyVectorArray(V.data[1600:1800,:]),modes=16)
# POD[9],_=pod(NumpyVectorArray(V.data[1800:2000,:]),modes=16)
# #
# #
# #
# FT=np.ones((10,6))
#
# for i in range(10):
#     for j in range(9,15):
#         print('i={}, m={}'.format(i,j+1))
#         redsol,x_discr = fp_system(m=j+1,basis_type='RB',basis_pl_discr=(NumpyVectorArray(POD[i].data[0:j+1,:]),v_discr))
#         FT[i,j-9]=fperror(redsol,FPLoes=FPLoes)
#
# d=date.now()
# with open('Tabelle POD Loes Fehler m=10-15 {} je 200 snapshots.csv'.format(d.strftime("%y-%m-%d %H:%M:%S")),'w') as csvfile:
#     writer=csv.writer(csvfile)
#     writer.writerow(['m','Nr.0', 'Nr.1','Nr.2', 'Nr.3','Nr.4', 'Nr.5','Nr.6','Nr.7','Nr.8','Nr.9'])
#     for j in range(9,15):
#         writer.writerow([j+1,FT[0,j-9],FT[1,j-9],FT[2,j-9],FT[3,j-9],FT[4,j-9],FT[5,j-9],FT[6,j-9],FT[7,j-9],FT[8,j-9],FT[9,j-9]])
#
#
# POD=dict.fromkeys(range(10))
# # #
# # #
#
#V4=pickle.load(open("rb-daten 20000, Nr. 4 15-01-29 19:26:52.p",'rb'))
#
#
#
#
# #
# POD[0],_=pod(NumpyVectorArray(V4.data[0:2000,:]),modes=17)
# POD[1],_=pod(NumpyVectorArray(V4.data[2000:4000,:]),modes=17)
# POD[2],_=pod(NumpyVectorArray(V4.data[4000:6000,:]),modes=17)
# POD[3],_=pod(NumpyVectorArray(V4.data[6000:8000,:]),modes=17)
# POD[4],_=pod(NumpyVectorArray(V4.data[8000:10000,:]),modes=17)
# POD[5],_=pod(NumpyVectorArray(V4.data[10000:12000,:]),modes=17)
# POD[6],_=pod(NumpyVectorArray(V4.data[12000:14000,:]),modes=17)
# POD[7],_=pod(NumpyVectorArray(V4.data[14000:16000,:]),modes=17)
# POD[8],_=pod(NumpyVectorArray(V4.data[16000:18000,:]),modes=17)
#POD[9],_=pod(NumpyVectorArray(V4.data[18000:20000,:]),modes=17)
# V4=None
# #
# #
# #
# FT=np.ones((10,6))
#
# for i in range(10):
#     for j in range(9,15):
#         print('i={}, m={}'.format(i,j+1))
#         redsol,x_discr = fp_system(m=j+1,basis_type='RB',basis_pl_discr=(NumpyVectorArray(POD[i].data[0:j+1,:]),v_discr))
#         FT[i,j-9]=fperror(redsol,FPLoes=FPLoes)
#
# d=date.now()
# with open('Tabelle POD Loes Fehler m=10-15 {} je 2000 snapshots.csv'.format(d.strftime("%y-%m-%d %H:%M:%S")),'w') as csvfile:
#     writer=csv.writer(csvfile)
#     writer.writerow(['m','Nr.0', 'Nr.1','Nr.2', 'Nr.3','Nr.4', 'Nr.5','Nr.6','Nr.7','Nr.8','Nr.9'])
#     for j in range(9,15):
#         writer.writerow([j+1,FT[0,j-9],FT[1,j-9],FT[2,j-9],FT[3,j-9],FT[4,j-9],FT[5,j-9],FT[6,j-9],FT[7,j-9],FT[8,j-9],FT[9,j-9]])
#


#V=pickle.load(open("rb-daten 20000, Nr. 0 15-01-29 16:57:02.p",'rb'))
#print('Import ready.')
#V.append(pickle.load(open("rb-daten 20000, Nr. 1 15-01-29 17:34:45.p",'rb')))
#POD[1],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 2 15-01-29 18:11:45.p",'rb')))
#POD[2],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 3 15-01-29 18:49:19.p",'rb')))
#POD[3],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 4 15-01-29 19:26:52.p",'rb')))
#POD[4],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 5 15-01-29 20:06:57.p",'rb')))
#POD[5],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 6 15-01-29 20:44:31.p",'rb')))
#POD[6],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 7 15-01-29 21:22:19.p",'rb')))
#POD[7],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 8 15-01-29 22:00:46.p",'rb')))
#POD[8],_=pod(V,modes=20)
#V.append(pickle.load(open("rb-daten 20000, Nr. 9 15-01-29 22:39:03.p",'rb')))
#POD[9],_=pod(V,modes=20)

#print('Computing POD...')
#POD[0],_=pod(NumpyVectorArray(V.data),modes=17)
Vektoren1=NumpyVectorArray(dirich1)
Vektoren2=NumpyVectorArray(dirich2)
Vektoren3=NumpyVectorArray(dirich3)
Vektoren4=NumpyVectorArray(dirich4)
Vektorenk=NumpyVectorArray(dirichk)

#v_discr.visualize((Vektoren1,Vektoren2,Vektoren3,Vektoren4,Vektorenk), legend=('Vektoren1','Vektoren2','Vektoren3','Vektoren4','Vektorenk'))

Vektoren1.append(POD[0])
RB[0]=gram_schmidt(Vektoren1,product=v_discr.products['l2'],offset=0,check=True)

Vektoren2.append(POD[0])
RB[1]=gram_schmidt(Vektoren2,product=v_discr.products['l2'],offset=0,check=True)

Vektoren3.append(POD[0])
RB[2]=gram_schmidt(Vektoren3,product=v_discr.products['l2'],offset=0,check=True)

Vektoren4.append(POD[0])
RB[3]=gram_schmidt(Vektoren4,product=v_discr.products['l2'],offset=0,check=True)

Vektorenk.append(POD[0])
RB[4]=gram_schmidt(Vektorenk,product=v_discr.products['l2'],offset=0,check=True)

V=None

FT=np.ones((5,16))
T=np.ones((5,16))
with open('Tabelle POD Dirich Loes Fehler 20 Nr.0 snapshots neu.csv','w') as csvfile:
    writer=csv.writer(csvfile)
    writer.writerow([' ','m','p=0.1','p=0.01','p=0.001','p=0.001','peak'])

for j in range(16):
    for i in range(5):
        print('i={}, m={}'.format(i,j+1))
        tic=time.time()
        redsol,x_discr = fp_system(m=j+1,basis_type='RB',basis_pl_discr=(NumpyVectorArray(RB[i].data[0:j+1,:]),v_discr))
        FT[i,j]=fperror(redsol,FPLoes=FPLoes)
        print('Error is {}'.format(FT[i,j]))
        T[i,j]=time.time()-tic

    with open('Tabelle POD Dirich Loes Fehler 20 Nr.0 snapshots neu.csv','a') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Error',j+1,FT[0,j],FT[1,j],FT[2,j],FT[3,j],FT[4,j]])
        writer.writerow(['Time',j+1,T[0,j],T[1,j],T[2,j],T[3,j],T[4,j]])


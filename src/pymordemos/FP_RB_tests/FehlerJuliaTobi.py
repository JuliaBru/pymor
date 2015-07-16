__author__ = 'j_brun16'

import numpy as np
import csv
import math


LsgJulia=np.zeros((1000,50))

with open('SourceBeam 15-07-14 13:31:13 Leg m=6.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i=0
    for row in reader:
        for j in range(50):
            LsgJulia[i,j]=float(row[j])
        i+=1


LsgTobi=np.zeros((1000,50))

with open('SourceBeam 15-07-14 13:03:30 Leg m=6.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i=0
    for row in reader:
        for j in range(50):
            LsgTobi[i,j]=float(row[j])
        i+=1


def fperror(FP1,FP2):
    (nt,nx)=FP1.shape
    (ntf,nxf)=FP2.shape
    FP1neu=np.zeros((ntf,nxf))
    for i in range(nxf):
        for j in range(ntf):
            FP1neu[j,i]=FP1[math.floor(j*nt/ntf),math.floor(i*nx/nxf)]
        fehl=np.sum(np.abs(FP2-FP1neu))/np.sum(np.abs(FP2))
    return fehl

print('Relative L1 Error is {}'.format(fperror(LsgJulia,LsgTobi)))
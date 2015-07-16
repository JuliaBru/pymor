'''
Usage:
  fokkerplanck.py [-hp] [--grid=NI] [--grid-type=TYPE] [--problem=TYPE] [--CFLtype=TYPE] [--CFL=VALUE]
          [--num-flux=FLUX] [--m=COUNT] [--basis_type=TYPE]


Options:
  --grid=NI              Use grid with NI elements [default: 500].

  --grid-type=TYPE       Type of grid to use (oned) [default: oned].

  --problem=TYPE         Select the problem (2Beams, 2Pulses,SourceBeam, SourceBeamNeu,RectIC) [default: SourceBeam].

  --CFLtype=TYPE         Type of CFL to use (matlab,computed, given)  [default: given].

  --CFL=VALUE            Value to use instead of CFL condition [default: 0.65]

  --num-flux=FLUX        Numerical flux to use [default: godunov_upwind].

  --m=COUNT              Dimension of the system [default: 1].

  --basis_type=TYPE      Type of basis to use (Leg,RB,Picklen) [default: Leg].

  -h, --help             Show this message.
'''


from __future__ import absolute_import, division, print_function

import numpy as np
from pymor.core import getLogger
from pymor.discretizers.ellipticplus import discretize_elliptic_cg_plus
from pymor.analyticalproblems.fokkerplanck_rb import Fokkerplanck_V
import pickle
from pymor.la import NumpyVectorArray
from pymor.la.pod import pod
from datetime import datetime as date
import time
from docopt import docopt
from pymordemos.fokkerplanck import fp_demo
from pymor.la.gram_schmidt import gram_schmidt
import csv




getLogger('pymor.discretizations').setLevel('INFO')





args = docopt(__doc__)


print('Setup Problem ...')


problem=Fokkerplanck_V(problem='SourceBeam', delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))

print('Discretize ...')

n=250

discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)


FPLoes=np.zeros((1000,500))

with open('fploes.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    i=0
    for row in reader:
        for j in range(500):
            FPLoes[i,j]=float(row[j])
        i+=1

FPLoe=NumpyVectorArray(FPLoes)

MaxOrdn=1
sample=1
tic = time.time()
grid=discretization.visualizer.grid
B=dict.fromkeys(range(sample))
Basis=dict.fromkeys(range(sample))
mudict=dict.fromkeys(range(sample))
V=dict.fromkeys(range(sample))
relerror=np.zeros(sample)
#StartB=NumpyVectorArray(np.ones((1,501)))
StartB=NumpyVectorArray.empty(501)
np.random.seed()


for j in range(MaxOrdn):
    i=0
    print('j={}'.format(j))
    for mu in problem.parameter_space.sample_randomly(sample):
        print('i={}'.format(i))
        B[i]=NumpyVectorArray(StartB.data)

        #mu['P']=[[0.2185377409,],]
        #mu['dirich']=(0.0991554474,0.628695232)
        #mu['dtP']=[[3.2020662085,],]
        #mu['dxP']=[[-2.6471565543,],]
        #mu['qtpoint']=1.6904509098
        #mu['qxpoint']=1.0205852155

        mu['P']=[[0.5398402092,],]
        mu['dirich']=(0.1561631951,0.8413070333)
        mu['dtP']=[[1.9343058427,],]
        mu['dxP']=[[0.4521704698,],]
        mu['qtpoint']=3.1804549132
        mu['qxpoint']=1.8013002252



        #while True:
            #try:
        B[i].append(discretization.solve(mu))
            #    break
            #except:
            #    for mun in problem.parameter_space.sample_randomly(1):
            #       mu=mun

        Basis[i]=gram_schmidt(B[i],discretization.products['l2'])
        mudict[i]=mu

        while True:
            try:
                V[i],discr=fp_demo(args,Basis[i])
                break
            except ValueError:
                args['--CFL']*=0.75
                print('neue CFL={}'.format(args['--CFL']))
        relerror[i]=np.sum(np.abs(FPLoe.data-V[i].data))/np.sum(np.abs(FPLoe.data))
        i+=1
        args['--CFL']=0.65


    d=date.now()
    with open('Greedy-Tabelle {} m {}.csv'.format(d.strftime("%y-%m-%d %H:%M:%S"),j+1),'w') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerow(['Nr.', 'P','Dirich-1', 'Dirich1', 'dtP', 'dxP', 'qtpoint', 'qxpoint', 'relerror'])
        for i in range(sample):
            writer.writerow([i,
                            mudict[i]['P'][0,0],mudict[i]['dirich'][0],mudict[i]['dirich'][1],
                            mudict[i]['dtP'][0,0], mudict[i]['dxP'][0,0], mudict[i]['qtpoint'], mudict[i]['qxpoint'],
                            relerror[i]])



    imin=np.ma.argmin(relerror)
    pickle.dump((V[imin]) ,open( '{} Beste Lsg m {}.p'.format(d.strftime("%y-%m-%d %H:%M:%S"),j+2), "wb" ))
    pickle.dump((Basis[imin]) ,open( '{} Beste Basis m {}.p'.format(d.strftime("%y-%m-%d %H:%M:%S"),j+2), "wb" ))

    discr.visualize(V[imin])
    discretization.visualize(Basis[imin])
    print(relerror[imin])

    args['--m']+=1
    StartB=Basis[imin]



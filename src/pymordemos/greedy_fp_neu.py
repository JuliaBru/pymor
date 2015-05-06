'''
Usage:
  fokkerplanck.py [-hp] [--grid=NI] [--grid-type=TYPE] [--problem=TYPE] [--CFLtype=TYPE] [--CFL=VALUE]
          [--num-flux=FLUX] [--m=COUNT] [--basis_type=TYPE]


Options:
  --grid=NI              Use grid with NI elements [default: 50].

  --grid-type=TYPE       Type of grid to use (oned) [default: oned].

  --problem=TYPE         Select the problem (2Beams, 2Pulses,SourceBeam, SourceBeamNeu,RectIC) [default: SourceBeam].

  --CFLtype=TYPE         Type of CFL to use (matlab,computed, given)  [default: given].

  --CFL=VALUE            Value to use instead of CFL condition [default: 1]

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




def dg(mu1,mu2):
    d=(1./1.19*(mu1['P'][0,0]-mu2['P'][0,0]))**2
    d+= (1./6.3*(mu1['dxP'][0,0]-mu2['dxP'][0,0]))**2
    d+= (1./5.*(mu1['dtP'][0,0]-mu2['dtP'][0,0]))**2
    d+= 0*(1./4.*(mu1['qtpoint']-mu2['qtpoint']))**2
    d+= (1./3.*(mu1['qxpoint']-mu2['qxpoint']))**2
    d+= (mu1['dirich'][0]-mu2['dirich'][0])**2
    d+= (mu1['dirich'][1]-mu2['dirich'][1])**2
    return np.sqrt(d)

def d(mu1,mu2):
    d=((mu1['P'][0,0]-mu2['P'][0,0]))**2
    d+= ((mu1['dxP'][0,0]-mu2['dxP'][0,0]))**2
    d+= ((mu1['dtP'][0,0]-mu2['dtP'][0,0]))**2
    d+= ((mu1['qtpoint']-mu2['qtpoint']))**2
    d+= ((mu1['qxpoint']-mu2['qxpoint']))**2
    d+= (mu1['dirich'][0]-mu2['dirich'][0])**2
    d+= (mu1['dirich'][1]-mu2['dirich'][1])**2
    return np.sqrt(d)



def fperror(V,FPLoes):
    tic=time.time()
    (nt,nx)=V.data.shape
    if nt < 1000 or nx < 500:
        fpmitt=np.zeros((nt,nx))
        for i in range(nx):
            for j in range(nt):
                fpmitt[j,i]=np.mean(FPLoes[j*1000/nt:(j+1)*1000/nt,i*500/nx:(i+1)*500/nx ])
    else:
        fpmitt=FPLoes
    fehl=np.sum(np.abs(fpmitt-V.data))/np.sum(np.abs(fpmitt))
    print(time.time()-tic)
    return fehl





MaxOrdn=5
imax=3
sample=30




def greedy_fp(MaxOrdn,imax,sample,seed=None):

    np.random.seed(seed)

    StartB=dict.fromkeys(range(MaxOrdn))
    StartB[0]=NumpyVectorArray.empty(501)

    problem=Fokkerplanck_V(problem='SourceBeam', delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))

    n=250
    #grid=discretization.visualizer.grid
    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)
    args = docopt(__doc__)


    FPLoes=np.zeros((1000,500))

    with open('fploes.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in reader:
            for j in range(500):
                FPLoes[i,j]=float(row[j])
            i+=1


    for m_ind in range(MaxOrdn):
        args['--m']=m_ind+1
        B=dict.fromkeys(range(imax*sample))
        Basis=dict.fromkeys(range(imax*sample))
        mudict=dict.fromkeys(range(imax*sample))
        V=dict.fromkeys(range(imax*sample))
        relerror=np.ones(imax*sample)*10000.


        sample_ind=0
        i_ind=0
        print('m_ind={}'.format(m_ind))
        print('Berechnung erste Basisvektoren')

        for mu in problem.parameter_space.sample_randomly(sample):
            B[sample_ind]=NumpyVectorArray(StartB[m_ind].data)
            B[sample_ind].append(discretization.solve(mu))
            Basis[sample_ind]=gram_schmidt(B[sample_ind],discretization.products['l2'])
            mudict[sample_ind]=mu

            V[sample_ind],discr=fp_demo(args,Basis[sample_ind])

            #relerror[sample_ind]=np.sum(np.abs(FPLoe.data-V[sample_ind].data))/np.sum(np.abs(FPLoe.data))
            relerror[sample_ind]=fperror(V[sample_ind],FPLoes)
            sample_ind+=1
            args['--CFL']=0.65

        snapshot_min_ind=np.ma.argmin(relerror)


        for i_ind in range(1,imax):
            mudiff = np.zeros(i_ind*sample)

            #Suche Parameter in der Naehe des besten Parameters aus erster Iteration
            for sample_ind in range(i_ind*sample, (i_ind+1)*sample):

                while True:
                    for mu in problem.parameter_space.sample_randomly(1):
                        for test_ind in range(i_ind*sample):
                            mudiff[test_ind]=dg(mudict[test_ind],mu)
                        test_ind_min=np.ma.argmin(mudiff)

                    if test_ind_min == snapshot_min_ind:
                        mudict[sample_ind]=mu
                        break

            #Berechne Snapshots und Fehler fuer neue Parameter
            for sample_ind in range(i_ind*sample, (i_ind+1)*sample):
                mu=mudict[sample_ind]
                B[sample_ind]=NumpyVectorArray(StartB[m_ind].data)
                B[sample_ind].append(discretization.solve(mu))
                Basis[sample_ind]=gram_schmidt(B[sample_ind],discretization.products['l2'])

                while True:
                    try:
                        V[sample_ind],discr=fp_demo(args,Basis[sample_ind])
                        break
                    except ValueError:
                        args['--CFL']*=0.75
                        print('neue CFL={}'.format(args['--CFL']))

                #relerror[sample_ind]=np.sum(np.abs(FPLoe.data-V[sample_ind].data))/np.sum(np.abs(FPLoe.data))
                relerror[sample_ind]=fperror(V[sample_ind],FPLoes)
                args['--CFL']=0.65
                snapshot_min_ind=np.ma.argmin(relerror)

        #Auswahl der besten Basis fuer naechstes m_ind
        if False:
            StartB[m_ind+1]=Basis[np.argmin(relerror)]




        print('relerror m_ind={} : {}'.format(m_ind,relerror))

        args['--grid']=500
        while True:
            try:
                Vend,discr=fp_demo(args,Basis[np.argmin(relerror)])
                break
            except ValueError:
                args['--CFL']*=0.75
                print('neue CFL={}'.format(args['--CFL']))
        args['--grid']=50

        d=date.now()
        with open('Greedy-Error {} m={} imax={} seed={}.csv'.format(d.strftime("%y-%m-%d %H:%M:%S"),m_ind+1,imax,seed),'w') as csvfile:
            writer=csv.writer(csvfile)
            writer.writerow(['Nr.', 'P','Dirich-1', 'Dirich1', 'dtP', 'dxP', 'qtpoint', 'qxpoint', 'relerror'])
            for i in range(sample*imax):
                if i==np.argmin(relerror):
                    writer.writerow([i,
                            mudict[i]['P'][0,0],mudict[i]['dirich'][0],mudict[i]['dirich'][1],
                            mudict[i]['dtP'][0,0], mudict[i]['dxP'][0,0], mudict[i]['qtpoint'], mudict[i]['qxpoint'],
                            relerror[i],fperror(Vend,FPLoes)])
                else:
                    writer.writerow([i,
                            mudict[i]['P'][0,0],mudict[i]['dirich'][0],mudict[i]['dirich'][1],
                            mudict[i]['dtP'][0,0], mudict[i]['dxP'][0,0], mudict[i]['qtpoint'], mudict[i]['qxpoint'],
                            relerror[i]])

        pickle.dump((B,relerror) ,open('Snapshots {}.p'.format(d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))






    print('Echter Fehler: {}'.format(fperror(Vend,FPLoes)))

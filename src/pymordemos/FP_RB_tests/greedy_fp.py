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
from pymordemos.fokkerplanck import fp_system
from pymor.la.gram_schmidt import gram_schmidt
import csv




getLogger('pymor.discretizations').setLevel('INFO')




def fperror(V,FDRef):
    tic=time.time()
    (nt,nx)=V.data.shape
    if nt < 1000 or nx < 500:
        fpmitt=np.zeros((nt,nx))
        for i in range(nx):
            for j in range(nt):
                fpmitt[j,i]=np.mean(FDRef[j*1000/nt:(j+1)*1000/nt,i*500/nx:(i+1)*500/nx ])
    else:
        fpmitt=FDRef
    fehl=np.sum(np.abs(fpmitt-V.data))/np.sum(np.abs(fpmitt))
    print(time.time()-tic)
    return fehl

# Definition of proximity function
def proximity_func(mu1,mu2):
    d=(1./1.19*(mu1['P'][0,0]-mu2['P'][0,0]))**2
    d+= (1./6.3*(mu1['dxP'][0,0]-mu2['dxP'][0,0]))**2
    d+= (1./5.*(mu1['dtP'][0,0]-mu2['dtP'][0,0]))**2
    d+= 0*(1./4.*(mu1['qtpoint']-mu2['qtpoint']))**2
    d+= (1./2.*(mu1['qxpoint']-mu2['qxpoint']))**2
    d+= (mu1['dirich'][0]-mu2['dirich'][0])**2
    d+= (mu1['dirich'][1]-mu2['dirich'][1])**2
    return np.sqrt(d)


def greedy_fp(mmax,imax,sample,test_grid,seed=None,start_basis=None,save_snapshots=False,compute_real_error=False):



    StartB=dict.fromkeys(range(mmax+1))
    StartB[0]=NumpyVectorArray.empty(501)
    np.random.seed(seed)
    mstart=0

    if start_basis is not None:
        mstart=start_basis._len
        StartB[mstart]=start_basis

    problem=Fokkerplanck_V(test_case='SourceBeam', quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))

    Beste=dict.fromkeys(range(mmax))

    n=250
    discretization, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)



    FDRef=np.zeros((1000,500))

    with open('FD_reference_solution.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in reader:
            for j in range(500):
                FDRef[i,j]=float(row[j])
            i+=1


    for m_ind in range(mstart,mmax):
        m=m_ind+1
        B=dict.fromkeys(range(imax*sample))
        Basis=dict.fromkeys(range(imax*sample))
        mudict=dict.fromkeys(range(imax*sample))
        V=dict.fromkeys(range(imax*sample))
        relerror=np.ones(imax*sample)*10000.


        sample_ind=0
        i_ind=0
        print('m_ind={}'.format(m_ind))


        snapshot_min_ind=np.ma.argmin(relerror)


        for i_ind in range(imax):          
            
            #Sample random parameter values
            
            if i_ind == 0:
            #In first iteration random parameter values from the whole domain    
                for mu in problem.parameter_space.sample_randomly(sample):
                    mudict[sample_ind]=mu
                    sample_ind+=1
            else:
            #Search for parameter values near the best value from the first iterations
                mudiff = np.zeros(i_ind*sample)
                for sample_ind in range(i_ind*sample, (i_ind+1)*sample):
    
                    while True:
                        for mu in problem.parameter_space.sample_randomly(1):
                            for test_ind in range(i_ind*sample):
                                mudiff[test_ind]=proximity_func(mudict[test_ind],mu)
                            test_ind_min=np.ma.argmin(mudiff)
    
                        if test_ind_min == snapshot_min_ind:
                            mudict[sample_ind]=mu
                            break

            #Compute snapshots and errors for new parameter values
            for sample_ind in range(i_ind*sample, (i_ind+1)*sample):
                mu=mudict[sample_ind]
                B[sample_ind]=NumpyVectorArray(StartB[m_ind].data)
                B[sample_ind].append(discretization.solve(mu))
                Basis[sample_ind]=gram_schmidt(B[sample_ind],discretization.products['l2'])

                V[sample_ind],discr=fp_system(m=m,n_grid=test_grid,basis_type='RB',basis_pl_discr=(Basis[sample_ind],discretization))

                relerror[sample_ind]=fperror(V[sample_ind],FDRef)
        

        #Choice of best basis for next m_ind
        snapshot_min_ind=np.ma.argmin(relerror)
        StartB[m_ind+1]=Basis[snapshot_min_ind]


        print('relerror m_ind={} : {}'.format(m_ind,relerror))

        if compute_real_error == True:
            Vend,discr=fp_system(m=m,basis_type='RB',basis_pl_discr=(Basis[snapshot_min_ind],discretization))
            real_error=fperror(Vend,FDRef)
            print('Real error for best snapshot from parameter value {} is {}'.format(mudict[snapshot_min_ind],real_error))
        else: real_error=-1

        d=date.now()

        if save_snapshots == True:

            with open('Greedy-Beste {} m={} imax={} snapshots={} grid={} seed={}.csv'.format(d.strftime("%y-%m-%d %H:%M:%S"),m_ind+1,imax,sample,test_grid,seed),'w') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(['P','Dirich-1', 'Dirich1', 'dtP', 'dxP', 'qtpoint', 'qxpoint', 'estimatedrelerror','realrelerror'])
                i=snapshot_min_ind
                writer.writerow([mudict[i]['P'][0,0],mudict[i]['dirich'][0],mudict[i]['dirich'][1],
                                 mudict[i]['dtP'][0,0], mudict[i]['dxP'][0,0], mudict[i]['qtpoint'], mudict[i]['qxpoint'],
                                 relerror[i],real_error])

            with open('Greedy-Error {} m={} imax={} snapshots={} grid={} seed={}.csv'.format(d.strftime("%y-%m-%d %H:%M:%S"),m_ind+1,imax,sample,test_grid,seed),'w') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(['Nr.', 'P','Dirich-1', 'Dirich1', 'dtP', 'dxP', 'qtpoint', 'qxpoint', 'relerror'])
                for i in range(sample*imax):
                    if i==np.argmin(relerror):
                        writer.writerow([i,
                                         mudict[i]['P'][0,0],mudict[i]['dirich'][0],mudict[i]['dirich'][1],
                                         mudict[i]['dtP'][0,0], mudict[i]['dxP'][0,0], mudict[i]['qtpoint'], mudict[i]['qxpoint'],
                                         relerror[i],real_error])
                    else:
                        writer.writerow([i,
                                         mudict[i]['P'][0,0],mudict[i]['dirich'][0],mudict[i]['dirich'][1],
                                         mudict[i]['dtP'][0,0], mudict[i]['dxP'][0,0], mudict[i]['qtpoint'], mudict[i]['qxpoint'],
                                         relerror[i]])

            pickle.dump(mudict,open('mudict for m={}, sample={}, seed={} {}.p'.format(m, sample, seed, d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))
            pickle.dump(B,open('snapshots for m={}, sample={}, seed={} {}.p'.format(m, sample, seed, d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))



    return StartB[mmax],discretization





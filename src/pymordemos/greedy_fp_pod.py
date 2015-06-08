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
from pymor.parameters import CubicParameterSpace





getLogger('pymor.discretizations').setLevel('INFO')




def dg(mu1,mu2):
    d=(1./1.19*(mu1['P'][0,0]-mu2['P'][0,0]))**2
    d+= (1./6.3*(mu1['dxP'][0,0]-mu2['dxP'][0,0]))**2
    d+= (1./5.*(mu1['dtP'][0,0]-mu2['dtP'][0,0]))**2
    d+= (1./2.*(mu1['qxpoint']-mu2['qxpoint']))**2
    d+= (mu1['dirich'][0]-mu2['dirich'][0])**2
    d+= (mu1['dirich'][1]-mu2['dirich'][1])**2
    return d

def d(mu1,mu2):
    d=((mu1['P'][0,0]-mu2['P'][0,0]))**2
    d+= ((mu1['dxP'][0,0]-mu2['dxP'][0,0]))**2
    d+= ((mu1['dtP'][0,0]-mu2['dtP'][0,0]))**2
    d+= ((mu1['qtpoint']-mu2['qtpoint']))**2
    d+= ((mu1['qxpoint']-mu2['qxpoint']))**2
    d+= (mu1['dirich'][0]-mu2['dirich'][0])**2
    d+= (mu1['dirich'][1]-mu2['dirich'][1])**2
    return d



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





MaxOrdn=2
imax=1
sample=5







def greedy_fp(mmax,imax,sample,test_grid,seed=None):

    np.random.seed(seed)

    StartB=dict.fromkeys(range(MaxOrdn))
    StartB[0]=NumpyVectorArray.empty(501)

    problem=Fokkerplanck_V(problem='SourceBeam', delta=0, quadrature_count=(1,1),P_parameter_range=(0.01,1.2),
                           dxP_parameter_range=(-5.4,0.9),dtP_parameter_range=(0,5))

    n=250
    #grid=discretization.visualizer.grid
    mu_discr, _ = discretize_elliptic_cg_plus(problem, diameter=1 / n)
    args = docopt(__doc__)


    FPLoes=np.zeros((1000,500))

    with open('fploes.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        i=0
        for row in reader:
            for j in range(500):
                FPLoes[i,j]=float(row[j])
            i+=1

    B=NumpyVectorArray.empty(mu_discr.dim_solution,reserve=imax*sample*mmax)
    #Basis=dict.fromkeys(range(imax*sample*mmax))
    mudict=dict.fromkeys(range(imax*sample*mmax))
    V=dict.fromkeys(range(imax*sample*mmax))
    relerror=dict.fromkeys(range(mmax))
    FehlerGes=np.zeros(mmax)




    print('Berechnung erste Basisvektoren')

    snapshot_ind=0
    for mu in problem.parameter_space.sample_randomly(sample):
        snapshot=mu_discr.solve(mu)
        B.append(snapshot)
        mudict[snapshot_ind]=mu
        snapshot_ind+=1



    for m_ind in range(mmax):
        m=m_ind+1
        relerror[m_ind]=np.ones(sample*(imax*(m_ind+1)+1))*10000

        #Fehler bisherige Snapshots mit bestehender Basis
        for ind in range(snapshot_ind):
            Basis=NumpyVectorArray(StartB[m_ind].data)
            Basis.append(NumpyVectorArray(B.data[ind,:]))
            Basis=gram_schmidt(Basis,mu_discr.products['l2'])


            print('m_ind={}, Fehler bisherige, ind={}'.format(m_ind,ind))
            V,fp_discr=fp_system(m=m,basis_type='RB',n_grid=test_grid,basis_pl_discr=(Basis,mu_discr))


            relerror[m_ind][ind]=fperror(V,FPLoes)


        snapshot_min_ind=np.ma.argmin(relerror[m_ind])


        for i_ind in range(imax):
            mudiff = np.zeros(snapshot_ind)

            s_now=snapshot_ind

            #Suche Parameter in der Naehe des bisher besten Parameters

            para_space=problem.parameter_space
            xi=0.8

            for ind in range(sample):
                print('Suche neue Parameter, m_ind={}, i_ind={}, ind={}'.format(m_ind,i_ind,ind))
                i=0

                while True:
                    if i >= 1000:
                        ranges_old=para_space.ranges
                        ranges_new= {'P': (float(ranges_old['P'][0] + (mudict[snapshot_min_ind]['P'][0] - ranges_old['P'][0])*xi),
                                                float(ranges_old['P'][1] - (ranges_old['P'][1] - mudict[snapshot_min_ind]['P'])*xi)),
                                        'dxP':(float(ranges_old['dxP'][0] + (mudict[snapshot_min_ind]['dxP'] - ranges_old['dxP'][0])*xi),
                                                float(ranges_old['dxP'][1] - (ranges_old['dxP'][1] - mudict[snapshot_min_ind]['dxP'])*xi)) ,
                                        'dtP':(float(ranges_old['dtP'][0] + (mudict[snapshot_min_ind]['dtP'] - ranges_old['dtP'][0])*xi),
                                                float(ranges_old['dtP'][1] - (ranges_old['dtP'][1] - mudict[snapshot_min_ind]['dtP'])*xi)),
                                        'dirich': (float(ranges_old['dirich'][0] + (min(mudict[snapshot_min_ind]['dirich']) - ranges_old['dirich'][0])*xi),
                                                float(ranges_old['dirich'][1] - (ranges_old['dirich'][1] - max(mudict[snapshot_min_ind]['dirich']))*xi)),
                                        'qxpoint':(float(ranges_old['qxpoint'][0] + (mudict[snapshot_min_ind]['qxpoint'] - ranges_old['qxpoint'][0])*xi),
                                                float(ranges_old['qxpoint'][1] - (ranges_old['qxpoint'][1] - mudict[snapshot_min_ind]['qxpoint'])*xi)) ,
                                        'qtpoint':ranges_old['qtpoint']}
                        para_space=CubicParameterSpace({'P': (1, 1),
                                               'dxP':(1, 1),
                                               'dtP':(1, 1),
                                               'dirich':(2),
                                               'qxpoint':0,
                                               'qtpoint':0},
                                              ranges=ranges_new)
                        print('new parameter range={}'.format(ranges_new))
                        i=0


                    for mu in para_space.sample_randomly(1):
                        for test_ind in range(s_now):
                            mudiff[test_ind]=dg(mudict[test_ind],mu)
                        test_ind_min=np.ma.argmin(mudiff)

                    if test_ind_min == snapshot_min_ind:
                        mudict[snapshot_ind]=mu
                        snapshot_ind+=1
                        break
                    else:
                        i+=1

                snapshot=mu_discr.solve(mu)
                B.append(snapshot)

                #Berechne Fehler fuer neuen Snapshot
                Basis=NumpyVectorArray(StartB[m_ind].data)
                Basis.append(snapshot)
                Basis=gram_schmidt(Basis,mu_discr.products['l2'])


                print('m_ind={}, Fehler neue, i_ind={}, ind={}'.format(m_ind,i_ind,ind))
                V,fp_discr=fp_system(m=m,basis_type='RB',n_grid=test_grid,basis_pl_discr=(Basis,mu_discr))

                relerror[m_ind][snapshot_ind-1]=fperror(V,FPLoes)
                args['--CFL']=0.65
            snapshot_min_ind=np.ma.argmin(relerror)



        #POD
        StartB[m_ind+1],_=pod(B,modes=m_ind+1,product=mu_discr.products['l2'])


    if True:
        index_tuple=[]
        med=np.median(relerror[m_ind])
        for i in range(relerror[m_ind].shape[0]):
            if relerror[m_ind][i]<=med:
                index_tuple.append(i)
        B_beste=NumpyVectorArray(B.data[index_tuple])
    if False:
        B_beste=NumpyVectorArray(B.data)


    FinalPOD,SV =pod(B_beste,modes=None,product=mu_discr.products['l2'])
    PODerrors=np.zeros(20)
    datenow=date.now()
    for k in range(np.min([20,SV.shape[0]])):
        PODerrors[k]=np.sqrt(np.sum(SV[k+1:SV.shape[0]]))
        print('Error of POD space Y_{} is {}'.format(k+1,PODerrors[k]))
    pickle.dump((FinalPOD,PODerrors),open( 'POD Space m_max={} i_max={} seed={} {}.p'.format(mmax,imax,seed,datenow.strftime("%y-%m-%d %H:%M:%S")), "wb" ))


    # args['--grid']=500
    # print('m_ind={}, Berechnung fertige Loesung')
    # Vend,fp_discr=fp_demo(args,(StartB[m_ind+1],mu_discr))
    #
    # args['--grid']=50
    #
    #
    # FehlerGes[m_ind]=fperror(Vend,FPLoes)
    # fp_discr.visualize(Vend,title='m={}, Fehler={}'.format(m_ind,FehlerGes[m_ind]))
    #
    #
    #
    # d=date.now()
    # with open('POD {} mmax={} imax={} sample={} seed={}.csv'.format(d.strftime("%y-%m-%d %H:%M:%S"),mmax,imax,sample,seed),'w') as csvfile:
    #     writer=csv.writer(csvfile)
    #     writer.writerow(['Ordnung','RelFehler'])
    #     for i in range(mmax):
    #         writer.writerow([i,FehlerGes[i]])

        #
        #pickle.dump((B,relerror) ,open('Snapshots {}.p'.format(d.strftime("%y-%m-%d %H:%M:%S")), "wb" ))






    #print('Echter Fehler: {}'.format(fperror(Vend,FPLoes)))

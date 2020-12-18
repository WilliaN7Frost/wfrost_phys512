# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:46:17 2020

@author: wilia
"""

import numpy as np

from matplotlib  import pyplot as plt
#import matplotlib.cm as cmx
#import matplotlib.colors as colors
#from mpl_toolkits.mplot3d import Axes3D

import particles as p

import os


if __name__=='__main__':
    
    n=100000
    oversamp=5
    bound = 'non-periodic'
    dt = 0.001
    soft = None
    save = True
    dim = 500
    
    if soft==None: soft_str = 'auto'
    else:
        soft_str = list(str(soft)); ip = soft_str.index('.')
        soft_str[ip] = 'p';  soft_str = ''.join(soft_str)
    dt_str = list(str(dt)); ip = dt_str.index('.')
    dt_str[ip] = 'p';  dt_str = ''.join(dt_str)
        
    part=p.particles2D( bound=bound , m=np.full(n,1.0) , npart=n , dt=dt , soft=soft
                       #, x=np.array([53]) , y=np.array([56])
                       #, vx=np.array([0]) , vy=np.array([0])
                       #, x=np.array([45,55]) , y=np.array([50,50])
                       #, vx=np.array([0,0]) , vy=np.array([-1,1])
                       , xrange=[0,100],yrange=[0,100]
                       , grid_size=(dim,dim)                      )
    
    directTV = '2D_n'+str(n)+'_dt'+str(dt_str)+'_soft'+soft_str+'_dim'+str(dim)+'_'+bound+'_ln(dist)'
    
    if save:
        if not os.path.exists(directTV):
            os.makedirs(directTV)
        else:
            raise Exception('The directory already exists')
    
    totE = []; kinE = []; potE = []
    for i in range(200):
        
        if (i+1)%20==0: print('Commencing '+str(i+1)+'th run')
        
        # Plots figures as the system evolves
        plt.figure(); plt.title( 'step '+str(round(i*oversamp,2))+' ; t='+str(round(dt*i*oversamp,2)) )
        plt.scatter(part.x, part.y , s=0.1)
        plt.xlim(part.xrange[0],part.xrange[1])
        plt.ylim(part.yrange[0],part.yrange[1])
        plt.xlabel('X-axis'); plt.ylabel('Y-axis')
        if save: plt.savefig(directTV+'/'+str(i)+'.png')
        plt.close()
        
        plt.pause(0.001)
        for j in range(oversamp):
            # Makes the simulation take a step
            part.evolve( start=(True if (i==0 and j==0) else False) , doPlot=False)#(True if ((i==0 or i==30) and j==0) else False) )
            totE.append((part.potE+part.kinE-part.totalE)/part.totalE * 10e2)
            kinE.append(part.kinE);  potE.append(part.potE)
        print('Current total energy is',part.potE+part.kinE)
    
    # Plots the final state of the system
    plt.figure(); plt.title( 'step '+str(round((i+1)*oversamp,2))+' ; t='+str(round(dt*i*oversamp,2)) )
    plt.scatter(part.x, part.y , s=0.1)
    plt.xlim(part.xrange[0],part.xrange[1])
    plt.ylim(part.yrange[0],part.yrange[1])
    plt.xlabel('X-axis'); plt.ylabel('Y-axis')
    if save: plt.savefig(directTV+'/end.png')
    plt.close()
    
    # Plots the energy evolution over time
    plt.figure(); plt.suptitle('Deviation from original total energy')
    plt.subplot(3,1,1)
    plt.plot(kinE, label='kinE', c='r')
    plt.ylabel('Energy'); plt.legend()
    plt.subplot(3,1,2)
    plt.plot(potE, label='potE' ,c='blue')
    plt.ylabel('Energy'); plt.legend()
    plt.subplot(3,1,3)
    plt.plot(totE)
    plt.xlabel('Steps (at each '+str(oversamp)+'th iteration)')
    plt.ylabel('delta_E (%)')
    if save: plt.savefig(directTV+'/energy_evol.png')
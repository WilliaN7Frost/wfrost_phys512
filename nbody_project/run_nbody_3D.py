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
    
    n=2
    oversamp=5
    bound = 'non-periodic'
    dt = 0.03
    soft = None
    save = True
    dim = 200
    
    if soft==None: soft_str = 'auto'
    else:
        soft_str = list(str(soft)); ip = soft_str.index('.')
        soft_str[ip] = 'p';  soft_str = ''.join(soft_str)
    if dt >= 0.0001:
        dt_str = list(str(dt)); ip = dt_str.index('.')
        dt_str[ip] = 'p';  dt_str = ''.join(dt_str)
    else:
        dt_str = str(dt)
        
    part=p.particles3D( bound=bound , m=np.full(n,1.0) , npart=n , dt=dt , soft=soft
                       #, x=np.array([55]) , y=np.array([45]) , z=np.array([52] )
                       #, vx=np.array([0]) , vy=np.array([0]) , vz=np.array([0])
                       , x=np.array([45,55]) , y=np.array([55,45]) , z=np.array([48,52] )
                       , vx=np.array([0,0]) , vy=np.array([-0.6,0.6]) , vz=np.array([-0.6,0.6])
                       , xrange=[0,100],yrange=[0,100],zrange=[0,100]
                       , grid_size=(dim,dim,dim)
                       )
    
    directTV = '3D_n'+str(n)+'_dt'+str(dt_str)+'_soft'+soft_str+'_dim'+str(dim)+'_'+bound
    
    if save:
        if not os.path.exists(directTV):
            os.makedirs(directTV)
        else:
            raise Exception('The directory already exists')
    
    totE = []; kinE = []; potE = []
    steps = 40
    elevation = 70
    rot_range = [0,10]
    angles = np.linspace(rot_range[0],rot_range[1],steps)
    for i in range(steps):
        
        if (i+1)%20==0: print('Commencing '+str(i+1)+'th run')
        
        # Plots figures as the system evolves
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        ax.set_title( 'step '+str(round(i*oversamp,2))+' ; t='+str(round(dt*i*oversamp,2)) )
        ax.scatter(part.x, part.y , part.z , color="royalblue",marker=".",s=2,alpha=1.0)
        ax.set_xlim(part.xrange[0],part.xrange[1])
        ax.set_ylim(part.yrange[0],part.yrange[1])
        ax.set_zlim(part.zrange[0],part.zrange[1])
        ax.set_xlabel('X-axis'); ax.set_ylabel('Y-axis'); ax.set_zlabel('Z-axis')
        ax.view_init(elevation,angles[i])
        if save: plt.savefig(directTV+'/'+str(i)+'.png')
        plt.close()
        
        plt.pause(0.001)
        for j in range(oversamp):
            # Makes the simulation take a step
            part.evolve( start=(True if (i==0 and j==0) else False) , doPlot=False)#(True if ((i==0 or i==30) and j==0) else False) )
            totE.append((part.potE+part.kinE-part.totalE)/part.totalE * 10e2)
            kinE.append(part.kinE);  potE.append(part.potE)
        print('Current total energy is',part.potE+part.kinE)
    
    # Plots the last figure
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.set_title( 'step '+str(round(i*oversamp,2))+' ; t='+str(round(dt*i*oversamp,2)) )
    ax.scatter(part.x, part.y , part.z , color="royalblue",marker=".",s=2,alpha=1.0)
    ax.set_xlim(part.xrange[0],part.xrange[1])
    ax.set_ylim(part.yrange[0],part.yrange[1])
    ax.set_zlim(part.zrange[0],part.zrange[1])
    ax.set_xlabel('X-axis'); ax.set_ylabel('Y-axis'); ax.set_zlabel('Z-axis')
    ax.view_init(20,300)
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
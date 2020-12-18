# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 00:13:35 2020

@author: wilia
"""
import numpy as np
from matplotlib  import pyplot as plt

# Used in the 3D case
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
# Used for plotting 3D densities, potentials and force streams. Not ideal for high grid size
def scatter3d(x,y,z, cs, colorsMap='jet' , title='' , axlabs=['X-axis','Y-axis','Z-axis'] , axlims=[] , alpha=1.0):
    cm = plt.get_cmap(colorsMap)
    cNorm = colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs) , alpha=alpha)
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    ax.set_title(title)
    ax.set_xlabel(axlabs[0]); ax.set_ylabel(axlabs[1]); ax.set_zlabel(axlabs[2])
    if len(axlims)!=0:
        ax.set_xlim(axlims[2][0],axlims[2][1]); ax.set_ylim(axlims[1][0],axlims[1][1]); ax.set_zlim(axlims[0][0],axlims[0][1])
    plt.show()




class particles2D:
    def __init__(  self , bound='periodic'
                 , m=[] , npart=1000,soft=None,G=1.0,dt=0.1
                 , xrange = [0,1] , yrange = [0,1]           # the extent of each dimension in units of distance
                 , x=[],y=[]
                 , vx=[],vy=[]
                 , grid_size=(100,100)                       # (x_size,y_size)
                ):
        
        self.opts = {}
        self.opts['bound'] = bound    # Determines boundry conditions
        if soft != None:
            self.opts['soft'] = soft  # Determines softening radius for potential
        self.opts['n'] = npart        # Amount of particles
        self.opts['G'] = G            # Gravitational constant
        self.opts['dt'] = dt          # Time step of our simulation
        
        # Masses are default set to 1.0 for all particles. Else use supplied ones
        if len(m)==0: self.m = np.full(self.opts['n'] , 1.0)
        else:         self.m = m
        
        # By default:
        #   Dimension ranges assume that both bounding values are obtainable
        #   Initial particle positions are uniform random.
        #   Initial speeds are set to zero.
        if len(x)==0:  self.x = np.random.uniform(low=xrange[0],high=xrange[1],size=self.opts['n'])
        else:          self.x = x.copy()
        if len(y)==0:  self.y = np.random.uniform(low=yrange[0],high=yrange[1],size=self.opts['n'])
        else:          self.y = y.copy()
        if len(vx)==0: self.vx = np.zeros(self.opts['n'])
        else:          self.vx = vx.copy()
        if len(vy)==0: self.vy = np.zeros(self.opts['n'])
        else:          self.vy = vy.copy()
        
        # Initialize forces arrays for each dimension
        self.fx = np.zeros(self.opts['n'])
        self.fy = self.fx.copy()
        
        #Initialize energy for the system
        self.potE = 0
        self.kinE = 0
        self.totalE = 0
        
        # Saving the dimension ranges
        self.xrange = xrange;  self.yrange = yrange
        
        # Grid values for x,y,z of the density grid
        # Grid size for each dimension needs to be even
        self.grid_size = grid_size
        self.grid_x = np.linspace( xrange[0] , xrange[1] , grid_size[1] )
        self.grid_rez_x = self.grid_x[1] - self.grid_x[0]
        self.grid_y = np.linspace( yrange[0] , yrange[1] , grid_size[0] )
        self.grid_rez_y = self.grid_y[1] - self.grid_y[0]
        
        # If the softening was left unspecified, set it using the grid resolution
        if soft==None: self.opts['soft'] = 2*np.max([ self.grid_rez_x , self.grid_rez_y ])
        
        # Initializes the (softened) Green's Function convolution kernel centered at (x,y)=(0,0)
        # If bound conditions are periodic, kernel is same shape as the density
        # Else kernel is double the size of the density in each dimension (to have zero-padding at boundries)
        x_interval = xrange[1]-xrange[0];   y_interval = yrange[1]-yrange[0]
        if bound=='periodic':
            grid_xx, grid_yy = np.meshgrid(  np.linspace( -x_interval/2 , x_interval/2 , len(self.grid_x) )
                                           , np.linspace( -y_interval/2 , y_interval/2 , len(self.grid_y) )  )
        elif bound=='non-periodic':
            grid_xx, grid_yy = np.meshgrid(  np.linspace( -x_interval , x_interval , int( len(self.grid_x)*2 + grid_size[0]%2) )
                                           , np.linspace( -y_interval , y_interval , int( len(self.grid_y)*2 + grid_size[1]%2) )  )  
        else:
            raise Exception('Illegal boundary condition was requested. Only periodic or non-periodic are supported')
        
        # Creating the Green's function of the Laplacian which solves for the potential using the density grid
        # Using the Green's function for 2D distances ( ln(dist)/2pi ) taken from
        # https://ocw.mit.edu/courses/mathematics/18-303-linear-partial-differential-equations-fall-2006/lecture-notes/greensfn.pdf
        dist = np.sqrt( grid_xx**2 + grid_yy**2 )
        """
        self.pot_conv_kern = -1/(2*np.pi*dist)
        self.pot_conv_kern = np.where( dist <= self.opts['soft'] ,  -1/(2*np.pi*self.opts['soft']) , self.pot_conv_kern )
        """
        self.pot_conv_kern = np.log(dist)/(2*np.pi)
        self.pot_conv_kern = np.where( dist <= self.opts['soft'] ,  np.log(self.opts['soft'])/(2*np.pi) , self.pot_conv_kern )
        grid_xx=None; grid_yy=None; dist=None
    
    
    
    
    
    def get_forces_mui(self , return_pot=False , doPlot=False):
        
        # TODO
        # 1) create density grid based on particle positions and the grid resolution   (DONE)
        # 2) convolve density with potential convolution kernel                        (DONE)
        # 3) use gradient of potential to get force map in (x,y)                       (DONE)

        density_grid_2d = np.zeros((self.grid_size))
        
        # Keep track of the indices the particles fall onto. Useful when assigning forces to particles
        indices = []
        for i in range(self.opts['n']):
            # Assign grid positions based on the Nearest-Grid-Point method
            xind = (np.abs(self.x[i] - self.grid_x)).argmin()
            yind = (np.abs(self.y[i] - self.grid_y)).argmin()
            density_grid_2d[xind,yind] += self.m[i]
            indices.append([xind,yind])
        # Multiply the density grid by the area contained in a cell to get actual density and not just mass
        density_grid_2d /= self.grid_rez_x*self.grid_rez_y
        
        
        # These doPlot booleans decide if density, kenel, potential and/or force plots
        # are generated to check if the simulation behaves as expected. In 2D, need to transpose
        # the arrays otherwise plt.imshow() will align x values with the y-axis and vice-versa.
        if doPlot:
            
            if self.opts['bound']=='periodic':
                x_interval = self.xrange[1]-self.xrange[0];   y_interval = self.yrange[1]-self.yrange[0]
            elif self.opts['bound']=='non-periodic':
                x_interval = (self.xrange[1]-self.xrange[0])/2;   y_interval = (self.yrange[1]-self.yrange[0])/2
            extent_field = [self.xrange[0],self.xrange[1],self.yrange[0],self.yrange[1]]
            extent_kern = [ -x_interval/2,x_interval/2 , -y_interval/2,y_interval/2 ]
            
            plt.figure()
            plt.subplot(1,2,1); plt.title('Density grid initial')
            plt.imshow(np.transpose(density_grid_2d) , origin='lower' , extent=extent_field)
            plt.xlabel('X'); plt.ylabel('Y')
            plt.subplot(1,2,2); plt.title('1/r kernel')
            plt.imshow(np.transpose(self.pot_conv_kern) , origin='lower' , extent=extent_kern)
            plt.xlabel('X'); plt.ylabel('Y')
            
        
        # Now we convolve the density with the Green's Function
        # Need to fft_shift the result so that the potential is aligned according to our particle positions
        if self.opts['bound']=='periodic':
            # If the bounds are periodic, no special treatment is needed
            potential = (4*np.pi*self.opts['G']) * np.fft.irfftn( np.fft.rfftn(density_grid_2d) * np.fft.rfftn(self.pot_conv_kern) )
            for i in range(len(self.grid_size)):
                potential = np.fft.fftshift(potential,axes=i)
            
        elif self.opts['bound']=='non-periodic':
            # If bounds are non-periodic, add a bunch of zeropadding around the original density grid
            # The Green's function kernel is adjusted according to this new size in the init phase
            # Once the convolution is performed, simply retrieve the section containing the original grid
            dims = density_grid_2d.shape
            potential = np.zeros(dims)
            zeropad_density = np.zeros(np.array(dims)*2)
            
            zeropad_density[ int(dims[0]/2):-int(dims[0]/2)-(dims[0]%2) , int(dims[1]/2):-int(dims[1]/2)-(dims[0]%2) ] = density_grid_2d
            potential_padded = np.fft.irfftn( np.fft.rfftn(zeropad_density) * np.fft.rfftn(self.pot_conv_kern) )
            for i in range(len(self.grid_size)):
                potential_padded = np.fft.fftshift(potential_padded,axes=i)
            
            if doPlot:
                plt.figure()
                plt.imshow(np.transpose(potential_padded) , origin='lower')
            
            potential = potential_padded[ int(dims[0]/2):-int(dims[0]/2)-(dims[0]%2) , int(dims[1]/2):-int(dims[1]/2)-(dims[0]%2) ]
            potential_padded = None
            zeropad_density = None
            
        
        # Assign forces by taking the gradient of the potential field
        # Take negative since Force = -gradient(Potential)
        x_force_grid, y_force_grid = np.gradient( potential , self.grid_x, self.grid_y )
        x_force_grid *= -1;  y_force_grid *= -1
        
        if doPlot:
            plt.figure()
            plt.subplot(2,2,1); plt.title('potenial field')
            plt.imshow(np.transpose(potential) , origin='lower' , extent=extent_field )
            #plt.subplot(2,2,2); plt.title('force lines')
            plt.subplot(2,2,2); plt.title('force field')
            plt.imshow(np.transpose(np.sqrt(x_force_grid**2+y_force_grid**2)) , origin='lower' , extent=extent_field)
            for i in range(self.opts['n']):
                plt.arrow(self.x[i] , self.y[i] , x_force_grid[tuple(indices[i])] , y_force_grid[tuple(indices[i])] )
            plt.quiver(self.grid_y, self.grid_x, np.transpose(x_force_grid), np.transpose(y_force_grid))
            plt.subplot(2,2,3); plt.title('gx')
            plt.imshow(np.transpose(x_force_grid) , origin='lower' , extent=extent_field)
            plt.subplot(2,2,4); plt.title('gy')
            plt.imshow(np.transpose(y_force_grid) , origin='lower' , extent=extent_field)
        
        # For each particle, make it feel a force based on its position in the force grid
        x_force = np.zeros(self.opts['n'])
        y_force = np.zeros(self.opts['n'])
        for i in range(self.opts['n']):
            x_force[i] = x_force_grid[tuple(indices[i])]
            y_force[i] = y_force_grid[tuple(indices[i])]
        
        if return_pot:                 # This is an attempt at calculating the potential
            return x_force , y_force , -0.5*np.sum(potential)*np.sum(self.m) * (self.grid_rez_x*self.grid_rez_y)
        else:
            return x_force , y_force
        
    
    
    # Implements leapfrog technique for updating positions and velocities
    # Inspired from the wikipedia page on leapfrogging
    def evolve(self, start=False , doPlot=False):
        
        # If start=True, calculate forces, energy at initial positions
        if start:
            self.fx , self.fy , self.potE = self.get_forces_mui( return_pot=True , doPlot=doPlot )
            self.kinE = 0.5*np.sum(self.m*(self.vx**2+self.vy**2))
            self.totalE = self.kinE+self.potE
            print('Initial kinetic energy is',self.kinE)
            print('Initial total energy is',self.totalE,'\n')
        
        # Calculate next time-step position
        # If particle falls out of grid dimensions, make it enter at the opposite end
        # Regardless of boundry conditions, this is how out-of-bound particles are treated
        self.x = self.x + self.vx*self.opts['dt']  +  0.5*self.fx*self.opts['dt']**2 / self.m
        self.y = self.y + self.vy*self.opts['dt']  +  0.5*self.fy*self.opts['dt']**2 / self.m
        for i in range(self.opts['n']):
            if self.x[i] < self.xrange[0]:
                self.x[i] = self.xrange[1] - ( (self.xrange[0]-self.x[i]) % (self.xrange[1]-self.xrange[0]) )
            elif self.x[i] >= self.xrange[1]:
                self.x[i] = self.xrange[0] + ( (self.x[i]-self.xrange[1]) % (self.xrange[1]-self.xrange[0]) )
            if self.y[i] < self.yrange[0]:
                self.y[i] = self.yrange[1] - ( (self.yrange[0]-self.y[i]) % (self.yrange[1]-self.yrange[0]) )
            elif self.y[i] >= self.yrange[1]:
                self.y[i] = self.yrange[0] + ( (self.y[i]-self.yrange[1]) % (self.yrange[1]-self.yrange[0]) )
            
        
        # Calculate forces from updated positions. Calculate potential energy as well
        fx_next , fy_next , self.potE = self.get_forces_mui(return_pot=True , doPlot=doPlot)
        
        # Calculate next time-step velocities
        self.vx = self.vx  +  0.5*(self.fx+fx_next)*self.opts['dt'] / self.m
        self.vy = self.vy  +  0.5*(self.fy+fy_next)*self.opts['dt'] / self.m
        
        # Officially update forces
        self.fx = fx_next;  self.fy = fy_next
        
        # Update kinetic energy
        self.kinE=0.5*np.sum(self.m*(self.vx**2+self.vy**2))












# This class performs the exact same things as the 2d-particle class
# The main difference being the extra dimension and a different Green's Function 
class particles3D:
    def __init__(  self , bound='periodic'
                 , m=[] , npart=1000,soft=None,G=1.0,dt=0.1
                 , xrange = [0,1] , yrange = [0,1] , zrange=[0,1] # the extent of each dimension in units of distance
                 , x=[],y=[],z=[]
                 , vx=[],vy=[],vz=[]
                 , grid_size=(100,100,100) # (x_size,y_size,z_size)
                ):
        
        self.opts = {}
        self.opts['bound'] = bound    # Determines boundry conditions
        if soft != None:
            self.opts['soft'] = soft  # Determines softening radius for potential
        self.opts['n'] = npart        # Amount of particles
        self.opts['G'] = G            # Gravitational constant
        self.opts['dt'] = dt          # Time step of our simulation
        
        # Masses are hard-coded to be the same for all particles
        if len(m)==0: self.m = np.full(self.opts['n'] , 1.0)
        else:         self.m = m
        
        
        # By default:
        #   Dimension ranges assume that both bounding values are obtainable
        #   Initial particle positions are uniform random.
        #   Initial speeds are set to zero.
        if len(x)==0:  self.x = np.random.uniform(low=xrange[0],high=xrange[1],size=self.opts['n'])
        else:          self.x = x.copy()
        if len(y)==0:  self.y = np.random.uniform(low=yrange[0],high=yrange[1],size=self.opts['n'])
        else:          self.y = y.copy()
        if len(z)==0:  self.z = np.random.uniform(low=zrange[0],high=zrange[1],size=self.opts['n'])
        else:          self.z = z.copy()
        if len(vx)==0: self.vx = np.zeros(self.opts['n'])
        else:          self.vx = vx.copy()
        if len(vy)==0: self.vy = np.zeros(self.opts['n'])
        else:          self.vy = vy.copy()
        if len(vz)==0: self.vz = np.zeros(self.opts['n'])
        else:          self.vz = vz.copy()
        
        # Initialize forces arrays for each dimension
        self.fx = np.zeros(self.opts['n'])
        self.fy = self.fx.copy()
        self.fz = self.fx.copy()
        
        #Initialize energy for the system
        self.potE = 0
        self.kinE = 0
        self.totalE = 0
        
        self.xrange = xrange;  self.yrange = yrange;  self.zrange = zrange
        
        # Grid values for x,y,z of the density grid
        # Grid size for each dimension needs to be even (I think)
        self.grid_size = grid_size
        self.grid_x = np.linspace( xrange[0] , xrange[1] , grid_size[0] )
        self.grid_rez_x = self.grid_x[1] - self.grid_x[0]
        self.grid_y = np.linspace( yrange[0] , yrange[1] , grid_size[1] )
        self.grid_rez_y = self.grid_y[1] - self.grid_y[0]
        self.grid_z = np.linspace( zrange[0] , zrange[1] , grid_size[2] )
        self.grid_rez_z = self.grid_z[1] - self.grid_z[0]
        
        # If the softening was left unspecified, set it using the grid resolution
        if soft==None: self.opts['soft'] = 2*np.max([ self.grid_rez_x , self.grid_rez_y , self.grid_rez_z ])
        
        # Initializes the (softened) Green's Function convolution kernel centered at (x,y,z)=(0,0,0)
        # If bound conditions are periodic, kernel is same shape as the density
        # Else kernel is double the size of the density in each dimension (to have zero-padding at boundries)
        x_interval = xrange[1]-xrange[0];   y_interval = yrange[1]-yrange[0];   z_interval = zrange[1] - zrange[0]
        if bound=='periodic':
            grid_xx , grid_yy, grid_zz = np.meshgrid(  np.linspace( -x_interval/2 , x_interval/2 , len(self.grid_x) )
                                                     , np.linspace( -y_interval/2 , y_interval/2 , len(self.grid_y) )
                                                     , np.linspace( -z_interval/2 , z_interval/2 , len(self.grid_z) )  )
        elif bound=='non-periodic':
            grid_zz , grid_yy, grid_xx = np.meshgrid(  np.linspace( -x_interval , x_interval , int( len(self.grid_x)*2 + grid_size[0]%2) )
                                                     , np.linspace( -y_interval , y_interval , int( len(self.grid_y)*2 + grid_size[1]%2) )
                                                     , np.linspace( -z_interval , z_interval , int( len(self.grid_z)*2 + grid_size[2]%2) )  )  
        else:
            raise Exception('Illegal boundary condition was requested. Only periodic or non-periodic are supported')
        
        # Creating the Green's function of the Laplacian which solves for the potential using the density grid
        # Using the Green's function for 3D distances ( -1/(4pi*dist) ) taken from
        # https://ocw.mit.edu/courses/mathematics/18-303-linear-partial-differential-equations-fall-2006/lecture-notes/greensfn.pdf
        dist = np.sqrt( grid_xx**2 + grid_yy**2 + grid_zz**2 )
        self.pot_conv_kern = -1/(4*np.pi*dist)
        self.pot_conv_kern = np.where( dist <= self.opts['soft'] ,  -1/(4*np.pi*self.opts['soft']) , self.pot_conv_kern )
        grid_xx=None; grid_yy=None; grid_zz=None; dist=None
    
    
    
    
    
    def get_forces_mui(self , return_pot=False , doPlot=False):
        
        # TODO
        # 1) create density grid based on particle positions
        #    and the grid resolution                                (DONE)
        # 2) convolve density with potential convolution kernel     (DONE)
        # 3) use gradient of potential to get force map in (x,y)  (DONE)

        density_grid_3d = np.zeros((self.grid_size))
        
        # Keep track of the indices the particles fall onto. Useful when assigning forces to particles
        indices = []
        for i in range(self.opts['n']):
            # Assign grid positions based on the Nearest-Grid-Point method
            xind = (np.abs(self.x[i] - self.grid_x)).argmin()
            yind = (np.abs(self.y[i] - self.grid_y)).argmin()
            zind = (np.abs(self.z[i] - self.grid_z)).argmin()
            density_grid_3d[xind,yind,zind] += self.m[i]
            indices.append([xind,yind,zind])
        # Multiply the density grid by the area contained in a cell to get actual density and not just mass
        density_grid_3d /= self.grid_rez_x*self.grid_rez_y*self.grid_rez_z
        
        
        # These doPlot booleans decide if density, kenel, potential and/or force plots
        # are generated to check if the simulation behaves as expected. In 2D, need to transpose
        # the arrays otherwise plt.imshow() will align x values with the y-axis and vice-versa.
        if doPlot:
            """
            #if self.opts['bound']=='periodic':
            #    x_interval = self.xrange[1]-self.xrange[0];  y_interval = self.yrange[1]-self.yrange[0];  z_interval = self.zrange[1]-self.zrange[0]
            #elif self.opts['bound']=='non-periodic':
            #    x_interval = (self.xrange[1]-self.xrange[0])*2;  y_interval = (self.yrange[1]-self.yrange[0])*2;  z_interval = (self.zrange[1]-self.zrange[0])*2
            #extent_kern = [ [-z_interval/2,z_interval/2] , [-y_interval/2,y_interval/2] , [-x_interval/2,x_interval/2] ]
            """
            extent_field = [ [self.xrange[0],self.xrange[1]] , [self.yrange[0],self.yrange[1]] , [self.zrange[0],self.zrange[1]] ]
            grid_xx , grid_yy, grid_zz = np.meshgrid(self.grid_x,self.grid_y,self.grid_z)
            
            scatter3d(grid_xx.ravel(),grid_yy.ravel(),grid_zz.ravel() , density_grid_3d.ravel() , colorsMap='inferno'
                      , title='Density Map' , axlims=extent_field , alpha=0.6 , axlabs=['X-axis','Y-axis','Z-axis'])
            
            if self.opts['bound']=='non-periodic':
                grid_xxp , grid_yyp, grid_zzp = np.meshgrid( np.arange(int(len(self.grid_x)*2))
                                                           , np.arange(int(len(self.grid_y)*2))
                                                           , np.arange(int(len(self.grid_z)*2)) )
                scatter3d(grid_xxp.ravel(),grid_yyp.ravel(),grid_zzp.ravel() , self.pot_conv_kern.ravel() , colorsMap='inferno'
                          , title='1/r kernel' , alpha=0.6 , axlabs=['X-axis','Y-axis','Z-axis'])
            elif self.opts['bound']=='periodic':
                scatter3d(grid_xx.ravel(),grid_yy.ravel(),grid_zz.ravel() , self.pot_conv_kern.ravel() , colorsMap='inferno'
                          , title='1/r kernel' , alpha=0.6 , axlabs=['X-axis','Y-axis','Z-axis'])
        
            
        # Now we convolve the density with the Green's Function
        # Need to fft_shift the result so that the potential is aligned according to our particle positions
        if self.opts['bound']=='periodic':
            # If the bounds are periodic, no special treatment is needed
            potential = 4*np.pi*self.opts['G'] * np.fft.irfftn( np.fft.rfftn(density_grid_3d) * np.fft.rfftn(self.pot_conv_kern) )
            for i in range(len(self.grid_size)):
                potential = np.fft.fftshift(potential,axes=i)
            
        elif self.opts['bound']=='non-periodic':
            # If bounds are non-periodic, add a bunch of zeropadding around the original density grid
            # The Green's function kernel is adjusted according to this new size in the init phase
            # Once the convolution is performed, simply retrieve the section containing the original grid
            
            dims = density_grid_3d.shape
            potential = np.zeros(dims)
            zeropad_density = np.zeros(np.array(dims)*2)
            
            zeropad_density[ int(dims[0]/2):-int(dims[0]/2)-(dims[0]%2)
                           , int(dims[1]/2):-int(dims[1]/2)-(dims[1]%2)
                           , int(dims[2]/2):-int(dims[2]/2)-(dims[2]%2) ] = density_grid_3d
            potential_padded = np.fft.irfftn( np.fft.rfftn(zeropad_density) * np.fft.rfftn(self.pot_conv_kern) )
            for i in range(len(self.grid_size)):
                potential_padded = np.fft.fftshift(potential_padded,axes=i)
            
            if doPlot:
                scatter3d( grid_xxp.ravel(),grid_yyp.ravel(),grid_zzp.ravel() , potential_padded.ravel() , colorsMap='inferno'
                         , title='Padded 1/r kernel' , axlims=[] , alpha=0.6 , axlabs=['X-axis','Y-axis','Z-axis'] )
            
            
            potential = potential_padded[ int(dims[0]/2):-int(dims[0]/2)-(dims[0]%2)
                                        , int(dims[1]/2):-int(dims[1]/2)-(dims[1]%2)
                                        , int(dims[2]/2):-int(dims[2]/2)-(dims[2]%2) ]
            potential_padded = None
            zeropad_density = None
            
            
        # Assign forces by taking the gradient of the potential field
        # Take negative since Force = -gradient(Potential)
        x_force_grid, y_force_grid, z_force_grid = np.gradient( potential , self.grid_x, self.grid_y, self.grid_z )
        x_force_grid *= -1;  y_force_grid *= -1; z_force_grid *= -1
        
        """
        if doPlot:
            plt.figure()
            plt.subplot(2,2,1); plt.title('potenial field')
            plt.imshow(potential , origin='lower' , extent=extent_field )
            #plt.subplot(2,2,2); plt.title('force lines')
            plt.subplot(2,2,2); plt.title('force field')
            plt.imshow(np.sqrt(x_force_grid**2+y_force_grid**2) , origin='lower' , extent=extent_field)
            for i in range(self.opts['n']):
                plt.arrow(self.x[i] , self.y[i] , y_force_grid[tuple(indices[i])] , x_force_grid[tuple(indices[i])] )
            plt.quiver(self.grid_y, self.grid_x, y_force_grid, x_force_grid)
            plt.subplot(2,2,3); plt.title('gx')
            plt.imshow(x_force_grid , origin='lower' , extent=extent_field)
            plt.subplot(2,2,4); plt.title('gy')
            plt.imshow(y_force_grid , origin='lower' , extent=extent_field)
        """
        
        x_force = np.zeros(self.opts['n'])
        y_force = np.zeros(self.opts['n'])
        z_force = np.zeros(self.opts['n'])
        #if return_pot: pot=0
        for i in range(self.opts['n']):
            # The switch seems necessary to make things work. Has to to with how I indexed my fields
            x_force[i] = x_force_grid[tuple(indices[i])]
            y_force[i] = y_force_grid[tuple(indices[i])]
            z_force[i] = z_force_grid[tuple(indices[i])]
        
        if return_pot:
            
            return x_force , y_force , z_force \
                 , 0.5*np.sum(potential)*np.sum(self.m) * self.grid_rez_x*self.grid_rez_y*self.grid_rez_z
        else:
            return x_force , y_force , z_force
        
    
    
    # Implements leapfrog technique for updating positions and velocities
    # Inspired from the wikipedia page on leapfrogging
    def evolve(self, start=False , doPlot=False):
        
        # If start=True, calculate forces, energy at initial positions
        if start:
            self.fx , self.fy , self.fz , self.potE = self.get_forces_mui( return_pot=True , doPlot=doPlot )
            self.kinE = 0.5*np.sum(self.m*(self.vx**2+self.vy**2+self.vz**2))
            self.totalE = self.kinE+self.potE
            print('Initial total energy is,',self.totalE,'\n')
        
        # Calculate next time-step position
        # If particle falls out of grid dimensions, make it enter at the opposite end
        # Regardless of boundry conditions, this is how out-of-bound particles are treated
        self.x = self.x + self.vx*self.opts['dt']  +  0.5*self.fx*self.opts['dt']**2 / self.m
        self.y = self.y + self.vy*self.opts['dt']  +  0.5*self.fy*self.opts['dt']**2 / self.m
        self.z = self.z + self.vz*self.opts['dt']  +  0.5*self.fz*self.opts['dt']**2 / self.m
        for i in range(self.opts['n']):
            if self.x[i] < self.xrange[0]:
                self.x[i] = self.xrange[1] - ( (self.xrange[0]-self.x[i]) % (self.xrange[1]-self.xrange[0]) )
            elif self.x[i] >= self.xrange[1]:
                self.x[i] = self.xrange[0] + ( (self.x[i]-self.xrange[1]) % (self.xrange[1]-self.xrange[0]) )
            if self.y[i] < self.yrange[0]:
                self.y[i] = self.yrange[1] - ( (self.yrange[0]-self.y[i]) % (self.yrange[1]-self.yrange[0]) )
            elif self.y[i] >= self.yrange[1]:
                self.y[i] = self.yrange[0] + ( (self.y[i]-self.yrange[1]) % (self.yrange[1]-self.yrange[0]) )
            if self.z[i] < self.zrange[0]:
                self.z[i] = self.zrange[1] - ( (self.zrange[0]-self.z[i]) % (self.zrange[1]-self.zrange[0]) )
            elif self.z[i] >= self.zrange[1]:
                self.z[i] = self.zrange[0] + ( (self.z[i]-self.zrange[1]) % (self.zrange[1]-self.zrange[0]) )
            
        
        # Calculate forces and the potential from updated positions
        fx_next , fy_next , fz_next , self.potE = self.get_forces_mui(return_pot=True , doPlot=doPlot)
        
        # Calculate next time-step velocities
        self.vx = self.vx  +  0.5*(self.fx+fx_next)*self.opts['dt'] / self.m
        self.vy = self.vy  +  0.5*(self.fy+fy_next)*self.opts['dt'] / self.m
        self.vz = self.vz  +  0.5*(self.fz+fz_next)*self.opts['dt'] / self.m
        
        # Officially update forces
        self.fx = fx_next;  self.fy = fy_next;  self.fz = fz_next
        
        # Update kinetic energy as well
        self.kinE = 0.5*np.sum(self.m*(self.vx**2+self.vy**2+self.vz**2))
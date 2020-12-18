Code functionality:

  - 'particles.py' contains the classes for 2D and 3D particles.
     It contains all the necessary functions to run the simulation.
     
  - 'run_nobody2D.py' uses the 'particles2d' class from 'particles.py
    to output plots of a 2D n-body simulation.
  
  - 'run_nobody3D.py' uses the 'particles3d' class from 'particles.py
    to output plots of a 3D n-body simulation.


Additional comments:

  - Due to the enormous runtime I encountered when simulating in 3D, coupled 
    with unsatisfactory plotting of the results,  most of my videos produced are in 2D.
    
  - Regardless of boundary conditions, particles that exit the grid are always put back in
    at the opposite end of the respective dimension it went out from. This can lead to
    some weird behaviour for the case with non-periodic boundaries, but I desired to see what would happen
  
  - Evident from the energy plots for the simulations with many particles, it revealed harder than expected
    to conserve the energy in my system when multiple particles were present.
    Either it wasn't being conserved or the way I was calculating potential energy was worng.

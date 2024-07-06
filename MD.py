# ==================================================================================================================== #
#
# Welcome to the major MD assignment.
# This skeleton is provided as a guideline.
# You can use other software (MATLAB, ...)
# and also deviate from the functions provided
# This file is given as a skeleton and guideline.
#
#
# ==================================================================================================================== #

# # # # # # # # # # # # #
# Load necessary packages
# # # # # # # # # # # # #

import configparser
import numbers
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as spc
import math
import sympy


# # # # # # # # # # # # #
# Functions you can use to help you
# # # # # # # # # # # # #

# # # # # # # # # # # # #
# Function Templates you can use to write your code
# You do not have to use these,
# but still read their description for extra information
# and hints
# # # # # # # # # # # # #

def read_xyz(input_conf):
    '''
        Function to read coordinates from xyz format.
    '''
    xyz_file = open(input_conf, 'r')

    nparts = int(xyz_file.readline())   #read the first line of the coordinates file, which tells you the amount of particles in the box
    xyz_file.readline() #Skip 2nd line of coordinates file (doesnt include valueable information)
    coord = np.zeros([nparts,3])    #Create a nParts x 3 array which will then be used to store the coordinates in x, y, z columns
    for k in range(0,nparts):
        line = xyz_file.readline()  #Read a new line (row) of the configuration for every iteration, starting from line 3 (coordinates of 1st particle)
        line = line.split() #Split line into different parts to then store each separately
        coord[k,0] = line[1]    #Store coordinate x
        coord[k,1] = line[2]    #Store coordinate y
        coord[k,2] = line[3]    #Store coordinate z

    return coord, nparts    #Return the coordinates array and the number of particles


def read_xyz_trj(file_name):    #Read a file containing the trajectory of a simulation

    xyz_file = open(file_name, 'r')

    frame = 0
    xyz = {}
    READING=True
    while READING:
        try:
            nparts = int(xyz_file.readline())
            xyz_file.readline()
            xyz[frame] = np.zeros([nparts, 3])
            for k in range(0, nparts):
                line = xyz_file.readline()
                line = line.split()
                xyz[frame][k, 0] = line[1]
                xyz[frame][k, 1] = line[2]
                xyz[frame][k, 2] = line[3]
            frame += 1
        except:
            print("Reach end of '" + file_name + "'")
            READING=False

    return xyz


def read_lammps_data(data_file, verbose=False): #Read a LAMMPS data file: file that contains structural and molecular information of a system to be simulated
    """Reads a LAMMPS data file
        Atoms
        Velocities
    Returns:
        lmp_data (dict):
            'xyz': xyz (numpy.ndarray)
            'vel': vxyz (numpy.ndarray)
        box (numpy.ndarray): box dimensions
    """
    print("Reading '" + data_file + "'")
    with open(data_file, 'r') as f:
        data_lines = f.readlines()

    # TODO: improve robustness of xlo regex
    directives = re.compile(r"""
        ((?P<n_atoms>\s*\d+\s+atoms)
        |
        (?P<box>.+xlo)
        |
        (?P<Atoms>\s*Atoms)
        |
        (?P<Velocities>\s*Velocities))
        """, re.VERBOSE)

    i = 0
    while i < len(data_lines):
        match = directives.match(data_lines[i])
        if match:
            if verbose:
                print(match.groups())

            elif match.group('n_atoms'):
                fields = data_lines.pop(i).split()
                n_atoms = int(fields[0])
                xyz = np.empty(shape=(n_atoms, 3))
                vxyz = np.empty(shape=(n_atoms, 3))

            elif match.group('box'):
                dims = np.zeros(shape=(3, 2))
                for j in range(3):
                    fields = [float(x) for x in data_lines.pop(i).split()[:2]]
                    dims[j, 0] = fields[0]
                    dims[j, 1] = fields[1]
                L = dims[:, 1] - dims[:, 0]

            elif match.group('Atoms'):
                if verbose:
                    print('Parsing Atoms...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    a_id = int(fields[0])
                    xyz[a_id - 1] = np.array([float(fields[2]),
                                         float(fields[3]),
                                         float(fields[4])])

            elif match.group('Velocities'):
                if verbose:
                    print('Parsing Velocities...')
                data_lines.pop(i)
                data_lines.pop(i)

                while i < len(data_lines) and data_lines[i].strip():
                    fields = data_lines.pop(i).split()
                    va_id = int(fields[0])
                    vxyz[va_id - 1] = np.array([float(fields[1]),
                                         float(fields[2]),
                                         float(fields[3])])

            else:
                i += 1
        else:
            i += 1

    return xyz, vxyz, L


def write_frame(coords, L, vels, forces, trajectory_name, step):
    '''
    function to write trajectory file in LAMMPS format

    In VMD you can visualize the motion of particles using this trajectory file.

    :param coords: coordinates
    :param vels: velocities
    :param forces: forces
    :param trajectory_name: trajectory filename

    :return:
    '''

    nPart = len(coords[:, 0])
    nDim = len(coords[0, :])
    with open(trajectory_name, 'a') as file:
        file.write('ITEM: TIMESTEP\n')
        file.write('%i\n' % step)
        file.write('ITEM: NUMBER OF ATOMS\n')
        file.write('%i\n' % nPart)
        file.write('ITEM: BOX BOUNDS pp pp pp\n')
        for dim in range(nDim):
            file.write('%.6f %.6f\n' % (-0.5 * L, 0.5 * L))   #For control volumes with Lx = Ly = Lz
            #file.write('%.6f %.6f\n' % (-0.5 * L[dim], 0.5 * L[dim]))  #For control volumes with Lx != Ly != Lz
        for dim in range(3 - nDim):
            file.write('%.6f %.6f\n' % (0, 0))
        file.write('ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz\n')

        temp = np.zeros((nPart, 9))
        for dim in range(nDim):
            temp[:, dim] = coords[:, dim]
            temp[:, dim + 3] = vels[:, dim]
            temp[:, dim + 6] = forces[:, dim]

        for part in range(nPart):
            file.write('%i %i %.4f %.4f %.4f %.6f %.6f %.6f %.4f %.4f %.4f\n' % (part + 1, 1, *temp[part, :]))


def read_lammps_trj(lammps_trj_file):

    def read_lammps_frame(trj):
        """Load a frame from a LAMMPS dump file.

        Args:
            trj (file): LAMMPS dump file of format 'ID type x y z' or
                                                   'ID type x y z vx vy vz' or
                                                   'ID type x y z fz'
            read_velocities (bool): if True, reads velocity data from file
            read_zforces (bool): if True, reads zforces data from file

        Returns:
            xyz (numpy.ndarray):
            types (numpy.ndarray):
            step (int):
            box (groupy Box object):
            vxyz (numpy.ndarray):
            fz (numpy.ndarray):
        """
        # --- begin header ---
        trj.readline()  # text "ITEM: TIMESTEP"
        step = int(trj.readline())  # timestep
        trj.readline()  # text "ITEM: NUMBER OF ATOMS"
        n_atoms = int(trj.readline())  # num atoms
        trj.readline()  # text "ITEM: BOX BOUNDS pp pp pp"
        Lx = trj.readline().split()  # x-dim of box
        Ly = trj.readline().split()  # y-dim of box
        Lz = trj.readline().split()  # z-dim of box
        L = np.array([float(Lx[1]) - float(Lx[0]),
                      float(Ly[1]) - float(Ly[0]),
                      float(Lz[1]) - float(Lz[0])])
        trj.readline()  # text
        # --- end header ---

        xyz = np.empty(shape=(n_atoms, 3))
        xyz[:] = np.NAN
        types = np.empty(shape=(n_atoms), dtype='int')
        vxyz = np.empty(shape=(n_atoms, 3))
        vxyz[:] = np.NAN
        fxyz = np.empty(shape=(n_atoms, 3))
        fxyz[:] = np.NAN

        # --- begin body ---

        IDs = []
        for i in range(n_atoms):
            temp = trj.readline().split()
            a_ID = int(temp[0]) - 0  # atom ID
            xyz[a_ID - 1] = [float(x) for x in temp[2:5]]  # coordinates
            types[a_ID - 1] = int(temp[1])  # atom type
            vxyz[a_ID - 1] = [float(x) for x in temp[5:8]]  # velocities
            fxyz[a_ID - 1] = [float(x) for x in temp[8:11]]  # map(float, temp[5]) # z-forces

        # --- end body ---
        return xyz, types, step, L, vxyz, fxyz


    xyz = {}
    vel = {}
    forces = {}
    with open(lammps_trj_file, 'r') as f:
        READING = True
        c = 0
        while READING:
            try:
                xyz[c], _, _, _, vel[c], forces[c] = read_lammps_frame(f)
                c += 1
            except:
                READING=False

    return xyz, vel, forces


def rdf(xyz, LxLyLz, n_bins=100, r_range=(0.01, 10.0)):
    '''
    rarial pair distribution function

    :param xyz: coordinates in xyz format per frame
    :param LxLyLz: box length in vector format
    :param n_bins: number of bins
    :param r_range: range on which to compute rdf
    :return:
    '''

    g_r, edges = np.histogram([0], bins=n_bins, range=r_range)
    g_r[0] = 0
    g_r = g_r.astype(np.float64)
    rho = 0

    for i, xyz_i in enumerate(xyz):
        xyz_j = np.vstack([xyz[:i], xyz[i + 1:]])
        d = np.abs(xyz_i - xyz_j)
        d = np.where(d > 0.5 * LxLyLz, LxLyLz - d, d)
        d = np.sqrt(np.sum(d ** 2, axis=-1))
        temp_g_r, _ = np.histogram(d, bins=n_bins, range=r_range)
        g_r += temp_g_r

    rho += (i + 1) / np.prod(LxLyLz)
    r = 0.5 * (edges[1:] + edges[:-1])
    V = 4./3. * np.pi * (np.power(edges[1:], 3) - np.power(edges[:-1], 3))
    norm = rho * i
    g_r /= norm * V

    return r, g_r


# # # # # #
# Functions
# # # # # #

def initGrid(L, nPart, nDim):
    '''
    function to determine particle positions and box size for a specified number of particles and density

    :param L: box dimension
    :return: coordinates
    '''
    coords = np.zeros((nPart, nDim))  # load empty array for coordinates
    n = np.ceil(nPart**(1/nDim))  # Find number of particles in each lattice line
    index = np.zeros(nDim)  # initiate lattice indexing
    spacing = L / n  # define lattice spacing
    if nDim == 1:
        for particle in range(nPart):  # assign particle positions
            coords[particle] = index * spacing   #Compute coordinates of particle
            index[0] += 1  # advance particle position
        plt.scatter(coords, np.zeros_like(coords))  #Plot for 1D
        plt.title("Initial configuration for %iD" % nDim)
        plt.xlabel('X [A]')
        plt.show()
    if nDim == 2:
        for particle in range(nPart):  # assign particle positions
            coords[particle, :] = index * spacing   #Compute coordinates of particle
            index[0] += 1  # advance particle position
            if index[0] == n:  # if last lattice point is reached jump to next line
                index[0] = 0
                index[1] += 1
        plt.scatter(coords[:, 0], coords[:, 1]) #Plot for 2D
        plt.title("Initial configuration for %iD"%nDim)
        plt.xlabel('X [A]')
        plt.ylabel('y [A]')
        plt.show()
    if nDim == 3:
        for particle in range(nPart):  # assign particle positions
            coords[particle, :] = index * spacing   #Compute coordinates of particle
            index[0] += 1  # advance particle position
            if index[0] == n:  # if last lattice point is reached jump to next line
                index[0] = 0
                index[1] += 1
            if index[1] == n:   #If last line is reached, jump to next level
                index[1] = 0
                index[2] += 1
        fig = plt.figure()  #Plot for 3D
        ax = plt.axes(projection="3d")
        ax.scatter3D(coords[:,0], coords[:,1], coords[:,2])
        ax.set(xlabel='X [A]', ylabel='Y [A]', zlabel='Z [A]')
        plt.title("Initial configuration for %iD"%nDim)
        plt.show()
    return coords

def initVel(coords, Temp, mass, R, nDim):
    '''
    Initiate velocities by drawing random numbers from numpy random
    :return: vels : velociies of the particles
    '''
    vel = np.random.randn(coords.shape[0],nDim)  #Initialise velocity vector array [nparticles x dimensions]
    Kin_energy_avg = (0.5*np.sum(np.dot(mass, vel**2)))*(10**4)/coords.shape[0] #Compute the average kinetic energy of the system given by the velocities
    Temp_now = 2*Kin_energy_avg/(nDim*R)    #Calculate the temperature at that moment given by the kinetic energy that was just calculated
    lam = np.sqrt(Temp/Temp_now)    #Calculate the ratio between the wanted temperature and the temperature at that moment
    vels = lam*vel  #Scale the velocity by the ratio between the instantaneous temperature and the wanted temperature
    return vels

def LJ_forces(coords, LxLyLz,Rcut):
    '''
    function to compute LJ forces between particles

    :param coords: position of partilces
    :param L: box size

    :return: forces: LJ forces
    '''
    #print(coords)
    forces = np.zeros(coords.shape) ## initialize empty array for forces
    nPart = coords.shape[0] # obtain number of particles
    #sr814 = np.zeros(65341) #Create an array of the size of all possible combinations
    #iteration = 0   #Initialize number of iterations
    for i in range(0, coords.shape[0], 1):
        for j in range(0, coords.shape[0],1):
            if i != j:
                v_ij = coords[j, :] - coords[i, :]  # Compute the distance vector between two particles. (1x3 array)
                #print(v_ij)
                # Correct distance for PBC/MIC
                v_ij -= LxLyLz * np.round(v_ij / LxLyLz)  # Apply BCs per dimension of the distance vector
                d2 = np.sum(v_ij ** 2)  # Compute the square of the magnitude of the distance


                # if Rcut != 0: #Cutoff != 0 then the cutoff distance will be considered. Cutoff = the cutoff distance
                #     sr2 = sigma ** 2 / d2[d2 < Rcut ** 2]  # Compute (sigma/r)^2 whilst applying a mask (filtering condition for only distance less than the Rcut value)
                #     sr8 = sr2 ** 3 / d2
                #     sr14 = 2* (sr2 ** 6 / d2)
                # if Rcut == 0: #Cutoff == 0 if the option to have a cutoff is not selected
                #     sr2 = sigma ** 2 / d2  # Compute (sigma/r)^2 without applying a mask
                #     sr8 = sr2 ** 3 / d2
                #     sr14 = 2*(sr2 ** 6 / d2)
                sr2 = sigma ** 2 / d2  # Compute (sigma/r)^2 without applying a mask
                sr8 = sr2 ** 3 / d2
                sr14 = 2 * (sr2 ** 6 / d2)

                sr814 = sr14 - sr8  #Scalar (1x1 array)
                #print(sr814)
                sr814_vector = v_ij*sr814   #Give a force vector
                #sr814_total = np.sum(sr814_vector, axis=0)
                #forces = 24 * epsilon * sr814_total
                force = 24 * epsilon * sr814_vector #Compute the force
                forces[i] += force  #Update force acting on particle i with the force due to particle j
    #print("forces: ",forces)   #Just to check
    return forces

def velocityVerlet(coords, L, vels, forces, mass, dt, dt2):
    """
    Implement integration by the Velocity Verlet algorithm.
    Parameters:
    - coords: positions of the particles
    - L: size of the cubic simulation box
    - vels:  velocities of particles calculates using initVel()
    - forces: forces on each particle calculated using LJ_forces()
    - mass: mass of the particles (assuming all particles have the same mass)
    - dt: time step for integration
    - dt2: dt**2
    -sigma: LJ parameter
    -epsilon: LJ parameter
    -cutoff: cutoff distance for LJ interactions
    - steps: number of simulation steps
    Returns:
    - coords
    -vels
    -forces
    """
    coords += vels * dt + (0.5 * forces * dt2 / mass)*(1/(10**4)) # Update positions. Divide by 10**4 for units to match.
    coords = coords % L # Apply periodic boundary conditions and update coordinates
    #print(coords)
    vels += (0.5 * forces * dt / mass) * (1/(10**4)) #Angstrom/fs   #Compute vels(t+dt/2) (computationally less demanding)
    forces = LJ_forces(coords,L,cutoff) # Compute forces for updated configuration
    vels += (0.5 * forces * dt / mass)* (1/(10**4)) # Update velocities

    return  coords,vels,forces

def kineticEnergy(vels,mass):   #Compute the kinetic energy of the system
    Kinetic = 0 #Initialize the kinetic energy of the system
    Kinetic += (0.5*np.sum(np.dot(mass, vels**2)))*(10**4)   #Multiply by 10**4 to get kinetic energy in kJ/mol
    return Kinetic

def potentialEnergy(r,Lbox,cutoff,sigma,epsilon,number_density,temperature):  #Compute the potential energy of the system
    Kb = spc.k  #Boltzmann constant
    Etot = 0  # Initialize the energy in order for it to be updated at every iteration
    for i in range(0, r.shape[0] - 1, 1):
        difference = r[i+1,:] - r[i,:]  #Compute the distance vector between two particles
        #print(difference)
        # Correct distance for PBC/MIC
        difference -= Lbox*np.round(difference/Lbox) #Apply BCs per dimension of the distance vector
        d2 = np.sum(difference**2)  #Compute the square of the magnitude of the distance

        if cutoff != 0:  # Cutoff != 0 then the cutoff distance will be considered. Cutoff = the cutoff distance
            sr2_pot = sigma ** 2 / d2[d2<cutoff**2]  # Compute (sigma/r)^2 whilst applying a mask (filtering condition for only distance less than the Rcut value)
            Rcut = cutoff   #If cutoff has been stated, use Rcut as the stated cutoff distance
        if cutoff == 0:  # Cutoff == 0 if the option to have a cutoff is not selected
            sr2_pot = sigma ** 2 / d2  # Compute (sigma/r)^2 without applying a mask
            Rcut = 1E30 #Set a really high cutoff distance to compute the tail correction (giving it a very high value will make the tail correction approximate 0)

        sr6 = sr2_pot**3
        sr12 = sr6**2
        Etot += np.sum(sr12 - sr6)  #Update the total energy (without having to do it in a for loop)
    tail = (8 / 3) * math.pi * number_density * epsilon * (sigma ** 3) * ((1 / 3) * ((sigma / Rcut) ** 9) - ((sigma / Rcut) ** 3)) #Compute the tail part of the L-J potential energy
    Etot = Etot*4*epsilon - tail*r.shape[0] #Correct the total energy considering the tail part
    return Etot

def temperature(Kinetic_Energy,coords,nDim):  #Compute the instantaneous temperature of the system
    NA = spc.N_A  # 6.02214076e23 [1/mol] # Avogadros number
    kB = spc.k * (1 / (10 ** 3))  # 1.380649e-23 [KJ/K]  #Boltazmann constant
    R = kB * NA  # 0.008314462618153242 # Boltzmann constant / NA [KJ/molK]  #Gas constant
    T = 0   #Initialize temperature
    T +=  (2.0 / nDim)*(Kinetic_Energy/(coords.shape[0] * R))    #Compute the temperature given the kinetic energy of the system
    return T

def pressure(r,Lbox,cutoff,sigma,epsilon,number_density,temperature):   #Compute the pressure of the system
    Kb = spc.k  #Boltzmann constant
    P_virial = 0  # Initialize the virial pressure
    for i in range(0, r.shape[0] - 1, 1):
        difference = r[i + 1, :] - r[i, :]  # Compute the distance vector between two particles
        # print(difference)
        # Correct distance for PBC/MIC
        difference -= Lbox * np.round(difference / Lbox)  # Apply BCs per dimension of the distance vector
        d2 = np.sum(difference ** 2)  # Compute the square of the magnitude of the distance

        if cutoff != 0:  # Cutoff != 0 then the cutoff distance will be considered. Cutoff = the cutoff distance
            sr2_pres = sigma ** 2 / d2[d2<cutoff**2]  # Compute (sigma/r)^2 whilst applying a mask (filtering condition for only distance less than the Rcut value)
            Rcut = cutoff   #If cutoff has been stated, use Rcut as the stated cutoff distance
        if cutoff == 0:  # Cutoff == 0 if the option to have a cutoff is not selected
            sr2_pres = sigma ** 2 / d2  # Compute (sigma/r)^2 without applying a mask
            Rcut = 1E30 #Set a really high cutoff distance to compute the tail correction (giving it a very high value will make the tail correction approximate 0)

        sr6 = sr2_pres**3
        sr12 = sr6 ** 2
        P_virial -= np.sum(epsilon * ((24 * sr6) - (48 * sr12)))  # Update the virial pressure (without having to do it in a for loop)
    P_virial = ((number_density) * R * temperature - (1 / (3 * (Lbox ** 3))) * P_virial)*(1/NA)*(10**28)  # [bar] Compute the total virial_pressure
    return P_virial

# # # # # # # # # # # #
# Set initial parameters
# # # # # # # # # # # #

# simulation parameters
nDim = 3
Nfreq = 100   # sample frequency
NA = spc.N_A #6.02214076e23 [1/mol] # Avogadros number
kB = spc.k*(1/(10**3)) #1.380649e-23 [KJ/K]  #Boltazmann constant
R = kB*NA   #0.008314462618153242 # Boltzmann constant / NA [KJ/molK]  #Gas constant
# 1 cal = 4.184 J

# CH4 methane
sigma = 3.73 # [angstrom]
epsilon = 148*R # [KJ/mol]
mass = 16.04 # [g/mol]

# density
rho = 358.4 # kg/m3
L = 30 # Angstrom
nPart = int(rho*(1/(10**30))*(L**3)*1000*(1/mass)*NA) #particles
number_density= nPart/(L**(nDim))

cutoff = 10

# Simulation parameters
steps = 3000 # length of simulation run [fs]
dt = 1     # timestep [fs]

# temperature
Tstart = 150   # [K]
Tend = 150   # [K]
Tdamp = 150    # temperature damping factor

log_filename = 'log'
trajectory_filename = 'trajectory'

# some unit conversions
# 1 cal = 4.184 J
# 1 J = 1 Kg m^2 / s^2
# 1 Kg = 1e3 g
# 1m = 1e10 Angstrom
# 1s = 1e15 fs
# 1 Kg m^2 / s^2 = 1e3*(1e10/1e15)^2 = 1e-7  g Angstrom^2 / fs^2
# Kcal/mol = 4.184 * 1e-4 g/mol*(Angstrom/fs)^2
# g/mol*(Angstrom/fs)^2 = 1/(4.184 * 1e-4) Kcal/mol

# # # # # # # # # # # #
# Initialize
# # # # # # # # # # # #

coords = initGrid(L, nPart, nDim)
vels = initVel(coords,Tstart, mass, R, nDim)
# coords, vels, L = read_lammps_data('VERIFY1/lammps.data') #To use if system is wrong
forces = LJ_forces(coords, L,cutoff)

# # # # # # # # # # # #
# MD
# # # # # # # # # # # #

dT = (Tend - Tstart)/steps
Temp = Tstart
time = 0
dt2 = dt * dt

with open('log', 'a') as file:
    file.write('Step KE Upot totE\n')
print('Step T KE Upot totE')

KE = np.zeros(int(steps/Nfreq))
Tempi = np.zeros(int(steps/Nfreq))
Upot = np.zeros(int(steps/Nfreq))
for step in range(steps+1):

    # integrate
    # cords, vels, forces, xi = velocityVerletThermostat(coords, L, vels, forces, xi, mass, dt, dt2, kB * Temp, sigma, epsilon, cutoff, Tdamp)
    coords, vels, forces = velocityVerlet(coords, L, vels, forces, mass, dt, dt2)

    # wrap coordinates
    #coords = wrap(coords, L)

    time += dt
    Temp += dT

    # write outputs
    # if step%Nfreq == 0:
    #     KE[int(step/Nfreq)] = kineticEnergy(vels,mass)   # kinetic energy
    #     Tempi[int(step/Nfreq)] = temperature(KE[int(step/Nfreq)],coords,nDim)     # instantaneous temperature
    #     Upot[int(step/Nfreq)] = potentialEnergy(coords,L,cutoff,sigma,epsilon,number_density,Tempi)   # potential energy


    #     print('%i %.4f %.4f %.4f %.4f' %(step, Tempi[int(step/Nfreq)], KE[int(step/Nfreq)], Upot[int(step/Nfreq)], KE[int(step/Nfreq)]+Upot[int(step/Nfreq)]))
    #     with open('log.dat', 'a') as file:
    #         file.write('%i %.4f %.4f %.4f %.4f\n' %(step, Tempi[int(step/Nfreq)], KE[int(step/Nfreq)], Upot[int(step/Nfreq)], KE[int(step/Nfreq)]+Upot[int(step/Nfreq)]))

    #     # write a file with atom coordinates
    #     write_frame(coords, L, vels, forces, 'trajectory', step)


    KE[int(step/Nfreq)] = kineticEnergy(vels,mass)   # kinetic energy
    Tempi[int(step/Nfreq)] = temperature(KE[int(step/Nfreq)],coords,nDim)     # instantaneous temperature
    Upot[int(step/Nfreq)] = potentialEnergy(coords,L,cutoff,sigma,epsilon,number_density,Tempi)   # potential energy


    print('%i %.4f %.4f %.4f %.4f' %(step, Tempi[int(step/Nfreq)], KE[int(step/Nfreq)], Upot[int(step/Nfreq)], KE[int(step/Nfreq)]+Upot[int(step/Nfreq)]))

# for post processing (make a new file)
# # read trajectory files
# xyz, vel, forces = read_lammps_trj(filename)
#
# # loop over trajectory file to get rdf
# gr = np.zeros(nbins)
# for key in xyz.keys():
#     r, temp = rdf(xyz[key], L, nbins, (0, 10.0))
#     gr += temp
# gr /= len(xyz.keys())
#
# # read xyz files
# xyz, npart = read_xyz(filename)
# xyz = read_xyz_trj(filename)





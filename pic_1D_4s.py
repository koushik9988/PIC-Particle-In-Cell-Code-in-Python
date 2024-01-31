"""
1D-1V particle in cell simulation code in python. The code is normalized with respect to electron debye lenght and electron plasma frequency.

"""
import numpy as np
import matplotlib.pyplot as plt
from random import seed, random
import time
import configparser
import os
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from numba import jit
import shutil
#from numba import njit, prange

start_time = time.time()

output= "output_data"

# Delete the output folder
print("Deleting the output folder ...")
shutil.rmtree(output, ignore_errors=True)

# Create an output folder
os.mkdir(output)
os.mkdir(os.path.join(output, "files"))


config = configparser.ConfigParser()
config.read('input_1d.ini')

# get data from input file
Cell_no = config.getint('Grid', 'Cellno')
nParticlesI = config.getint('Particles', 'nparticlesI')  
nParticlesE = config.getint('Particles', 'nparticlesE')
nParticlesB = config.getint('Particles', 'nparticlesB')  
nParticlesN = config.getint('Particles', 'nparticlesN')
tempI = config.getfloat('Particles', 'tempI')
tempE = config.getfloat('Particles', 'tempE')
tempB = config.getfloat('Particles', 'tempB')
tempN = config.getfloat('Particles', 'tempN')
AMU = config.getfloat('Constants', 'AMU')
massI = AMU*config.getfloat('Particles', 'massI')
massB = AMU*config.getfloat('Particles', 'massB')
massN = AMU*config.getfloat('Particles', 'massN')
massE = config.getfloat('Particles', 'massE')
density = config.getfloat('Particles', 'density')
chargeE = config.getfloat('Constants', 'chargeE')
eV = config.getfloat('Constants', 'eV')
EV_TO_K = config.getfloat('Constants', 'EV_TO_K')
K_b = config.getfloat('Constants', 'K_b')
DT = config.getfloat('Simulation', 'DT')
T = config.getint('Simulation', 'T')
v_e = config.getfloat('Simulation', 'v_e')
v_i = config.getfloat('Simulation', 'v_i')
v_b = config.getfloat('Simulation', 'v_b')
v_n = config.getfloat('Simulation', 'v_n')
delta_x = config.getfloat('Simulation', 'delta_x')
conv_check = config.getint('Simulation', 'conv_check')
epsilon = config.getfloat('Constants', 'epsilon')
interval = config.getint('Simulation','interval')
solver_type = config.get('Simulation', 'solver_type')
boundary_condition = config.get('Simulation', 'B.C')
u_left = config.getfloat('Simulation', 'u_left')
u_right = config.getfloat('Simulation', 'u_right')
inject_rate = config.getfloat('Simulation', 'inject_rate')



alpha = config.getfloat('Simulation', 'alpha')
beta = config.getfloat('Simulation', 'beta')


# no of grid points( no of grid point is one more than no of cells)
Nx = Cell_no + 1

rho = np.zeros(Nx)
efx = np.zeros(Nx)
efy = np.zeros(Nx)
u = np.zeros(Nx)

class Particle:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel

class Species:
    def __init__(self, name, mass, charge, spwt, num_particles, temperature):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.spwt = spwt
        self.num_particles = num_particles
        self.temperature = temperature
        self.den = np.zeros(Nx)
        self.part_list = []


ni0 = density
ne0 = ni0*(1- alpha - beta)
nb0 = alpha*ni0 # alpha is the fraction of beam negative ion to background positive  ion
nn0 = beta*ni0  # beta is the fraction of negative ion to background positive ion


wp = np.sqrt((ne0 * chargeE * chargeE) / (massE * epsilon))
f = (wp / (2 * np.pi))
LD = np.sqrt((epsilon * K_b * tempE * EV_TO_K) / (ne0 * chargeE * chargeE))
acoustic_speed = np.sqrt((K_b*tempE*EV_TO_K)/massI)


stepsize = LD
dx = LD / stepsize  # normalized spacing
dt = DT * (1 / wp)
dt = dt*wp
Lx = (Nx - 1) * dx

ion_spwt = (ni0*Lx) / nParticlesI
electron_spwt = (ne0*Lx) / nParticlesE
negion_spwt = (nn0*Lx) / nParticlesN
beam_spwt = (nb0*Lx) / nParticlesB

p1 = 0
p2 = 0  

ions = Species("ion", massI, chargeE, ion_spwt , nParticlesI, tempI)
electrons = Species("electrons", massE, -chargeE, electron_spwt,  nParticlesE, tempE)
negion = Species("negion", massN, -chargeE, negion_spwt,  nParticlesN, tempN)
beam = Species("beam", massB, -chargeE, beam_spwt,  nParticlesB, tempB)

species_list = [ions, electrons,negion,beam]

ions = species_list[0]
electrons = species_list[1]
negion = species_list[2]
beam = species_list[3]

print("ion acoustic speed is ", acoustic_speed)
print("time period :", 1 / f)
print("Electron debye lenght  : ", LD)
print("No of cell  :", Cell_no)
print("system lenght : ", Lx)
print("time step is  ",dt)

#---------------------------------------------
#-------------------------------------------------
@jit(nopython=True)
def gather(data, lc):
    i = int(np.trunc(lc))
    di = lc- i

    return (data[i] * (1 - di) + data[i + 1] * di)

@jit(nopython=True)
def scatter(data, lc, value):
    i = int(np.trunc(lc))
    di = lc- i

    data[i] += (1 - di) * value
    data[i + 1] += di * value

@jit(nopython=True)
def XtoL(pos):
    lc = pos / dx
    return lc

@jit(nopython=True)
def Pos(lc):
    pos = lc * dx
    return pos

# Both the fucntion pos and xtoL are not needed  when the  code is normalized
#@jit(nopython=True)
def ScatterSpecies(species):
    p = 0 
    while p < species.num_particles:
        part = species.part_list[p]
        lc = XtoL(part.pos)
        scatter(species.den, lc, species.spwt)
        p += 1
        #print(p)

    species.den = species.den / dx   # divide by cell volume
    species.den = species.den /density

    if boundary_condition == 'pbc':
        species.den[0] += species.den[-1]
        species.den[-1] = species.den[0]

    elif boundary_condition == 'open':
        species.den[0] *= 2
        species.den[-1] *= 2
    #print(species.den)

    #return lc

#@jit(nopython=True)
def compute_rho(rho, ions, electrons, negion, beam):
    rho[0:Nx] = ions.den[0:Nx] -  electrons.den[0:Nx] - negion.den[0:Nx] - beam.den[0:Nx]
  
    #rho = ions.charge * ions.den + electrons.charge * electrons.den
    #print(rho)
    return rho

#@jit(nopython=True)
def sampleVel(species):
    v_th = np.sqrt((K_b * species.temperature * EV_TO_K) / (species.mass))
    vel = np.sqrt(2) * v_th * ((random() + random() + random() - 1.5))  
    return vel


def Init(name):
    v_th = np.sqrt((K_b * electrons.temperature * EV_TO_K) / (electrons.mass))
    if name == 'electrons':
        for p in range(electrons.num_particles):
            pos = (p * Lx) / (electrons.num_particles) + delta_x * dx 
            #pos = Lx*random()
            vel = sampleVel(electrons) + v_e * v_th
            vel = vel / v_th
            electrons.part_list.append(Particle(pos, vel))
    elif name == 'ions':  
        for p in range(ions.num_particles):
            pos = (p * Lx) / (ions.num_particles) 
            #pos = Lx*random()
            vel = sampleVel(ions) + v_i * v_th
            vel = vel / v_th
            ions.part_list.append(Particle(pos, vel))
    elif name == 'negion':  
        for p in range(ions.num_particles):
            pos = (p * Lx) / (negion.num_particles) 
            #pos = Lx*random()
            vel = sampleVel(negion) + v_n * v_th
            vel = vel / v_th
            negion.part_list.append(Particle(pos, vel))
    elif name == 'beam':  
        for p in range(ions.num_particles):
            pos = Lx*0.98 #(p * Lx) / (beam.num_particles) 
            #pos = Lx*random()
            vel = sampleVel(beam) + v_b * v_th
            vel = vel / v_th
            beam.part_list.append(Particle(pos, vel))


@jit(nopython=True)
def solve_potential_iterative(u, rho):
    dx2 = dx * dx
    # boundary condition
    u[0] = u_left 
    u[-1] = u_right
    conv_threshold = 1e-2
    it = 0
    while it < 5000:
        u_prev = np.copy(u)
        u[1:Nx-1] = 0.5 *((u[2:Nx] + u[0:Nx-2]) + (dx2*rho[1:Nx-1]))

        if (it % conv_check == 0):
            R = - u[1:Nx - 1] * (2 / dx2)  + ((u[0:Nx - 2] + u[2:Nx]) + rho[1:Nx - 1] / epsilon)
            L2 = np.sqrt(np.sum(R**2) / (Nx))
            print(L2)
            if L2 < conv_threshold:
                break
    return L2


@jit(nopython=True)
def solve_potential_cg(u, rho):
    dx2 = dx * dx
    # boundary condition
    u[0] = u_left
    u[-1] = u_right
    
    # Construct the matrix for the linear system
    A = diags([-1, 2, -1], [-1, 0, 1], shape=(Nx-2, Nx-2)) / dx2

    # Right-hand side
    b = rho[1:Nx-1]

    # Solve using Conjugate Gradient
    u[1:Nx-1], _ = cg(A, b, tol=10e-2)

    return u

@jit(nopython=True)
def solve_potential_tridiagonal(x, rho):
    dx2 = dx * dx
    # Coefficients for the tridiagonal system
    a = np.ones(Nx)
    b = -2.0 * np.ones(Nx)
    c = np.ones(Nx)

    a[0] = 0
    b[0] = 1
    c[0] = 0
    a[Nx - 1] = 0
    b[Nx - 1] = 1
    c[Nx - 1] = 0

    # Multiply RHS
    x[1:Nx - 1] = -rho[1:Nx - 1] * dx2
    x[0] = u_left
    x[Nx - 1] = u_right

    c[0] /= b[0]
    x[0] /= b[0]

    for i in range(1, Nx):
        id = b[i] - c[i - 1] * a[i]
        c[i] /= id
        x[i] = (x[i] - x[i - 1] * a[i]) / id

    # Now back substitute
    for i in range(Nx - 2, -1, -1):
        x[i] = x[i] - c[i] * x[i + 1]

    return x

@jit(nopython=True)
def computeEF(u, efx):
    efx[1:Nx-1] =  (u[0:Nx-2] - u[2:Nx+1]) / (2 * dx)
    #efx[1:-1] = (u[0:-3] - u[2:-1]) / (2 * dx)
    
    efx[0] = - (u[-1] - u[1]) / (2*dx)
    efx[-1] = efx[0]#(u[-2] - u[-1]) / dx

#"""
def push_species(species, efx):
    wl = LD * LD * wp * wp

    for part in species.part_list:
        lc = XtoL(part.pos)
        qm = species.charge / species.mass
        part_ef = gather(efx, lc)

        #p1 = part.pos
        
        #part.vel += qm * part_ef * dt
        part.vel += (1 / wl) * chargeE * (qm * tempE / chargeE) * part_ef * dt
        part.pos += part.vel * dt

        #p2 = part.pos

        if part.pos< 0:
            part.pos += Lx
        elif part.pos >= Lx:
           part.pos -= Lx
           

def push_species_s(species, efx):
    wl = LD * LD * wp * wp

    # Create a new list with particles that stay within the domain
    new_part_list = []

    for part in species.part_list:
        lc = XtoL(part.pos)
        qm = species.charge / species.mass
        part_ef = gather(efx, lc)

        part.vel += (1 / wl) * chargeE * (qm * tempE / chargeE) * part_ef * dt
        part.pos += part.vel * dt

        # Check if the particle is within the domain
        if 0 <= part.pos <= Lx:
            new_part_list.append(part)
        else:
            species.num_particles -= 1  # Reduce num_particles when a particle leaves the domain

    # Update the part_list with particles that stay within the domain
    species.part_list = new_part_list


# function to check if partcles are leaving one complete cell in a sigle time step
def check(p1,p2):

    if(abs(p1-p2)>LD):
        print('particle is crossing one cell lenght')

    return abs((abs(p1-p2)) -LD)

#@jit(nopython=True)
def rewind_species(species,efx):
    wl = LD * LD * wp * wp
    for part in species.part_list:
        lc = XtoL(part.pos)
        qm = species.charge / species.mass
        part_ef = gather(efx, lc)
        
        part.vel -= 0.5*(1 / wl) * chargeE * (qm * tempE / chargeE) * part_ef * dt


def compute_ke(species):
    ke = 0
    for p in species.part_list:
        # Un-normalize the velocity by multiplying with the electron thermal velocity
        ke += (p.vel * p.vel)* (wp * LD) * (wp * LD)

    # Multiply 0.5 * mass for all particles
    ke *= 0.5 * (species.spwt * species.mass)
    Th = (electrons.temperature * EV_TO_K*K_b) * electrons.spwt * nParticlesE 
    ke /= Th
    
    return ke


def compute_pe():
    pe = 0

    ef_magnitude = np.sqrt(efx**2)
    ef_magnitude = np.sum(ef_magnitude)

    # Un-normalize the electric(species.den
    ef_magnitude = ef_magnitude * (massE * wp * wp * LD / chargeE)

    pe = 0.5 * epsilon * (ef_magnitude**2)
    Th = (electrons.temperature * EV_TO_K*K_b) * electrons.spwt * nParticlesE 
    pe /= Th

    return pe

@jit(parallel = True, nopython=True)
def write_particle_data(species, ts):
    species_name = species.name[0]
    file_path = os.path.join(output_directory, f"{species_name}{ts}.txt")

    with open(file_path, 'a') as file:
        for part in species.part_list:
            file.write(f"{part.pos} \t {part.vel}\n")


@jit(nopython=True)
def write_data(electrons, ions, negion, beam):
    data_folder = os.path.join(output_directory, "data")
    os.makedirs(data_folder, exist_ok=True)

    file_path = os.path.join(data_folder, "Results.txt")

    with open(file_path, 'a') as file:
        for i in range(len(u)):
            file.write(f"{i * dx:.4f} \t {electrons.den[i]:.4f} \t {ions.den[i]:.4f} \t {negion.den[i]:.4f} \t {beam.den[i]:.4f} \t {u[i]:.4f} \t {efx[i]:.4f}\n")
            #file.write(f"{i * dx:.4f}\t{electrons.den[i]:.4f}\t{ions.den[i]:.4f}\t{negion.den[i]:.4f}\t{beam.den[i]:.4f}\t{u[i]:.4f}\t{efx[i]:.4f}\n")


Init('ions')
Init('electrons')
Init('negion')
Init('beam')

rewind_species(electrons, efx)
rewind_species(ions, efx)
rewind_species(negion, efx)
rewind_species(beam, efx)

output_directory = "output_data"
os.makedirs(output_directory, exist_ok=True)

fig,(ax1,ax2) = plt.subplots(2,1)
x = np.linspace(0,Lx,Nx)

ti = []
dens_1 = []
dens_2 = []

for ts in range(0, T+1):
    ScatterSpecies(electrons)
    ScatterSpecies(ions)
    ScatterSpecies(negion)
    ScatterSpecies(beam)
    compute_rho(rho, ions, electrons,negion,beam)
    if solver_type == 'iterative':
        L2 = solve_potential_iterative(u,rho)
    elif solver_type == 'tridiagonal':
        solve_potential_tridiagonal(u,rho)
    elif solver_type == 'cg':
        solve_potential_cg(u,rho)
    else:
        raise ValueError(f"Invalid solver_type: {solver_type}. Choose 'iterative' or 'tridiagonal' or 'cg'.")
    computeEF(u, efx)
    if boundary_condition == 'pbc':
        push_species(electrons, efx)
        push_species(ions, efx)
        push_species(negion, efx)
        push_species(beam, efx)
    elif boundary_condition == 'open':
        push_species_s(electrons,efx)
        push_species_s(ions,efx)
        push_species_s(negion,efx)
        push_species_s(beam,efx)
    
    #k = check(p1,p2)
    kee = compute_ke(electrons)
    kei = compute_ke(ions)
    ken = compute_ke(negion)
    keb = compute_ke(beam)
    pe = compute_pe()
    #print(L2)
    
    ti.append(ts*dt/wp)
    #dens.append(electrons.den[50])
    dens_1.append(np.copy(electrons.den))
    dens_2.append(np.copy(ions.den))

    max_phi = u[0]
    for i in range(Nx):
        if u[i] > max_phi:
            max_phi = u[i]

    #print("TS: {}  nI: {} \t nE: {} \t  L2 norm: {} \t  max_phi: {}".format(
        #ts, len(ions.part_list), len(electrons.part_list), L2, max_phi))

    #ax.clear()
    if (ts % interval == 0):
        print("TS: {}\t nI: {} \t nE: {} \t nB: {}\t nN: {} \t ke_e: {:.4f} \t ke_i: {:.4f} \t ke_n: {:.4f} \t ke_b: {:.4f} \t total energy: {:.4f}  \t max_phi: {:.4f} ".format(
        ts, len(ions.part_list), len(electrons.part_list), len(beam.part_list),len(negion.part_list), kee, kei, ken, keb, kee + kei + pe, max_phi))

        
        #write_particle_data(electrons, ts)
        #write_particle_data(ions, ts)
        #write_data(electrons, ions, negion, beam)

ax1.plot(x, dens_1[-1],label = "beam")
ax1.set_xlabel("lenght")
ax1.set_ylabel("density")
ax1.legend()

ax2.plot(x, dens_2[-1],label ="negion")
ax2.set_xlabel("lenght")
ax2.set_ylabel("density")
ax2.legend()

#plt.plot(ti,dens)
plt.show()

end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"Time taken: {elapsed_time} minutes")

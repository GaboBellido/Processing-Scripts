import numpy as np
import pbc_utils as pbc
import math

class input_config: 
    def __init__(self, xbox, ybox, zbox):
        self.natoms = 0
        self.nbonds = 0
        self.nmasses = 0
        self.ndihedrals = 0
        self.nimpropers = 0
        self.masses = []
        self.ang_types = []
        self.bond_types = []
        self.ms_points = []
        self.bonds = np.array([], dtype=np.int64).reshape(0,4)
        self.nbond_types = 0
        self.nangles = 0
        self.nang_types = 0
        self.x = np.array([], dtype=np.int64).reshape(0,6)
        self.angles = np.array([], dtype=np.int64).reshape(0,5)
        self.RESID = np.zeros((0, 3), 'd')
        self.L = np.zeros(3, 'd')
        self.L[0] = float(xbox)
        self.L[1] = float(ybox)
        self.L[2] = float(zbox)
        self.lo = -(self.L)/2
        self.hi = (self.L)/2
        self.xlo = self.lo[0]
        self.ylo = self.lo[1]
        self.zlo = self.lo[2]
        self.xhi = self.hi[0]
        self.yhi = self.hi[1]
        self.zhi = self.hi[2]
        self.np_list = np.array([], dtype=np.int64).reshape(0,4)
        self.num_chns = 0
        self.periodic = False
        self.tags = []
        
    # Adds particle type if not already in self.masses
    def __add_particle_type(self, part_type):
        if ( part_type not in  self.masses and part_type is not None ):
            self.masses.append(part_type)
            self.nmasses += 1
    
    # Adds bond type if not already in self.bond_types
    def __add_bond_type(self, bond_type):
        if ( bond_type not in self.bond_types and bond_type is not None):
            self.bond_types.append(bond_type)
            self.nbond_types += 1

    # Adds angle type if not already in self.ang_types
    def __add_angle_type(self, angle_type):
        if ( angle_type not in self.ang_types and angle_type is not None):
            self.ang_types.append(angle_type)
            self.nang_types += 1

    # Updates number of particles in simulation
    def __update_particle_count(self, count_new_atoms):
        self.natoms += count_new_atoms
    # Updates number of chains in simulation
    def __update_chain_count(self, count_new_chains):
        self.num_chns += count_new_chains 

    # Adds a bonded particle in a random direction
    def __add_bond_check_bond_overlap(self,loc_array, index, monomer_increment, Lbond, rmin, old_index = None, rad = None, direction = None):
        if old_index == None:
            old_index = index - 1
        theta = 2 * np.pi * np.random.random_sample()
        phi = np.pi * np.random.random_sample()

        
        if direction == None:
            theta = 2 * np.pi * np.random.random_sample()
            phi = np.arccos(1- 2* np.random.random_sample())
            direction = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])

        # Location of new bonded particle
        new_loc = direction * Lbond + loc_array[old_index,3:6]



        # if the box is periodic, force the new position within it without using PBC
        if self.periodic:
            if new_loc[0] > self.xhi:
                new_loc[0] -= self.L[0]
            if new_loc[1] > self.yhi:
                new_loc[1] -= self.L[1]
            if new_loc[2] > self.zhi:
                new_loc[2] -= self.L[2]
            if new_loc[0] < self.xlo:
                new_loc[0] += self.L[0]
            if new_loc[1] < self.ylo:
                new_loc[1] += self.L[1]
            if new_loc[2] < self.zhi:
                new_loc[2] += self.L[2]
            
        loc_array[index, 3:6] = new_loc
        


    # Adds diblocks until desired volume fraction is reached
    def add_diblock_rho0(self, part1, part2, frac, chl, rho0, Lbond, bond_type, rmin = 0.0):
        num_chns = int(self.L[0] * self.L[1] * self.L[2] * rho0/chl)
        self.add_diblock(part1, part2, frac, chl, num_chns, Lbond, bond_type, rmin)

    # Adds a given number of diblock polymer chains
    def add_diblock(self, part1, part2, frac, chl, num_chns, Lbond,bond_type, rmin = 0.0, rad = None, tag = None):
        self.__add_particle_type(part1)
        self.__add_particle_type(part2)
        if (chl > 1): 
            self.__add_bond_type(bond_type)

        # resid = self.natoms + 1
        # Array for atom ids, chain ids, particle types, and locations
        ns_loc = chl * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')

        # Array for bond number, bond type, current bonded atom id, previous bonded atom id
        nbonds_loc = num_chns * (chl - 1)
        bond_loc = np.empty((nbonds_loc,4), int)

        atom_id = self.natoms
        atom_id -= 1
        molecule_len = chl 

        self.__update_particle_count(molecule_len*num_chns)

        chn_id = self.num_chns
        self.num_chns += num_chns
        bond_count = 0

        # Loop through addition of chains and bonded particles
        for i_ch in range(num_chns):
            for i_monomer in range(chl):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer
                
                # Assign particle type matching diblock fraction
                if float(i_monomer)/float(chl) < frac:
                    xloc[tmp_index,2] = part1
                else:
                    xloc[tmp_index,2] = part2

                # Atom and chain id assignment
                xloc[tmp_index, 0] = atom_id
                xloc[tmp_index, 1] = chn_id + i_ch

                # If first particle in chain, assign random location, otherwise update bond count and add bonded particle in random direction
                if i_monomer == 0:
                    xloc[tmp_index, 3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index, 4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index, 5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = bond_type
                    bond_loc[bond_count, 2] = atom_id
                    bond_loc[bond_count, 3] = atom_id - 1 
                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)

        # Store new particle locations and bond data
        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])

    # Add specified amount of homopolymers
    def add_homopolymer(self, part, chl, num_chns, Lbond, bond_type):
        self.add_diblock(part, part, 1.0, chl, num_chns, Lbond, bond_type)
    
    # Add single nanoparticle 
    def add_np(self, part, num_part, radius):
        self.add_diblock(part, part, 1.0, 1, num_part, Lbond=0, bond_type=None, rad = radius)

    # Add homopolymers up to desired volume fraction
    def add_homopolymer_rho0(self, part, chl, rho0, Lbond, bond_type):
        num_chns = int(self.L[0] * self.L[1] * self.L[2] * rho0/chl)
        self.add_diblock(part, part, 1.0, chl, num_chns, Lbond, bond_type)
        
    # Add comb polymer, Nb is length of backbone, Ns is length of side chain, part1/2 and pt1/2 are particle types, bonds are bond types
    def add_comb_rho0(self, bb_part1, Nb,Ns, rho0, ss_pt1, back_bond, bb_part2=None, frac_bb=1, ss_pt2=None,
            frac_side=1.0, Lbond=1.0, freq=1,
            back_side_bond=None, side_bond=None, rmin = 0.0):

        num_chns = int(self.L[0] * self.L[1] * self.L[2] * rho0 / (Nb + math.ceil(float(Nb)/freq ) * Ns))
        self.add_comb(bb_part1, Nb, Ns, num_chns, ss_pt1, back_bond, bb_part2=bb_part2, frac_bb=frac_bb, ss_pt2=ss_pt2,
                frac_side = frac_side, Lbond = Lbond, freq = freq, 
                back_side_bond = back_side_bond, side_bond = side_bond, rmin = rmin)

    def add_comb(self, bb_part1, Nb,Ns, num_chns, ss_pt1, back_bond, bb_part2=None, frac_bb=1, ss_pt2=None,
            frac_side=1.0, Lbond=1.0, freq=1,
            back_side_bond=None, side_bond=None, rmin = 0.0):
        self.__add_particle_type(bb_part1)
        self.__add_particle_type(bb_part2)
        self.__add_particle_type(ss_pt1)
        self.__add_particle_type(ss_pt2)
        
        self.__add_bond_type(back_bond)
        self.__add_bond_type(back_side_bond)
        self.__add_bond_type(side_bond)

        # If no bond type specified for backbone-side chain or side chain-side chain, use backbone bond type.
        if side_bond == None:
            side_bond = back_bond
        if back_side_bond == None:
            back_side_bond = back_bond


        # resid = self.natoms + 1
        # Array for atom ids, chain ids, particle types, and locations
        ns_loc = int((Nb + Ns * Nb//freq) * num_chns)
        xloc =  np.zeros((ns_loc, 6), 'd')

        old_natoms = self.natoms

        # Array for bond number, bond type, current bonded atom id, previous bonded atom id
        nbonds_loc = int(num_chns * ( (Nb - 1) + Nb//freq * (Ns) ))
        bond_loc = np.empty((nbonds_loc,4), int)

        atom_id = self.natoms
        atom_id -= 1
        # Number of particles per molecule
        molecule_len =  int(Nb + Ns * int(Nb//freq))

        self.__update_particle_count( molecule_len * num_chns)

        chn_id = self.num_chns
        self.num_chns += num_chns
        bond_count = 0
        # Loop through addition of chains and monomers
        for i_ch in range(num_chns):
            # Create backbone
            for i_monomer in range(Nb):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer
                if float(i_monomer)/float(Nb) < frac_bb:
                    xloc[i_ch*molecule_len+i_monomer,2] = bb_part1
                else:
                    xloc[i_ch*molecule_len+i_monomer,2] = bb_part2

                xloc[tmp_index,0] = atom_id
                xloc[tmp_index,1] = chn_id + i_ch # molecule id 
                if i_monomer == 0:
                    xloc[tmp_index,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index,5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = back_bond 
                    bond_loc[bond_count, 2] = atom_id - 1
                    bond_loc[bond_count, 3] = atom_id 

                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)
            # Add side chains 
            for i_side_chain in range(int(Nb//freq)):
                i_monomer = int(i_side_chain * freq)
                indbb = i_ch * molecule_len + i_monomer + 1
                for i_side in range(Ns): 
                    atom_id += 1

                    tmp_index = int(i_ch * molecule_len + Nb + i_side_chain * Ns + i_side)
                    xloc[tmp_index,0] = atom_id 
                    xloc[tmp_index,1] = chn_id + i_ch # molecule id 
                    # Assign appropriate type to side chain particles
                    if float(i_side)/float(Ns) < frac_side:
                        xloc[tmp_index,2] = ss_pt1
                    else:
                        xloc[tmp_index,2] = ss_pt2
                    # If first side chain particle, bond to backbone, else bond to side chain
                    if i_side == 0:
                        bond_loc[bond_count, 0] = self.nbonds
                        bond_loc[bond_count, 1] = back_side_bond 
                        bond_loc[bond_count, 2] = indbb + old_natoms
                        # bond_loc[bond_count, 3] = tmp_index 
                        bond_loc[bond_count, 3] = atom_id
                        bond_count += 1
                        self.nbonds += 1
                        self.__add_bond_check_bond_overlap(xloc, tmp_index, i_side+1, Lbond, rmin, old_index = indbb-1 )

                    else:
                        bond_loc[bond_count, 0] = self.nbonds
                        bond_loc[bond_count, 1] = side_bond 
                        bond_loc[bond_count, 2] = atom_id- 1
                        bond_loc[bond_count, 3] = atom_id
                        bond_count += 1
                        self.nbonds += 1
                        self.__add_bond_check_bond_overlap(xloc, tmp_index, i_side+1, Lbond, rmin)
        # Add particle locations and bond information
        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])
    # Add ABA triblocks up to desired volume fraction
    def add_simple_ABA_rho0(self,part1, part2, fracA, chl, rho0, Lbond=1.0, bond_type=1, rmin = 0.0):
        self.add_triblock(part1, part2, part1, fracA, 1-2*fracA, chl, int(self.L[0] * self.L[1] * self.L[2] * rho0/chl), Lbond = Lbond, 
                bond_type12 = bond_type, bond_type23= bond_type,rmin=rmin)
    # Add desired number of ABA triblocks
    def add_simple_ABA(self,part1, part2, fracA, chl, num_chns, Lbond=1.0, bond_type=1, rmin = 0.0):
        self.add_triblock(part1, part2, part1, fracA, 1-2*fracA, chl, num_chns, Lbond = Lbond, 
                bond_type12 = bond_type, bond_type23= bond_type,rmin=rmin)
    # Add triblocks up to desired volume fraction
    def add_triblock_rho0(self,part1, part2, part3, frac1, frac2, chl, rho0, Lbond=1.0, bond_type12=1, bond_type23=1, rmin = 0.0):
        self.add_triblock(part1, part2, part3, frac1, frac2, chl, int(self.L[0] * self.L[1] * self.L[2] * rho0/chl), Lbond = Lbond, 
                bond_type12 = bond_type12, bond_type23= bond_type23, rmin=rmin)
    # Add a specified number of triblocks
    def add_triblock(self, part1, part2, part3, frac1, frac2, chl, num_chns, Lbond=1.0,bond_type12=1, bond_type23=1, rmin = 0.0):
        self.__add_particle_type(part1)
        self.__add_particle_type(part2)
        self.__add_particle_type(part3)

        self.__add_bond_type(bond_type12)
        self.__add_bond_type(bond_type23)

        # Array for atom ids, chain ids, particle types, and locations
        ns_loc = chl * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')

        # Array for bond number, bond type, current bonded atom id, previous bonded atom id
        nbonds_loc = num_chns * (chl - 1)
        bond_loc = np.empty((nbonds_loc,4), int)
        
        molecule_len = chl

        atom_id = self.natoms

        self.natoms += chl * num_chns
        chn_id = self.num_chns
        self.num_chns += num_chns
        bond_count = 0
        # Loop through creation of triblock
        for i_ch in range(num_chns):
            for i_monomer in range(chl):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer

                f_along = float(i_monomer)/float(chl) 
                if f_along < frac1:
                    xloc[tmp_index,2] = part1
                elif f_along < frac1+frac2:
                    xloc[tmp_index,2] = part2
                else:
                    xloc[tmp_index,2] = part3

                xloc[tmp_index,0] = atom_id 
                xloc[tmp_index,1] = chn_id + i_ch
                if i_monomer == 0:
                    xloc[tmp_index,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index,5] = self.zlo + np.random.random_sample() * self.L[2]

                else:
                    if f_along >= frac1 + frac2:
                        bndtyp = bond_type23
                    else: 
                        bndtyp = bond_type12

                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = bndtyp
                    bond_loc[bond_count, 2] = atom_id
                    bond_loc[bond_count, 3] = atom_id - 1 

                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)
        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])


    # Writes LAMMPS initial configuration file
    def write(self, output, output_lc = None):

        if output_lc is not None:
            self.write_lc(output_lc)

        otp = open(output, 'w')
        otp.write("Generated by Chris' code\n\n")
        
        line = "%d atoms\n" % (self.natoms  )
        otp.write(line)
        line = "%d bonds\n" % len(self.bonds)
        otp.write(line)
        line = "%d angles\n" % (self.nangles)
        otp.write(line)
        # line = "%d dihedrals\n" % (self.ndihedrals)
        # otp.write(line)
        # line = "%d impropers\n" % (self.ndihedrals)
        # otp.write(line)
        line = "\n" 
        otp.write(line)

        line = "%d atom types\n" % len(self.masses)
        otp.write(line)
        line = "%d bond types\n" % np.max([len(self.bond_types), np.max(self.bond_types)])
        otp.write(line)
        tmp_val = len(self.ang_types)
        if tmp_val > 0: 
            tmp_val = np.max([tmp_val, np.max(self.ang_types)])
        line = "%d angle types\n" % tmp_val
        otp.write(line)
        # line = "%d dihedral types\n" % self.ndihedrals
        # otp.write(line)
        # line = "%d improper types\n" % self.nimpropers
        # otp.write(line)
        line = "\n" 
        otp.write(line)

        line = '%f  %f xlo xhi\n' % (self.lo[0], self.hi[0])
        otp.write(line)
        line = '%f  %f ylo yhi\n' % (self.lo[1], self.hi[1])
        otp.write(line)
        line = '%f  %f zlo zhi\n\n' % (self.lo[2], self.hi[2])
        otp.write(line)
        
        if len(self.masses) > 0 :
            otp.write("Masses \n\n")
        for i, val in enumerate(self.masses):
                    line = "%d 1.000\n" % (val)
                    otp.write(line)

        otp.write("\nAtoms \n\n")
        # Write atom ids, chain ids, particle types, and locations
        for i, val in enumerate(self.x):
                        line = "{:d} {:d} {:d} {:f} {:f} {:f}\n" 
                        idx,m,t,x,y,z = val
                        otp.write(line.format(int(idx)+1,int(m),int(t),x,y,z))
        # Write bond number, bond type, current bonded atom id, previous bonded atom id
        if len(self.bonds) > 0 :
            otp.write("\nBonds \n\n")

            for i, val in enumerate(self.bonds):
                line = "{:d} {:d} {:d} {:d}\n"
                idx,t,a,b = val
                otp.write(line.format(int(i)+1,int(t),int(a)+1,int(b)+1))
                # line = ' '.join(map(str, val))
                # otp.write(line + "\n")

        if len(self.angles) > 0 :
            otp.write("\nAngles\n\n")
            for i, val in enumerate(self.angles):
                line = "{:d} {:d} {:d} {:d} {:d}\n"
                idx,t,a,b,c = val
                otp.write(line.format(int(idx)+1,int(t),int(a)+1,int(b)+1,int(c)+1))
                # line = ' '.join(map(str, val))
                # otp.write(line + "\n")
    # Write points used for Maier Saupe potential in LC simulation (lc.input)
    def write_lc(self, output):
        otp = open(output, 'w')
        otp.write(str(len(self.ms_points)) + "\n")
        for i, val in enumerate(self.ms_points):
            line = '{:d} {:d} {:d}\n'
            otp.write(line.format(int(i+1), int(val[0]), int(val[1])))

    
    def add_diblock_angle(self, part1, part2, frac, chl, num_chns, Lbond,bond_type,angle_type = None, rmin = 0.0):
        self.__add_particle_type(part1)
        self.__add_particle_type(part2)
        self.__add_bond_type(bond_type)

        resid = self.natoms + 1
        ns_loc = chl * num_chns
        xloc =  np.zeros((ns_loc, 6), 'd')
        # bond_loc = np.zeros((0, 4), 'd')
        # bond_loc = np.([], dtype=np.float).reshape(0,4)
        nbonds_loc = num_chns * (chl - 1)
        bond_loc = np.empty((nbonds_loc,4), int)
        nangles_loc = num_chns * (chl -2 )
        bond_loc = np.empty((nangles_loc,4), int)
        # self.nbonds 
        natoms = self.natoms
        self.natoms += chl * num_chns
        chn_id = self.num_chns
        self.num_chns += chl
        bond_count = 0
        # Shouldn't it be angle_type instead of part2?
        if not angle_type == None:
            if ( not angle_type in self.ang_types):
                self.ang_types.append(part2)

        for i_ch in range(num_chns):
            for i_monomer in range(chl):
                natoms += 1
                if float(i_monomer)/float(chl) < frac:
                    xloc[i_ch*chl+i_monomer,2] = part1
                else:
                    xloc[i_ch*chl+i_monomer,2] = part2

                xloc[i_ch*chl+i_monomer,0] = natoms
                xloc[i_ch*chl+i_monomer,1] = chn_id + i_ch
                if i_monomer == 0:
                    xloc[i_ch*chl,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[i_ch*chl,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[i_ch*chl,5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = bond_type
                    bond_loc[bond_count, 2] = natoms
                    bond_loc[bond_count, 3] = natoms - 1 
                    bond_count += 1
                    self.nbonds += 1
                    theta = 2 * np.pi * np.random.random_sample()
                    phi = np.pi * np.random.random_sample()

                    dx = Lbond * np.sin(phi) * np.cos(theta)
                    dy = Lbond * np.sin(phi) * np.sin(theta)
                    dz = Lbond * np.cos(phi)

                    xprev = xloc[i_ch*chl+i_monomer-1,3]
                    yprev = xloc[i_ch*chl+i_monomer-1,4]
                    zprev = xloc[i_ch*chl+i_monomer-1,5]
                    

                    restriction = True
                    while restriction:
                        theta = 2 * np.pi * np.random.random_sample()
                        phi = np.pi * np.random.random_sample()

                        dx = Lbond * np.sin(phi) * np.cos(theta)
                        dy = Lbond * np.sin(phi) * np.sin(theta)
                        dz = Lbond * np.cos(phi)

                        xx = xprev + dx
                        yy = yprev + dy
                        zz = zprev + dz

                        if np.abs(zz) < self.L[2]/2. :
                            if i_monomer == 1:
                                restriction = False
                            else:
                                xpp = xloc[i_ch*chl+i_monomer-2,3]
                                ypp = xloc[i_ch*chl+i_monomer-2,4]
                                zpp = xloc[i_ch*chl+i_monomer-2,5]

                                dxp = xx - xpp
                                dyp = yy - ypp
                                dzp = zz - zpp

                                rpsq = dxp*dxp+dyp*dyp+dzp*dzp
                                rp = np.sqrt(rpsq)
                                if rp > rmin:
                                    restriction = False
                            
                                if self.periodic == True:
                                    if xx > self.xhi:
                                        xx -= self.L[0]
                                    if yy > self.yhi:
                                        yy -= self.L[1]
                                    if xx < self.xlo:
                                        xx += self.L[0]
                                    if yy < self.ylo:
                                        yy += self.L[1]

                    xloc[i_ch*chl+i_monomer,3] = xx
                    xloc[i_ch*chl+i_monomer,4] = yy
                    xloc[i_ch*chl+i_monomer,5] = zz
        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])


    def add_homo_LC(self, bb_part1, Nb,Ns, num_chns, ss_pt1, back_bond, bb_part2=None, frac_bb=1, ss_pt2=None,
            frac_side=1.0, Lbond=1.0, freq=1,
            back_side_bond=None, side_bond=None, rmin = 0.0):
        self.__add_particle_type(bb_part1)
        self.__add_particle_type(bb_part2)
        self.__add_particle_type(ss_pt1)
        self.__add_particle_type(ss_pt2)
        
        self.__add_bond_type(back_bond)
        self.__add_bond_type(back_side_bond)
        self.__add_bond_type(side_bond)

        if side_bond == None:
            side_bond = back_bond
        if back_side_bond == None:
            back_side_bond = back_bond

        # Array for atom ids, chain ids, particle types, and locations

        # resid = self.natoms + 1
        ns_loc = int((Nb + Ns * Nb//freq) * num_chns)
        xloc =  np.zeros((ns_loc, 6), 'd')

        old_natoms = self.natoms

        # Array for bond number, bond type, current bonded atom id, previous bonded atom id

        nbonds_loc = int(num_chns * ( (Nb - 1) + Nb//freq * (Ns) ))
        bond_loc = np.empty((nbonds_loc,4), int)

        atom_id = self.natoms

        molecule_len =  int(Nb + Ns * int(Nb//freq))

        self.__update_particle_count( molecule_len * num_chns)

        chn_id = self.num_chns
        self.num_chns += num_chns
        bond_count = 0
        # Loop through backbone and side chains
        for i_ch in range(num_chns):
            # Create backbone
            for i_monomer in range(Nb):
                atom_id += 1

                tmp_index = i_ch * molecule_len + i_monomer
                if float(i_monomer)/float(Nb) < frac_bb:
                    xloc[i_ch*molecule_len+i_monomer,2] = bb_part1
                else:
                    xloc[i_ch*molecule_len+i_monomer,2] = bb_part2

                xloc[tmp_index,0] = atom_id
                xloc[tmp_index,1] = chn_id + i_ch # molecule id 
                if i_monomer == 0:
                    xloc[tmp_index,3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index,4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index,5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 1] = back_bond 
                    bond_loc[bond_count, 2] = atom_id - 1
                    bond_loc[bond_count, 3] = atom_id 

                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)
            # Create side chains
            for i_side_chain in range(int(Nb//freq)):
                i_monomer = int(i_side_chain * freq)
                indbb = i_ch * molecule_len + i_monomer + 1
                for i_side in range(Ns): 
                    atom_id += 1

                    tmp_index = int(i_ch * molecule_len + Nb + i_side_chain * Ns + i_side)
                    xloc[tmp_index,0] = atom_id 
                    xloc[tmp_index,1] = chn_id + i_ch # molecule id 

                    if float(i_side)/float(Ns) < frac_side:
                        xloc[tmp_index,2] = ss_pt1
                    else:
                        xloc[tmp_index,2] = ss_pt2

                    if i_side == 0:
                        bond_loc[bond_count, 0] = self.nbonds
                        bond_loc[bond_count, 1] = back_side_bond 
                        bond_loc[bond_count, 2] = indbb + old_natoms
                        # bond_loc[bond_count, 3] = tmp_index 
                        bond_loc[bond_count, 3] = atom_id
                        bond_count += 1
                        self.nbonds += 1
                        self.__add_bond_check_bond_overlap(xloc, tmp_index, i_side+1, Lbond, rmin, old_index = indbb-1 )

                    else:
                        bond_loc[bond_count, 0] = self.nbonds
                        bond_loc[bond_count, 1] = side_bond 
                        bond_loc[bond_count, 2] = atom_id- 1
                        bond_loc[bond_count, 3] = atom_id
                        bond_count += 1
                        self.nbonds += 1
                        self.__add_bond_check_bond_overlap(xloc, tmp_index, i_side+1, Lbond, rmin)

        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])

    
    def add_pure_LC_rho0(self,  
        type1= None, type2 = None, typeLC=None,
        bond11 = None, bond12 = None, bond22 = None, bond1LC = None, bond2LC = None, bondLCLC = None,
        angle11 = None, angle12 = None, angle22 = None, angleLC = None,
        angle1LC = None, angle2LC = None, 
        attachment_type = None,
        initial_direction = None,
        LC_freq = None,
        rho = None,
        length_LC = None, Nb = None, nb_frac = None,
        smectic = False, lc_frac = 1, typeLC2 = None
        ): 
        if rho is not None:
            # if attachment_type == "headon_1" or attachment_type == "sideon_1":
            if "_1" in attachment_type:
                len_to_attach_LC = int(Nb * nb_frac)
                num_lc_per_mol = int( len_to_attach_LC / LC_freq)
                molecule_len = Nb + num_lc_per_mol * length_LC
            # elif attachment_type == "headon1_and_2" or attachment_type == "sideon1_and_2":
            elif "_2" in attachment_type:
                len_to_attach_LC = Nb
                num_lc_per_mol = int( len_to_attach_LC / LC_freq)
                molecule_len = Nb + num_lc_per_mol * length_LC
                

            if 'free_lc' in attachment_type:
                molecule_len = length_LC
                num_lc_per_mol = 1
                len_to_attach_LC = 0

            num_chns = int(rho * self.L[0] * self.L[1] * self.L[2] / molecule_len)
        
            self.add_pure_LC(type1, type2, typeLC,
                bond11, bond12, bond22, bond1LC, bond2LC, bondLCLC,
                angle11, angle12, angle22, angleLC,
                angle1LC, angle2LC, 
                attachment_type,
                initial_direction,
                LC_freq,
                num_chns,
                length_LC, Nb, nb_frac,
                smectic, lc_frac, typeLC2
                )
            

    def add_pure_LC(self,  
        type1= None, type2 = None, typeLC=None,
        bond11 = None, bond12 = None, bond22 = None, bond1LC = None, bond2LC = None, bondLCLC = None,
        angle11 = None, angle12 = None, angle22 = None, angleLC = None,
        angle1LC = None, angle2LC = None, 
        attachment_type = None,
        initial_direction = None,
        LC_freq = None,
        num_chns = None,
        length_LC = None, Nb = None, nb_frac = None,
        smectic = False,
        lc_frac = 1, typeLC2=None,
        ):

        attachment_types = ["headon_1", "headon1_and_2", "sideon_1", "sideon1_and_2", 'free_lc']
        if attachment_type not in attachment_types and not any(x in attachment_type for x in attachment_types):
            mesg = "attachment_type must be one of the following: " + str(attachment_types)
            raise ValueError(mesg)

        if initial_direction is not None and len(initial_direction) != 3:
            raise ValueError("initial_direction must be a list or array of length 3")


        # defining useful variables

        if "headon_1" in attachment_type or "sideon_1" in attachment_type:
            len_to_attach_LC = int(Nb * nb_frac)
            num_lc_per_mol = int( len_to_attach_LC / LC_freq)
            molecule_len = Nb + num_lc_per_mol * length_LC
        elif "headon1_and_2" in attachment_type or "sideon1_and_2" in attachment_type:
            len_to_attach_LC = Nb
            num_lc_per_mol = int( len_to_attach_LC / LC_freq)
            molecule_len = Nb + num_lc_per_mol * length_LC
        elif 'free_lc' in attachment_type:
            Nb = 0 
            len_to_attach_LC =1
            LC_freq = 1
            num_lc_per_mol = 0
            molecule_len = length_LC

        chn_id = self.num_chns
        self.num_chns += num_chns

        Lbond = 1
        rmin = 0.0

        # Create the arrays to store the data
        xloc = np.zeros([num_chns * molecule_len, 6], dtype=float)
        bond_loc = np.zeros([num_chns * molecule_len * 4, 4], dtype=int)
        angle_loc = np.zeros([num_chns * molecule_len * 6, 5], dtype=int)
        ms_loc = []

        angle_count = 0
        bond_count = 0
        base_atom_id = self.natoms
        atom_id = self.natoms 
        self.natoms += num_chns * molecule_len


        # Build up the data
        # Looping over the amount of chains
        for i_ch in range(num_chns):

            # Looping over the length of the chain
            for i_monomer in range(Nb):

                tmp_index = i_ch * molecule_len + i_monomer
                if float(i_monomer)/float(Nb) < nb_frac:
                    xloc[tmp_index,2] = type1
                    bond_tmp = bond11
                else:
                    xloc[tmp_index,2] = type2
                    bond_tmp = bond12 if bond_tmp == bond11 else bond22

                xloc[tmp_index, 0] = atom_id
                xloc[tmp_index, 1] = chn_id + i_ch
                if i_monomer == 0:
                    xloc[tmp_index, 3] = self.xlo + np.random.random_sample() * self.L[0]
                    xloc[tmp_index, 4] = self.ylo + np.random.random_sample() * self.L[1]
                    xloc[tmp_index, 5] = self.zlo + np.random.random_sample() * self.L[2]
                else:
                    bond_loc[bond_count, 0] = self.nbonds
                    bond_loc[bond_count, 2] = atom_id - 1
                    bond_loc[bond_count, 3] = atom_id 
                     
                    bond_loc[bond_count, 1] = bond_tmp

                    bond_count += 1
                    self.nbonds += 1

                    self.__add_bond_check_bond_overlap(xloc, tmp_index, i_monomer, Lbond, rmin)
                 
                atom_id += 1
                
            # Connect the angles along the polymer
            for i_monomer in range(2, Nb):

                tmp_index = i_ch * molecule_len + i_monomer

                if xloc[tmp_index, 2] == type1 and xloc[tmp_index-2, 2] == type1:
                    tmp_angle = angle11
                elif xloc[tmp_index, 2] == type2 and xloc[tmp_index-2, 2] == type2:
                    tmp_angle = angle22
                else:
                    tmp_angle = angle12

                if tmp_angle is not None:
                    angle_loc[angle_count, 0] = self.nangles
                    angle_loc[angle_count, 1] = tmp_angle
                    angle_loc[angle_count, 2] = tmp_index - 2
                    angle_loc[angle_count, 3] = tmp_index - 1
                    angle_loc[angle_count, 4] = tmp_index
                    angle_count += 1
                    self.nangles += 1



            # Attaching the Liquid Cystal to the chain
            for i_side_chain in range(num_lc_per_mol):
                i_monomer = int(i_side_chain * LC_freq)
                indbb = i_ch * molecule_len + i_monomer

                # Defining the atom type, chain id and molecule type
                # Must define positions elsewhere
                
                starting_lc_index = i_ch * molecule_len + Nb + i_side_chain * length_LC
                
                xloc[starting_lc_index:starting_lc_index  + length_LC, 2] = typeLC
                xloc[starting_lc_index:starting_lc_index  + length_LC, 1] = chn_id + i_ch
                xloc[starting_lc_index:starting_lc_index  + length_LC, 0] = np.arange(atom_id , atom_id + length_LC)

                # Random unit vector
                
                direction = np.random.random_sample(3)
                direction /= np.linalg.norm(direction)

                if "headon_1" in attachment_type or "headon1_and_2" in attachment_type:
                    link_point = 0 
                if "sideon_1" in attachment_type or "sideon1_and_2" in attachment_type:
                    link_point = length_LC // 2 

                xloc[starting_lc_index+link_point, 3:6] = xloc[indbb, 3:6] + direction 
                diff = np.arange(length_LC) - link_point

                if initial_direction is not None:
                    direction = initial_direction
                    displacement = (diff* np.array([direction]).T).T
                else:
                    direction = np.random.random_sample((3,length_LC)) -0.5
                    displacement = (diff* direction).T
    
                xloc[starting_lc_index:(starting_lc_index + length_LC), 3:6] = displacement  + xloc[starting_lc_index+link_point, 3:6]

                 
                # Defining the internal bonds
                bond_loc[bond_count:(bond_count + length_LC-1), 0] = np.arange(self.nbonds, self.nbonds + length_LC-1)
                bond_loc[bond_count:(bond_count + length_LC-1), 1] = bondLCLC
                bond_loc[bond_count:(bond_count + length_LC-1), 2] = np.arange(atom_id, atom_id + length_LC-1)
                bond_loc[bond_count:(bond_count + length_LC-1), 3] = np.arange(atom_id + 1, atom_id + length_LC)
                
                bond_count += (length_LC -1)
                self.nbonds += (length_LC -1)

                tmp_bond = bond1LC if xloc[indbb, 2] == type1 else bond2LC

                if tmp_bond is not None:
                    # Defining the linking bonds to the main chain
                    bond_loc[bond_count, 0] = self.nbonds 
                    bond_loc[bond_count, 1] = tmp_bond
                    bond_loc[bond_count, 2] = indbb + base_atom_id
                    bond_loc[bond_count, 3] = starting_lc_index + link_point + base_atom_id

                    bond_count += 1
                    self.nbonds += 1


                # Defining the angles

                if angleLC is not None:
                    angle_loc[angle_count:angle_count + length_LC-2, 0] = np.arange(self.nangles, self.nangles + length_LC-2)
                    angle_loc[angle_count:angle_count + length_LC-2, 1] = angleLC
                    angle_loc[angle_count:angle_count + length_LC-2, 2] = np.arange(atom_id, atom_id + length_LC-2)
                    angle_loc[angle_count:angle_count + length_LC-2, 3] = np.arange(atom_id + 1, atom_id + length_LC - 1)
                    angle_loc[angle_count:angle_count + length_LC-2, 4] = np.arange(atom_id + 2, atom_id + length_LC)

                    angle_count += length_LC - 2
                    self.nangles += length_LC - 2


                tmp_angle = angle1LC if xloc[indbb, 2] == type1 else angle2LC

                # End on attachment
                if "headon_1" in attachment_type or "headon1_and_2" in attachment_type:
                    
                    if tmp_angle is not None:
                        angle_loc[angle_count, 0] = self.nangles
                        angle_loc[angle_count, 1] = tmp_angle
                        angle_loc[angle_count, 2] = indbb
                        angle_loc[angle_count, 3] = starting_lc_index + link_point
                        angle_loc[angle_count, 4] = starting_lc_index + link_point + 1

                        angle_count += 1
                        self.nangles += 1

                # Side on attachment
                if "sideon_1" in attachment_type or "sideon1_and_2" in attachment_type:
                    if tmp_angle is not None: 
                        if (i_monomer - 1) >= 0:
                            angle_loc[angle_count, 0] = self.nangles
                            angle_loc[angle_count, 1] = tmp_angle
                            angle_loc[angle_count, 2] = indbb - 1 + base_atom_id
                            angle_loc[angle_count, 3] = indbb + base_atom_id
                            angle_loc[angle_count, 4] = starting_lc_index + link_point + base_atom_id

                            angle_count += 1
                            self.nangles += 1


                        if (i_monomer + 1) < Nb:
                            angle_loc[angle_count, 0] = self.nangles
                            angle_loc[angle_count, 1] = tmp_angle
                            angle_loc[angle_count, 2] = indbb + 1 + base_atom_id
                            angle_loc[angle_count, 3] = indbb  + base_atom_id
                            angle_loc[angle_count, 4] = starting_lc_index + link_point + base_atom_id

                            angle_count += 1
                            self.nangles += 1

                        angle_loc[angle_count, 0] = self.nangles
                        angle_loc[angle_count, 1] = tmp_angle
                        angle_loc[angle_count, 2] = starting_lc_index + link_point + base_atom_id -1
                        angle_loc[angle_count, 3] = starting_lc_index + link_point + base_atom_id
                        angle_loc[angle_count, 4] = indbb + base_atom_id

                        angle_count += 1
                        self.nangles += 1


                        angle_loc[angle_count, 0] = self.nangles
                        angle_loc[angle_count, 1] = tmp_angle
                        angle_loc[angle_count, 2] = starting_lc_index + link_point + base_atom_id + 1 
                        angle_loc[angle_count, 3] = starting_lc_index + link_point + base_atom_id
                        angle_loc[angle_count, 4] = indbb + base_atom_id

                        angle_count += 1
                        self.nangles += 1

                ms_loc.append([atom_id + length_LC // 2, atom_id + length_LC // 2 + 1])
                atom_id += length_LC



            # Pure LC data
            if (type1 is None and type2 is None) or (Nb == 0 or Nb is None) or (attachment_type == 'free_lc'):

                tmp_index = i_ch * molecule_len 

                xloc[tmp_index, 3] = self.xlo + np.random.random_sample() * self.L[0]
                xloc[tmp_index, 4] = self.ylo + np.random.random_sample() * self.L[1]
                xloc[tmp_index, 5] = self.zlo + np.random.random_sample() * self.L[2]

                diff = np.arange(length_LC) 

                if initial_direction is not None:
                    direction = initial_direction
                    displacement = (diff* np.array([direction]).T).T
                else:
                    direction = 2 * np.random.random_sample((3,length_LC)) -1 
                    displacement = (diff* direction).T

    

                xloc[tmp_index:tmp_index + length_LC, 3:6] = displacement + xloc[tmp_index, 3:6]
                if lc_frac == 1:
                    xloc[tmp_index:tmp_index  + length_LC, 2] = typeLC
                else:
                    xloc[tmp_index:tmp_index  + length_LC, 2] = (diff/molecule_len < lc_frac)*typeLC2 + (1-(diff/molecule_len < lc_frac))*typeLC

                xloc[tmp_index:tmp_index  + length_LC, 1] = chn_id + i_ch
                xloc[tmp_index:tmp_index  + length_LC, 0] = np.arange(atom_id , atom_id + length_LC)

                if smectic is True:
                    num_layers = self.L[2] // (length_LC + 1) 
                    xloc[tmp_index, 5] = np.random.randint(num_layers) * (length_LC + 1) + 0.5 + self.zlo + + np.random.random_sample()
                    xloc[tmp_index:tmp_index  + length_LC, 5] = xloc[tmp_index, 5] + diff
                 
                # Defining the internal bonds
                bond_loc[bond_count:bond_count + length_LC-1, 0] = np.arange(self.nbonds, self.nbonds + length_LC-1)
                bond_loc[bond_count:bond_count + length_LC-1, 1] = bondLCLC
                bond_loc[bond_count:bond_count + length_LC-1, 2] = np.arange(atom_id, atom_id + length_LC-1)
                bond_loc[bond_count:bond_count + length_LC-1, 3] = np.arange(atom_id + 1, atom_id + length_LC)
                
                bond_count += length_LC -1 

                if angleLC is not None:
                    angle_loc[angle_count:angle_count + length_LC-2, 0] = np.arange(self.nangles, self.nangles + length_LC-2)
                    angle_loc[angle_count:angle_count + length_LC-2, 1] = angleLC
                    angle_loc[angle_count:angle_count + length_LC-2, 2] = np.arange(atom_id, atom_id + length_LC-2)
                    angle_loc[angle_count:angle_count + length_LC-2, 3] = np.arange(atom_id + 1, atom_id + length_LC - 1)
                    angle_loc[angle_count:angle_count + length_LC-2, 4] = np.arange(atom_id + 2, atom_id + length_LC)

                    angle_count += length_LC - 2
                    self.nangles += length_LC - 2

                ms_loc.append([atom_id - (-length_LC // 2), atom_id - (-length_LC // 2) + 1])
                atom_id += length_LC






        # Delete rows if all values are zero 



        # Delete rows if bond_loc or angle_loc have a zero in any element
        bond_loc  =  bond_loc[~np.all(bond_loc == 0, axis=1)]
        angle_loc = angle_loc[~np.all(angle_loc == 0, axis=1)]

        for bonds in np.unique(bond_loc[:,1]):
            self.__add_bond_type(bonds)

        for angle in np.unique(angle_loc[:,1]):
            self.__add_angle_type(angle)

        for part_type in np.unique(xloc[:,2]):
            self.__add_particle_type(part_type)
            

        self.x = np.concatenate([self.x, xloc])
        self.bonds = np.vstack([self.bonds, bond_loc])
        self.angles= np.vstack([self.angles, angle_loc])
        self.ms_points += ms_loc

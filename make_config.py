import numpy as np
import lammps_initv2 as lmp
LC_frac = 1.0
Nb = 3
density = 3
box = [30.0, 30.0, 30.0]
#anchoring_frac = 0.50

filef = "input.data"
fileg = "lc.input"
box = lmp.input_config(box[0], box[1], box[2])
#box.add_homopolymer_rho0(1, Nb, density*(1-LC_frac), 1, 1)
box.add_pure_LC_rho0(typeLC= 1, bondLCLC = 1, angleLC = 1, attachment_type = "free_lc", rho = density*LC_frac, length_LC= Nb, lc_frac=2.0/3.0, typeLC2= 2)    
box.write(filef)
box.write_lc(fileg)

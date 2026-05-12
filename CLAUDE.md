# Processing-Scripts — Claude Code Context

Post-processing and analysis scripts for MATILDA.FT field-theoretic / hybrid particle-field simulations.

## Design Principles

### Multiprocessing for File Processing

When processing multiple independent files (position binaries, density binaries, etc.), use `multiprocessing.Pool` to parallelize. Choose the number of workers based on the file count: **4** if divisible by 4, **3** if divisible by 3, **2** otherwise. Extract the per-file logic into a standalone worker function (e.g., `_process_position_file`) that takes a single tuple of arguments and returns a list of result dicts. Use `pool.map()` over the task list and flatten results afterwards.

```python
from multiprocessing import Pool

def choose_n_workers(n_files):
    if n_files % 4 == 0: return 4
    elif n_files % 3 == 0: return 3
    else: return 2

n_workers = choose_n_workers(len(tasks))
with Pool(n_workers) as pool:
    all_results = pool.map(worker_fn, tasks)
results = [r for batch in all_results for r in batch]
```

---

## Jupyter Notebook Editing

**Never use the NotebookEdit tool.** When Jupyter has a notebook open, autosave overwrites disk changes silently. Always use a direct Python/JSON script via Bash:

```python
python3 << 'PYEOF'
import json

NB_PATH = "/home/gabriel/Processing-Scripts/<notebook>.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

cells = {c['id']: c for c in nb['cells']}
cells['<cell_id>']['source'] = ['...new source...']
cells['<cell_id>']['outputs'] = []
cells['<cell_id>']['execution_count'] = None

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
PYEOF
```

After writing, tell the user to reload: **File → Revert to Saved**.

---

## MATILDA.FT Output File Formats

The simulation code lives at `/home/gabriel/MATILDA.FT`. All output is controlled by frequency parameters (`bin_freq`, `log_freq`, `traj_freq`, etc.) with separate equilibration (`equil_*`) and production (`prod_*`) variants.

### Binary Density Grid (`*_densities.bin`)

The primary data source for structure factor and density analyses.

**Header** (written once):
| Field | Type | Count |
|-------|------|-------|
| Dim | int32 | 1 |
| Nx[0..Dim-1] | int32 | Dim |
| L[0..Dim-1] | float32 | Dim |
| ntypes | int32 | 1 |

**Per-frame** (appended each output step):
| Field | Type | Count |
|-------|------|-------|
| all_rho | float32 | M × ntypes |

where M = prod(Nx). Data layout: `all_rho[type * M + j]`, with j = ix + (iy + iz·Nx[1])·Nx[0] (ix-fastest, C-order). To match FFTW/numpy FFT, reshape as `(Nx[2], Nx[1], Nx[0])` (iz-slowest, ix-fastest).

**Typical values:** Dim=3, Nx=81, L=52.5, ntypes=3 (type 0 = wetting polymer, type 1 = non-wetting polymer, type 2 = nanoparticles).

### Binary Positions (`positions.bin`, `equil_positions.bin`)

**Header** (written once):
| Field | Type | Count |
|-------|------|-------|
| ns | int32 | 1 |
| Dim | int32 | 1 |
| L[0..2] | float32 | 3 (always 3, even in 2D) |
| tp[0..ns-1] | int32 | ns (particle types) |
| molecID[0..ns-1] | int32 | ns (molecule IDs) |
| charges[0..ns-1] | float32 | ns (only if do_charges=1) |

**Per-frame** (appended):
| Field | Type | Count |
|-------|------|-------|
| positions | float32 | ns × Dim |

Layout: `positions[particle * Dim + d]`.

### Binary Order Parameter (`order_parameter.bin`, `equil_order_parameter.bin`)

Nematic order parameter from the Maier-Saupe tensor potential.

**Header** (written once):
| Field | Type | Count |
|-------|------|-------|
| Dim | int32 | 1 |
| Nx[0..Dim-1] | int32 | Dim |
| L[0..Dim-1] | float32 | Dim |
| M | int32 | 1 |

**Per-frame** (appended):
| Field | Type | Count |
|-------|------|-------|
| step | int32 | 1 |
| lambda[0..M-1] | float32 | M |

Lambda is the normalized max eigenvalue of the S-tensor at each grid point, range [0, 1]. Physical max = (Dim-1)/Dim.

### GSD Trajectory (`traj.gsd`)

Standard [GSD format](https://gsd.readthedocs.io/) v1.4. Contains topology (first frame) + per-frame positions and optional charges. Readable with `gsd` and `freud` Python packages.

Key fields per frame:
- `particles/position` (float32, N×3)
- `particles/typeid` (uint32, N)
- `log/particles/moleculeid` (uint32, N)
- `bonds/group` (uint32, Nbonds×2), `angles/group` (uint32, Nangles×3)

### LAMMPS Trajectory (`traj.lammpstrj`, `equil.lammpstrj`)

Standard LAMMPS dump text format:
```
ITEM: TIMESTEP
<step>
ITEM: NUMBER OF ATOMS
<ns>
ITEM: BOX BOUNDS pp pp pp
0 Lx
0 Ly
0 Lz
ITEM: ATOMS id type mol x y z [q]
...
```

Types are 1-indexed (type+1), molecule IDs are 1-indexed (mol+1).

### Energy / Virial Log (`data.dat`, `equil_data.dat`)

Space-separated text with header row:
```
step  Upe  Ubond  [Uangle]  [Pxx Pyy [Pzz] [Pxy [Pxz Pyz]]]  [Ugauss_i...]  [Uerf_i...]  [UMS_i...]
```

- `Upe`: potential energy per volume (Upe/V)
- `Ubond`: total bonded energy
- `Uangle`: angle energy (if angles exist)
- `Pxx, Pyy, ...`: pressure tensor components (if computed)
- Per-potential energy columns for each Gaussian, Erf, GaussianErf, and Maier-Saupe potential

### Structure Factor Text Files

**End-of-simulation average** (`sk0.dat`, `sk1.dat`, ...): 7 columns:
```
kx  ky  [kz]  |S(k)|  k_magnitude  Re[S(k)]  Im[S(k)]
```

**Per-frame** (`Sk_<id>_step_<step>.dat`): same format, from Compute_sk.

### FTS-Specific Output

- `fts_data.dat`: step, effective Hamiltonian (real, [imag]), per-potential terms, -nmolecs·log(Q) terms
- Complex field grids (`.dat`): `x y [z] Re Im` per grid point

### Order Parameter CSV (`order_parameter_step_<step>.csv`)

Per-frame text files:
```
x,y,z,lambda
<ix>,<iy>,<iz>,<eigenvalue>
```

### Compute-Specific Files

- `Widom<id>.dat`: one Boltzmann factor per line (Widom insertion)
- `chemPotential<id>.dat`: one energy difference per line
- `avg_sk_<id>.dat`: time-averaged S(k), same 7-column format
- `deltaU_<sys>_<dir>_areaD_<val>.dat`: Bennett acceptance ratio energy differences

---

## Simulation Conventions

- **Particle types**: For Nanoparticle confinement project - 0 = wetting polymer, 1 = non-wetting polymer, 2 = nanoparticles (project-specific; not hardcoded in MATILDA). For Liquid Crystal Phase Separation project - 0 = isotropic fluid (gaussian polymer chain), 1 = liquid crystal, 2 = liquid crystal with modified anchoring potential (not always present). For study of liquid crystalline order in pure liquid crystal system - 0 = liquid crystal 
- **Grid storage order**: ix-fastest (Fortran-like innermost loop), matching `j = ix + (iy + iz*Nx[1])*Nx[0]`
- **FFT compatibility**: reshape density as `tuple(Nx[::-1])` so numpy's C-order FFT matches FFTW
- **DC mode**: k=0 is the mean density; exclude with `k_mag > 0` when computing S(k)
- **FFT normalization**: `np.fft.fftn(rho) / M` matches MATILDA's FFTW normalization
- **Timestep conversion**: simulation time t = step × 0.005 (typical dt)
- **Logarithmic output scheduling**: frames spaced as `freq *= (max_steps/freq)^(1/n_frames)` each iteration
- **Shell-averaging S(k)**: use bin width dk = 2π/L (Fourier resolution of the box)

## Common Analysis Parameters (Confined PS Project)

These are typical values; always verify against the specific binary file header.

```python
L_BOX     = 52.5      # box side length
Nx        = 81        # grid points per axis
max_steps = 2000001   # total simulation steps
n_frames  = 33        # logarithmic output frames
freq      = 1000      # initial output frequency
min_k     = 2 * np.pi / L_BOX  # ≈ 0.1197
```

## File Naming Conventions (Confined PS Project)

Standard production runs:
```
PS_in_NP_frac_{phi_np}_rad_{rad}_POT_{pot}_Nb_{nb}_rep_{rep}_densities.bin
```

Symmetric-reset runs (restarted from POT=10 equilibrated state):
```
sym_reset_PS_in_NP_frac_{phi_np}_rad_{rad}_POT_{pot}_Nb_{nb}_rep_{rep}_densities.bin
```

Parameters: phi_np ∈ {0.5, 0.64}, rad ∈ {3.75, 5.0}, POT ∈ {10.0, 13.75, 17.5, 21.25, 25.0}, Nb ∈ {5, 10, 20}, rep ∈ {0, 1, 2}.

## Source Code Reference

Key MATILDA.FT source files for output formats:
- `src/io_utils.cu` — grid density, positions, S(k) binary/text output
- `src/config_utils.cu` — LAMMPS trajectory, GSD trajectory
- `src/tensor_potential_MaierSaupe.cu` — order parameter output
- `src/FTS_Box.cu` — field-theoretic simulation output
- `src/Compute_*.cu` — specialized compute output (Widom, chem. potential, Bennett, avg S(k))
- `src/globals.h` — global variables and configuration

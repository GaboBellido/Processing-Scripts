#!/usr/bin/env python3
"""
Morphology Analysis for MATILDA.FT Density Fields
===================================================
Reads binary density output from MATILDA.FT (grid_densities.bin),
applies density thresholding, and computes interfacial curvature
metrics to classify phase-separated morphologies (spheres, cylinders,
lamellae, bicontinuous networks).

Key metrics computed:
  - Minkowski functionals (volume, surface area, mean curvature, Euler characteristic)
  - Principal curvature distributions at the interface
  - Interfacial shape distribution (ISD)
  - Scalar morphology descriptors suitable for Gaussian process modeling

Usage:
  python matilda_morphology_analysis.py grid_densities.bin [--frame -1] [--type 0] [--threshold auto]

Dependencies:
  numpy, scipy, scikit-image (for marching_cubes)
  Optional: matplotlib (for plotting)
"""

import numpy as np
import struct
import argparse
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# 1. BINARY FILE READER — matches io_utils.cu format
# =============================================================================

@dataclass
class DensityFrame:
    """Container for a single density field frame from MATILDA.FT."""
    dim: int
    nx: np.ndarray       # grid points per dimension, shape (dim,)
    L: np.ndarray        # box lengths per dimension, shape (dim,)
    ntypes: int
    densities: dict      # {type_index: np.ndarray of shape (nx[2], nx[1], nx[0]) or (nx[1], nx[0])}
    dx: np.ndarray       # grid spacing per dimension


def read_density_binary(filename: str, frame: int = -1) -> DensityFrame:
    """
    Read MATILDA.FT grid_densities.bin file.

    Binary layout (from io_utils.cu init_binary_output / write_binary):
      Header:
        int    Dim          — dimensionality (2 or 3)
        int[Dim] Nx         — grid points in each dimension
        float[Dim] L        — box lengths in each dimension
        int    ntypes       — number of particle types

      Per frame (appended by write_binary):
        float[M * ntypes]   — density data, where M = product(Nx)
                              stored as ntypes contiguous blocks of M floats

    Parameters
    ----------
    filename : str
        Path to grid_densities.bin
    frame : int
        Which frame to read. -1 = last frame.

    Returns
    -------
    DensityFrame
    """
    with open(filename, 'rb') as f:
        # --- Read header ---
        dim = struct.unpack('i', f.read(4))[0]

        nx = np.array(struct.unpack(f'{dim}i', f.read(4 * dim)), dtype=int)
        L = np.array(struct.unpack(f'{dim}f', f.read(4 * dim)), dtype=float)
        ntypes = struct.unpack('i', f.read(4))[0]

        header_size = f.tell()

        # Total grid points
        M = int(np.prod(nx))
        frame_size = M * ntypes * 4  # each float is 4 bytes

        # Determine number of frames
        f.seek(0, 2)  # seek to end
        file_size = f.tell()
        data_size = file_size - header_size
        n_frames = data_size // frame_size

        if n_frames == 0:
            raise ValueError("No density frames found in file.")

        # Handle negative indexing
        if frame < 0:
            frame = n_frames + frame
        if frame < 0 or frame >= n_frames:
            raise IndexError(f"Frame {frame} out of range [0, {n_frames})")

        print(f"Binary file info: dim={dim}, nx={nx}, L={L}, ntypes={ntypes}, "
              f"n_frames={n_frames}, reading frame {frame}")

        # Seek to the requested frame
        f.seek(header_size + frame * frame_size)
        raw = np.frombuffer(f.read(frame_size), dtype=np.float32)

    # Reshape: raw is [type0_grid, type1_grid, ...] each of length M
    # MATILDA.FT uses row-major (C) ordering with the grid index computed as:
    #   3D: i = nz * Nx[1]*Nx[0] + ny * Nx[0] + nx_idx
    #   i.e., fastest index is x, then y, then z
    dx = L / nx.astype(float)

    densities = {}
    for t in range(ntypes):
        rho_flat = raw[t * M : (t + 1) * M]
        if dim == 3:
            # Reshape to (Nz, Ny, Nx) so that axis 0=z, 1=y, 2=x
            rho = rho_flat.reshape((nx[2], nx[1], nx[0]))
        elif dim == 2:
            rho = rho_flat.reshape((nx[1], nx[0]))
        else:
            raise ValueError(f"Unsupported dimensionality: {dim}")
        densities[t] = rho.astype(np.float64)

    return DensityFrame(
        dim=dim, nx=nx, L=L, ntypes=ntypes,
        densities=densities, dx=dx
    )


# =============================================================================
# 2. DENSITY THRESHOLDING
# =============================================================================

def compute_threshold(rho: np.ndarray, method: str = 'auto') -> float:
    """
    Determine the density threshold for binarizing the field.

    Parameters
    ----------
    rho : np.ndarray
        Density field (2D or 3D).
    method : str
        'auto'  — use the midpoint between the two peaks of the density histogram
                   (works well for clearly phase-separated systems)
        'half'  — use rho_0 / 2 where rho_0 is the mean density
        'otsu'  — Otsu's method (maximizes inter-class variance)
        float   — use that value directly

    Returns
    -------
    float : threshold value
    """
    if method == 'half':
        return float(np.mean(rho)) / 2.0

    elif method == 'otsu':
        # Otsu's method on the density histogram
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(rho.ravel()))

    elif method == 'auto':
        # Find two peaks in histogram, threshold at the valley between them
        hist, bin_edges = np.histogram(rho.ravel(), bins=200)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Smooth histogram to find peaks robustly
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=5)

        # Find peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist_smooth, height=0.05 * hist_smooth.max())

        if len(peaks) >= 2:
            # Take the two tallest peaks
            sorted_peaks = peaks[np.argsort(hist_smooth[peaks])][-2:]
            sorted_peaks.sort()
            # Threshold is the minimum between them
            valley_region = hist_smooth[sorted_peaks[0]:sorted_peaks[1]]
            threshold = bin_centers[sorted_peaks[0] + np.argmin(valley_region)]
        else:
            # Fallback to mean
            threshold = float(np.mean(rho))
            print(f"  Warning: could not find two histogram peaks, using mean={threshold:.4f}")

        return threshold

    else:
        return float(method)


def binarize_field(rho: np.ndarray, threshold: float) -> np.ndarray:
    """Create a binary mask: 1 where rho >= threshold (LC-rich), 0 otherwise."""
    return (rho >= threshold).astype(np.float64)


# =============================================================================
# 3. MARCHING CUBES — EXTRACT ISOSURFACE
# =============================================================================

def extract_isosurface(rho: np.ndarray, threshold: float, dx: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the isosurface of the density field at the given threshold
    using the marching cubes algorithm.

    Parameters
    ----------
    rho : np.ndarray, shape (Nz, Ny, Nx)
        3D density field.
    threshold : float
        Isosurface level.
    dx : np.ndarray, shape (3,)
        Grid spacing [dx, dy, dz].

    Returns
    -------
    verts : np.ndarray, shape (V, 3)
        Vertex positions of the triangulated surface.
    faces : np.ndarray, shape (F, 3)
        Triangle connectivity (indices into verts).
    """
    from skimage.measure import marching_cubes

    # marching_cubes expects spacing as (z_spacing, y_spacing, x_spacing)
    # since our array is (Nz, Ny, Nx)
    spacing = (dx[2] if len(dx) > 2 else dx[0],
               dx[1] if len(dx) > 1 else dx[0],
               dx[0])

    verts, faces, normals, values = marching_cubes(
        rho, level=threshold, spacing=spacing
    )

    return verts, faces


# =============================================================================
# 4. CURVATURE CALCULATIONS ON TRIANGULATED SURFACE
# =============================================================================

def compute_vertex_areas_and_curvatures(verts: np.ndarray, faces: np.ndarray
                                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-vertex mean curvature (H) and Gaussian curvature (K)
    using the discrete Laplace-Beltrami operator (cotangent weights).

    Also returns the mixed Voronoi area associated with each vertex,
    needed for proper integration of curvatures.

    Parameters
    ----------
    verts : (V, 3) array of vertex positions
    faces : (F, 3) array of triangle indices

    Returns
    -------
    H : (V,) mean curvature at each vertex
    K : (V,) Gaussian curvature at each vertex
    A_mixed : (V,) mixed area at each vertex
    """
    V = len(verts)
    F = len(faces)

    # Initialize accumulators
    H_vec = np.zeros((V, 3), dtype=np.float64)   # Laplace-Beltrami vector
    A_mixed = np.zeros(V, dtype=np.float64)       # mixed Voronoi area
    angle_sum = np.zeros(V, dtype=np.float64)     # sum of angles at each vertex

    for f_idx in range(F):
        i, j, k = faces[f_idx]
        pi, pj, pk = verts[i], verts[j], verts[k]

        # Edge vectors
        eij = pj - pi
        ejk = pk - pj
        eki = pi - pk

        # Compute angles at each vertex of this triangle
        cos_i = np.dot(eij, -eki) / (np.linalg.norm(eij) * np.linalg.norm(eki) + 1e-30)
        cos_j = np.dot(-eij, ejk) / (np.linalg.norm(eij) * np.linalg.norm(ejk) + 1e-30)
        cos_k = np.dot(-ejk, eki) / (np.linalg.norm(ejk) * np.linalg.norm(eki) + 1e-30)

        cos_i = np.clip(cos_i, -1, 1)
        cos_j = np.clip(cos_j, -1, 1)
        cos_k = np.clip(cos_k, -1, 1)

        angle_i = np.arccos(cos_i)
        angle_j = np.arccos(cos_j)
        angle_k = np.arccos(cos_k)

        angle_sum[i] += angle_i
        angle_sum[j] += angle_j
        angle_sum[k] += angle_k

        # Cotangent weights for Laplace-Beltrami
        cot_i = cos_i / (np.sin(angle_i) + 1e-30)
        cot_j = cos_j / (np.sin(angle_j) + 1e-30)
        cot_k = cos_k / (np.sin(angle_k) + 1e-30)

        # Triangle area
        tri_area = 0.5 * np.linalg.norm(np.cross(eij, -eki))

        # Mixed area contributions (Meyer et al. 2003)
        # Check for obtuse triangles
        is_obtuse_i = angle_i > np.pi / 2
        is_obtuse_j = angle_j > np.pi / 2
        is_obtuse_k = angle_k > np.pi / 2
        is_obtuse = is_obtuse_i or is_obtuse_j or is_obtuse_k

        if not is_obtuse:
            # Voronoi area
            A_mixed[i] += (1.0 / 8.0) * (cot_k * np.dot(eij, eij) + cot_j * np.dot(eki, eki))
            A_mixed[j] += (1.0 / 8.0) * (cot_i * np.dot(ejk, ejk) + cot_k * np.dot(eij, eij))
            A_mixed[k] += (1.0 / 8.0) * (cot_j * np.dot(eki, eki) + cot_i * np.dot(ejk, ejk))
        else:
            # Use triangle area / 2 for obtuse vertex, area / 4 for others
            for v_idx, is_obt in [(i, is_obtuse_i), (j, is_obtuse_j), (k, is_obtuse_k)]:
                if is_obt:
                    A_mixed[v_idx] += tri_area / 2.0
                else:
                    A_mixed[v_idx] += tri_area / 4.0

        # Laplace-Beltrami operator contributions
        H_vec[i] += cot_k * eij + cot_j * (-eki)
        H_vec[j] += cot_i * ejk + cot_k * (-eij)
        H_vec[k] += cot_j * eki + cot_i * (-ejk)

    # Mean curvature: H = |Laplacian| / (2 * A_mixed)
    # The sign is determined by the direction of the normal
    A_safe = np.maximum(A_mixed, 1e-30)
    H_magnitude = np.linalg.norm(H_vec, axis=1) / (2.0 * A_safe)

    # For sign: use the dot product of H_vec with vertex normals
    # (positive = curving toward normal, i.e., convex)
    normals = compute_vertex_normals(verts, faces)
    H_sign = np.sign(np.sum(H_vec * normals, axis=1))
    H = H_sign * H_magnitude

    # Gaussian curvature via angle defect: K = (2*pi - sum_angles) / A_mixed
    K = (2.0 * np.pi - angle_sum) / A_safe

    return H, K, A_mixed


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area-weighted vertex normals."""
    V = len(verts)
    normals = np.zeros((V, 3), dtype=np.float64)

    for f_idx in range(len(faces)):
        i, j, k = faces[f_idx]
        e1 = verts[j] - verts[i]
        e2 = verts[k] - verts[i]
        fn = np.cross(e1, e2)
        normals[i] += fn
        normals[j] += fn
        normals[k] += fn

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-30)
    return normals


def principal_curvatures(H: np.ndarray, K: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute principal curvatures from mean (H) and Gaussian (K) curvatures.
      kappa_1 = H + sqrt(H^2 - K)
      kappa_2 = H - sqrt(H^2 - K)

    Returns kappa_1 >= kappa_2.
    """
    discriminant = np.maximum(H**2 - K, 0.0)
    sqrt_disc = np.sqrt(discriminant)
    kappa_1 = H + sqrt_disc
    kappa_2 = H - sqrt_disc
    return kappa_1, kappa_2


# =============================================================================
# 5. MINKOWSKI FUNCTIONALS (on voxelized binary field)
# =============================================================================

def minkowski_functionals_3d(binary: np.ndarray, dx: np.ndarray) -> Dict[str, float]:
    """
    Compute the four 3D Minkowski functionals on a binary voxel field
    using the Crofton/lattice approach.

    Parameters
    ----------
    binary : np.ndarray, shape (Nz, Ny, Nx), values 0 or 1
    dx : np.ndarray, shape (3,), grid spacing [dx, dy, dz]

    Returns
    -------
    dict with keys:
      'volume_fraction' : V / V_total
      'surface_area'    : total interface area (in simulation units^2)
      'mean_curvature'  : integrated mean curvature
      'euler_char'      : Euler characteristic
    """
    voxel_vol = float(np.prod(dx))
    total_vol = float(np.prod(binary.shape)) * voxel_vol

    # V0: Volume
    V0 = np.sum(binary) * voxel_vol
    vol_frac = V0 / total_vol

    # For surface area and curvatures, use marching cubes
    # This is more accurate than counting face-shared neighbor differences
    from skimage.measure import marching_cubes, mesh_surface_area

    try:
        spacing = (dx[2] if len(dx) > 2 else dx[0],
                   dx[1] if len(dx) > 1 else dx[0],
                   dx[0])
        verts, faces, _, _ = marching_cubes(binary.astype(float), level=0.5, spacing=spacing)

        # V1: Surface area
        surface_area = mesh_surface_area(verts, faces)

        # V2 and V3: Mean curvature and Euler characteristic from mesh
        H, K, A = compute_vertex_areas_and_curvatures(verts, faces)

        integrated_H = np.sum(H * A)
        euler_char = np.sum(K * A) / (2.0 * np.pi)  # Gauss-Bonnet: integral(K dA) = 2*pi*chi

    except (ValueError, RuntimeError) as e:
        print(f"  Warning: marching_cubes failed ({e}), returning zeros for surface metrics.")
        surface_area = 0.0
        integrated_H = 0.0
        euler_char = 0.0

    return {
        'volume_fraction': vol_frac,
        'surface_area': surface_area,
        'mean_curvature': integrated_H,
        'euler_char': euler_char,
    }


# =============================================================================
# 6. MORPHOLOGY CLASSIFICATION
# =============================================================================

def classify_morphology(metrics: Dict[str, float],
                        H: Optional[np.ndarray] = None,
                        K: Optional[np.ndarray] = None
                        ) -> Dict[str, object]:
    """
    Classify the morphology based on Minkowski functionals and curvature stats.

    Returns a dict with the classification and supporting evidence.
    """
    chi = metrics['euler_char']
    vol_frac = metrics['volume_fraction']
    sa = metrics['surface_area']
    int_H = metrics['mean_curvature']

    result = {
        'metrics': metrics,
        'classification': 'unknown',
        'confidence': 'low',
        'evidence': []
    }

    # Euler characteristic is the strongest discriminator
    # Normalize by some characteristic scale to make threshold-independent
    if chi > 2:
        result['classification'] = 'spheres'
        result['evidence'].append(f'Large positive Euler char ({chi:.1f}) indicates disconnected objects')
    elif chi < -2 and abs(int_H) / (sa + 1e-30) < 0.1:
        result['classification'] = 'bicontinuous'
        result['evidence'].append(f'Large negative Euler char ({chi:.1f}) with small mean curvature')
    elif abs(chi) < 2 and abs(int_H) / (sa + 1e-30) < 0.05:
        result['classification'] = 'lamellae'
        result['evidence'].append(f'Near-zero Euler char ({chi:.1f}) and near-zero mean curvature')
    elif chi < 0:
        result['classification'] = 'cylinders'
        result['evidence'].append(f'Moderately negative Euler char ({chi:.1f})')

    # Refine with curvature distributions if available
    if H is not None and K is not None:
        H_var = np.var(H)
        K_mean = np.mean(K)
        frac_saddle = np.mean(K < 0)

        result['H_variance'] = H_var
        result['K_mean'] = K_mean
        result['fraction_saddle_points'] = frac_saddle

        if frac_saddle > 0.4:
            if result['classification'] != 'bicontinuous':
                result['evidence'].append(
                    f'High saddle-point fraction ({frac_saddle:.2f}) suggests bicontinuous')
                if result['classification'] == 'unknown':
                    result['classification'] = 'bicontinuous'

    return result


# =============================================================================
# 7. SCALAR METRICS FOR GP MODELING
# =============================================================================

def gp_scalar_metrics(metrics: Dict[str, float],
                      H: Optional[np.ndarray] = None,
                      K: Optional[np.ndarray] = None,
                      A: Optional[np.ndarray] = None
                      ) -> Dict[str, float]:
    """
    Compute scalar quantities suitable as outputs for a Gaussian process model.
    These vary continuously across phase boundaries and together distinguish
    all four morphology types.

    Returns
    -------
    dict with:
      'euler_density'     : chi / V_total — primary topology metric
      'mean_curv_ratio'   : |integrated_H| / surface_area — separates flat vs curved
      'gaussian_curv_var' : variance of K — high for bicontinuous
      'anisotropy'        : ratio of principal curvature variances
    """
    sa = metrics['surface_area']
    chi = metrics['euler_char']
    int_H = metrics['mean_curvature']

    result = {
        'euler_density': chi,  # already extensive — divide by vol if desired
        'mean_curv_ratio': abs(int_H) / (sa + 1e-30),
    }

    if K is not None and A is not None:
        # Area-weighted variance of Gaussian curvature
        total_A = np.sum(A) + 1e-30
        K_mean = np.sum(K * A) / total_A
        K_var = np.sum((K - K_mean)**2 * A) / total_A
        result['gaussian_curv_var'] = K_var

    if H is not None and K is not None:
        kappa_1, kappa_2 = principal_curvatures(H, K)
        if A is not None:
            total_A = np.sum(A) + 1e-30
            var_k1 = np.sum((kappa_1 - np.mean(kappa_1))**2 * A) / total_A
            var_k2 = np.sum((kappa_2 - np.mean(kappa_2))**2 * A) / total_A
            result['anisotropy'] = var_k1 / (var_k2 + 1e-30)
        else:
            result['anisotropy'] = np.var(kappa_1) / (np.var(kappa_2) + 1e-30)

    return result


# =============================================================================
# 8. MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_morphology(filename: str,
                       frame: int = -1,
                       type_idx: int = 0,
                       threshold_method: str = 'auto',
                       verbose: bool = True) -> Dict:
    """
    Full analysis pipeline: read density → threshold → extract surface →
    compute curvatures → classify morphology.

    Parameters
    ----------
    filename : str
        Path to grid_densities.bin
    frame : int
        Frame index (-1 for last)
    type_idx : int
        Which particle type's density to analyze
    threshold_method : str
        'auto', 'otsu', 'half', or a float value
    verbose : bool
        Print progress

    Returns
    -------
    dict with all computed metrics and classification
    """
    if verbose:
        print(f"Reading {filename}, frame={frame}, type={type_idx}")

    data = read_density_binary(filename, frame=frame)

    if data.dim != 3:
        raise NotImplementedError("Curvature analysis requires 3D density fields. "
                                  "For 2D, use contour-based curvature methods.")

    rho = data.densities[type_idx]

    if verbose:
        print(f"  Density field shape: {rho.shape}")
        print(f"  Density range: [{rho.min():.4f}, {rho.max():.4f}], mean={rho.mean():.4f}")

    # Threshold
    thresh = compute_threshold(rho, method=threshold_method)
    if verbose:
        print(f"  Threshold: {thresh:.4f} (method='{threshold_method}')")

    binary = binarize_field(rho, thresh)
    vol_frac = np.mean(binary)
    if verbose:
        print(f"  Volume fraction above threshold: {vol_frac:.4f}")

    # Minkowski functionals
    if verbose:
        print("  Computing Minkowski functionals...")
    mink = minkowski_functionals_3d(binary, data.dx)
    if verbose:
        for k, v in mink.items():
            print(f"    {k}: {v:.4f}")

    # Detailed curvature from isosurface
    if verbose:
        print("  Extracting isosurface and computing curvatures...")
    try:
        verts, faces = extract_isosurface(rho, thresh, data.dx)
        if verbose:
            print(f"    Mesh: {len(verts)} vertices, {len(faces)} triangles")

        H, K, A = compute_vertex_areas_and_curvatures(verts, faces)
        kappa_1, kappa_2 = principal_curvatures(H, K)

        curvature_stats = {
            'H_mean': np.average(H, weights=A),
            'H_std': np.sqrt(np.average((H - np.average(H, weights=A))**2, weights=A)),
            'K_mean': np.average(K, weights=A),
            'K_std': np.sqrt(np.average((K - np.average(K, weights=A))**2, weights=A)),
            'frac_saddle': np.average(K < 0, weights=A),
            'frac_convex': np.average((kappa_1 > 0) & (kappa_2 > 0), weights=A),
        }
    except Exception as e:
        print(f"  Warning: curvature computation failed: {e}")
        H, K, A = None, None, None
        kappa_1, kappa_2 = None, None
        curvature_stats = {}

    # Classification
    classification = classify_morphology(mink, H, K)

    # GP-ready scalars
    gp_metrics = gp_scalar_metrics(mink, H, K, A)

    if verbose:
        print(f"\n  Classification: {classification['classification']}")
        for ev in classification['evidence']:
            print(f"    → {ev}")
        print(f"\n  GP-ready scalar metrics:")
        for k, v in gp_metrics.items():
            print(f"    {k}: {v:.6f}")

    return {
        'density_frame': data,
        'threshold': thresh,
        'minkowski': mink,
        'curvature_stats': curvature_stats,
        'classification': classification,
        'gp_metrics': gp_metrics,
        'H': H, 'K': K, 'A': A,
        'kappa_1': kappa_1, 'kappa_2': kappa_2,
    }


# =============================================================================
# 9. VISUALIZATION HELPERS
# =============================================================================

def plot_interfacial_shape_distribution(kappa_1: np.ndarray, kappa_2: np.ndarray,
                                        A: np.ndarray, save_path: str = None):
    """
    Plot the interfacial shape distribution (ISD): a 2D histogram of
    (kappa_1, kappa_2) weighted by vertex area.

    Reference morphology signatures:
      Spheres:      cluster at kappa_1 ≈ kappa_2 > 0
      Cylinders:    cluster at kappa_1 > 0, kappa_2 ≈ 0
      Lamellae:     cluster at kappa_1 ≈ kappa_2 ≈ 0
      Bicontinuous: broad, centered near kappa_1 ≈ -kappa_2
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Determine range
    k_max = np.percentile(np.abs(np.concatenate([kappa_1, kappa_2])), 98)

    hist, xedges, yedges = np.histogram2d(
        kappa_1, kappa_2, bins=100,
        range=[[-k_max, k_max], [-k_max, k_max]],
        weights=A, density=True
    )

    ax.imshow(hist.T, origin='lower', aspect='equal',
              extent=[-k_max, k_max, -k_max, k_max],
              cmap='viridis', interpolation='gaussian')

    # Reference lines
    ax.axhline(0, color='white', ls='--', alpha=0.5, lw=0.8)
    ax.axvline(0, color='white', ls='--', alpha=0.5, lw=0.8)
    ax.plot([-k_max, k_max], [-k_max, k_max], 'w--', alpha=0.3, lw=0.8,
            label='κ₁ = κ₂ (spheres)')
    ax.plot([-k_max, k_max], [k_max, -k_max], 'r--', alpha=0.3, lw=0.8,
            label='κ₁ = -κ₂ (saddle)')

    ax.set_xlabel('κ₁')
    ax.set_ylabel('κ₂')
    ax.set_title('Interfacial Shape Distribution')
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved ISD plot to {save_path}")
    else:
        plt.show()


def plot_curvature_histograms(H: np.ndarray, K: np.ndarray, A: np.ndarray,
                              save_path: str = None):
    """Plot distributions of mean and Gaussian curvature."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Mean curvature
    H_range = np.percentile(np.abs(H), 98)
    axes[0].hist(H, bins=100, weights=A, density=True, range=(-H_range, H_range),
                 color='steelblue', alpha=0.7)
    axes[0].axvline(0, color='red', ls='--', alpha=0.5)
    axes[0].set_xlabel('Mean Curvature H')
    axes[0].set_ylabel('Area-weighted density')
    axes[0].set_title('Mean Curvature Distribution')

    # Gaussian curvature
    K_range = np.percentile(np.abs(K), 98)
    axes[1].hist(K, bins=100, weights=A, density=True, range=(-K_range, K_range),
                 color='darkorange', alpha=0.7)
    axes[1].axvline(0, color='red', ls='--', alpha=0.5)
    axes[1].set_xlabel('Gaussian Curvature K')
    axes[1].set_ylabel('Area-weighted density')
    axes[1].set_title('Gaussian Curvature Distribution')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved curvature histograms to {save_path}")
    else:
        plt.show()


# =============================================================================
# 10. COMMAND-LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Morphology analysis of MATILDA.FT density fields')
    parser.add_argument('filename', help='Path to grid_densities.bin')
    parser.add_argument('--frame', type=int, default=-1,
                        help='Frame index to analyze (-1 = last)')
    parser.add_argument('--type', type=int, default=0, dest='type_idx',
                        help='Particle type index to analyze')
    parser.add_argument('--threshold', type=str, default='auto',
                        help='Thresholding method: auto, otsu, half, or a float')
    parser.add_argument('--plot', action='store_true',
                        help='Generate curvature distribution plots')
    parser.add_argument('--plot-prefix', type=str, default='morphology',
                        help='Prefix for plot filenames')

    args = parser.parse_args()

    # Run analysis
    results = analyze_morphology(
        args.filename, frame=args.frame,
        type_idx=args.type_idx,
        threshold_method=args.threshold
    )

    # Optional plotting
    if args.plot and results['H'] is not None:
        plot_curvature_histograms(
            results['H'], results['K'], results['A'],
            save_path=f"{args.plot_prefix}_curvatures.png"
        )
        plot_interfacial_shape_distribution(
            results['kappa_1'], results['kappa_2'], results['A'],
            save_path=f"{args.plot_prefix}_isd.png"
        )


if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""
Plotting utility for Gaia RVS spectral line identification
===========================================================

Use this to add colored rectangles marking known spectral features
to your outlier residual plots.

Based on GSP-Spec line list from Contursi et al. (2021), A&A, 654, A130

Usage:
    lines = load_linelists()
    fig, ax = plt.subplots()
    ax.plot(wl, flux)
    add_line_markers(ax, lines, show_strong=True, show_abundance=False, ...)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# ============================================================================
# Color scheme for different species
# ============================================================================

SPECIES_COLORS = {
    # Molecules
    'CN': '#FF69B4',      # Hot pink (Cyanogen)
    'DIB': '#808080',     # Gray (interstellar)
    
    # Hydrogen
    'H I': '#E6E6FA',     # Lavender (Paschen)
    
    # Alpha elements
    'Mg I': '#90EE90',    # Light green
    'Si I': '#98FB98',    # Pale green  
    'S I': '#FFFF00',     # Yellow
    'Ca I': '#FFA500',    # Orange
    'Ca II': '#FF4500',   # Orange-red (Ca triplet)
    'Ti I': '#DDA0DD',    # Plum
    
    # Iron-peak
    'Cr I': '#87CEEB',    # Sky blue
    'Fe I': '#6495ED',    # Cornflower blue
    'Ni I': '#4169E1',    # Royal blue
    
    # s-process (heavy elements)
    'Zr I': '#00CED1',    # Dark turquoise
    'Ce II': '#20B2AA',   # Light sea green
    'Nd II': '#3CB371',   # Medium sea green
    
    # Nitrogen
    'N I': '#FFB6C1',     # Light pink
}


# ============================================================================
# Data container
# ============================================================================

@dataclass
class RVSLineLists:
    """Container for all RVS line list components."""
    strong: pd.DataFrame      # Strong diagnostic features (Ca II, Paschen, etc.)
    abundance: pd.DataFrame   # GSP-Spec abundance lines with atomic data
    cn_bands: pd.DataFrame    # CN molecular band regions
    dib: pd.DataFrame         # Diffuse interstellar band(s)
    
    def summary(self):
        """Print summary of loaded data."""
        print("RVS Line Lists Summary")
        print("=" * 40)
        print(f"  Strong features:    {len(self.strong):3d} lines")
        print(f"  Abundance lines:    {len(self.abundance):3d} lines")
        print(f"  CN band regions:    {len(self.cn_bands):3d} regions")
        print(f"  DIB features:       {len(self.dib):3d} lines")
        print("=" * 40)


def load_linelists(data_dir: Optional[str] = None) -> RVSLineLists:
    """
    Load all RVS line list components.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing the CSV files. If None, searches current 
        directory and common locations.
    
    Returns
    -------
    RVSLineLists
        Container with .strong, .abundance, .cn_bands, .dib DataFrames
    """
    # Find data directory
    if data_dir is not None:
        base = Path(data_dir)
    else:
        # Try current directory first, then common locations
        candidates = [
            Path('.'),
            Path(__file__).parent if '__file__' in dir() else Path('.'),
            Path.home() / 'rvs_linelists',
        ]
        base = None
        for c in candidates:
            if (c / 'rvs_strong_features.csv').exists():
                base = c
                break
        if base is None:
            raise FileNotFoundError(
                "Could not find line list CSVs. Provide data_dir parameter."
            )
    
    # Load each component
    strong = pd.read_csv(base / 'rvs_strong_features.csv')
    abundance = pd.read_csv(base / 'rvs_gspspec_lines.csv')
    cn_bands = pd.read_csv(base / 'rvs_cn_band_regions.csv')
    dib = pd.read_csv(base / 'rvs_dib_features.csv')
    
    return RVSLineLists(strong=strong, abundance=abundance, 
                        cn_bands=cn_bands, dib=dib)


# ============================================================================
# Plotting functions
# ============================================================================

def add_line_markers(
    ax,
    lines: RVSLineLists,
    # Component toggles
    show_strong: bool = True,
    show_abundance: bool = True,
    show_cn: bool = True,
    show_dib: bool = True,
    # Filtering
    species_filter: Optional[List[str]] = None,
    # Appearance
    line_width_nm: float = 0.3,
    alpha: float = 0.3,
    cn_alpha: Optional[float] = None,  # Defaults to alpha * 0.5
    # Labels
    show_labels: bool = True,
    label_fontsize: int = 6,
    label_ypos: float = 0.95,
):
    """
    Add colored rectangles marking spectral line positions.
    
    Parameters
    ----------
    ax : matplotlib axis
        The axis to add markers to
    lines : RVSLineLists
        Line list container from load_linelists()
    show_strong : bool
        Show strong diagnostic features (Ca II triplet, Paschen, etc.)
    show_abundance : bool
        Show detailed GSP-Spec abundance lines
    show_cn : bool
        Show CN molecular band regions
    show_dib : bool
        Show diffuse interstellar band
    species_filter : list of str, optional
        Only show these species (e.g., ['Ca II', 'Fe I']). 
        Applied to strong and abundance lines, not CN/DIB.
    line_width_nm : float
        Half-width of each line marker in nm (default 0.3 nm ~ RVS resolution)
    alpha : float
        Transparency of rectangles
    cn_alpha : float, optional
        Transparency for CN bands (default: alpha * 0.5)
    show_labels : bool
        Whether to show species labels above lines
    label_fontsize : int
        Font size for labels
    label_ypos : float
        Y position for labels (in axis fraction, 0=bottom, 1=top)
    """
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    
    if cn_alpha is None:
        cn_alpha = alpha * 0.5
    
    def _add_rect(wl, color, width=line_width_nm, a=alpha):
        rect = Rectangle(
            (wl - width, ymin),
            2 * width,
            height,
            facecolor=color,
            edgecolor='none',
            alpha=a
        )
        ax.add_patch(rect)
    
    def _add_label(wl, species):
        if show_labels:
            y = ymin + height * label_ypos
            ax.text(wl, y, species, ha='center', va='top', 
                    fontsize=label_fontsize, rotation=90)
    
    def _process_lines(df, category_alpha=alpha):
        """Process a DataFrame of lines."""
        for _, row in df.iterrows():
            species = row['species']
            if species_filter is not None and species not in species_filter:
                continue
            wl = row['lambda_vac_nm']
            color = SPECIES_COLORS.get(species, '#CCCCCC')
            _add_rect(wl, color, a=category_alpha)
            _add_label(wl, species)
    
    # Add each component
    if show_strong:
        _process_lines(lines.strong)
    
    if show_abundance:
        _process_lines(lines.abundance)
    
    if show_dib:
        # DIB doesn't get filtered by species_filter
        for _, row in lines.dib.iterrows():
            wl = row['lambda_vac_nm']
            color = SPECIES_COLORS.get('DIB', '#808080')
            _add_rect(wl, color)
            _add_label(wl, 'DIB')
    
    if show_cn:
        # CN bands are regions, not single lines
        for _, row in lines.cn_bands.iterrows():
            wl_start = row['lambda_vac_nm_start']
            wl_end = row['lambda_vac_nm_end']
            color = SPECIES_COLORS.get('CN', '#FF69B4')
            
            rect = Rectangle(
                (wl_start, ymin),
                wl_end - wl_start,
                height,
                facecolor=color,
                edgecolor='none',
                alpha=cn_alpha
            )
            ax.add_patch(rect)
            
            if show_labels:
                wl_mid = (wl_start + wl_end) / 2
                _add_label(wl_mid, 'CN')


def add_legend(ax, species_list: Optional[List[str]] = None, **kwargs):
    """
    Add a legend showing the species colors.
    
    Parameters
    ----------
    ax : matplotlib axis
    species_list : list of str, optional
        Species to include. If None, shows all.
    **kwargs : 
        Passed to ax.legend()
    """
    if species_list is None:
        species_list = list(SPECIES_COLORS.keys())
    
    handles = []
    labels = []
    for species in species_list:
        if species in SPECIES_COLORS:
            handle = plt.Rectangle((0, 0), 1, 1, 
                                    facecolor=SPECIES_COLORS[species], 
                                    edgecolor='none',
                                    alpha=0.5)
            handles.append(handle)
            labels.append(species)
    
    legend_kwargs = dict(loc='upper left', ncol=3, fontsize=7, framealpha=0.9)
    legend_kwargs.update(kwargs)
    
    if handles:
        ax.legend(handles, labels, **legend_kwargs)


# ============================================================================
# Convenience functions for common use cases
# ============================================================================

def plot_hot_star_features(ax, lines: RVSLineLists, **kwargs):
    """
    Plot features relevant for hot stars (O/B/A).
    Shows: Paschen series, Ca II triplet. Hides: CN, most metals.
    """
    defaults = dict(
        show_strong=True,
        show_abundance=False,
        show_cn=False,
        show_dib=True,
        species_filter=['H I', 'Ca II'],
        alpha=0.4,
    )
    defaults.update(kwargs)
    add_line_markers(ax, lines, **defaults)


def plot_cool_star_features(ax, lines: RVSLineLists, **kwargs):
    """
    Plot features relevant for cool stars (G/K/M).
    Shows: All metals, CN bands, Ca II. De-emphasizes Paschen.
    """
    defaults = dict(
        show_strong=True,
        show_abundance=True,
        show_cn=True,
        show_dib=True,
        alpha=0.25,
    )
    defaults.update(kwargs)
    add_line_markers(ax, lines, **defaults)


def plot_strong_only(ax, lines: RVSLineLists, **kwargs):
    """
    Plot only the major diagnostic features.
    Good for overview plots without clutter.
    """
    defaults = dict(
        show_strong=True,
        show_abundance=False,
        show_cn=False,
        show_dib=True,
        alpha=0.4,
    )
    defaults.update(kwargs)
    add_line_markers(ax, lines, **defaults)


# ============================================================================
# Example / test
# ============================================================================

if __name__ == "__main__":
    # Load line lists
    lines = load_linelists()
    lines.summary()
    
    # Create example plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Wavelength range and fake data
    wl = np.linspace(845, 872, 1000)
    np.random.seed(42)
    flux = 1.0 + 0.02 * np.random.randn(len(wl))
    flux[(wl > 849.5) & (wl < 850.5)] -= 0.1
    flux[(wl > 862.0) & (wl < 863.0)] += 0.05
    
    # ----- Panel 1: Hot star view -----
    ax1 = axes[0]
    ax1.plot(wl, flux, 'k-', lw=0.5)
    ax1.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax1.set_ylim(0.85, 1.15)
    
    plot_hot_star_features(ax1, lines)
    add_legend(ax1, ['H I', 'Ca II', 'DIB'])
    ax1.set_ylabel('Flux')
    ax1.set_title('Hot star view: Paschen + Ca II only')
    
    # ----- Panel 2: Cool star view -----
    ax2 = axes[1]
    ax2.plot(wl, flux, 'k-', lw=0.5)
    ax2.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax2.set_ylim(0.85, 1.15)
    
    plot_cool_star_features(ax2, lines)
    add_legend(ax2)
    ax2.set_ylabel('Flux')
    ax2.set_title('Cool star view: All features + CN bands')
    
    # ----- Panel 3: Custom selection -----
    ax3 = axes[2]
    ax3.plot(wl, flux, 'k-', lw=0.5)
    ax3.axhline(1.0, color='gray', ls='--', lw=0.5)
    ax3.set_ylim(0.85, 1.15)
    
    # Custom: only Ca II triplet and Fe I
    add_line_markers(ax3, lines, 
                     show_strong=True, 
                     show_abundance=True,
                     show_cn=False, 
                     show_dib=False,
                     species_filter=['Ca II', 'Fe I'],
                     alpha=0.4,
                     show_labels=True)
    add_legend(ax3, ['Ca II', 'Fe I'])
    ax3.set_xlabel('Vacuum Wavelength (nm)')
    ax3.set_ylabel('Flux')
    ax3.set_title('Custom: Ca II + Fe I only')
    
    plt.tight_layout()
    plt.savefig('rvs_line_identification_example.png', dpi=150, bbox_inches='tight')
    print("\nSaved: rvs_line_identification_example.png")

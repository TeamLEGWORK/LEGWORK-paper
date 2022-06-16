import legwork as lw
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import paths

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

# spread out some frequencies and eccentricities
f_orb_s = np.logspace(-4, -1, 200)
ecc_s = np.linspace(0, 0.9, 150)

# turn them into a grid
F, E = np.meshgrid(f_orb_s, ecc_s)

# flatten the grid
F_flat, E_flat = F.flatten(), E.flatten()

# put all of the sources at the same distance with the same mass
m_1 = np.repeat(10, len(F_flat)) * u.Msun
m_2 = np.repeat(10, len(F_flat)) * u.Msun
dist = np.repeat(8, len(F_flat)) * u.kpc

# define a set of sources
sources = lw.source.Source(m_1=m_1, m_2=m_2, f_orb=F_flat * u.Hz, ecc=E_flat, dist=dist, gw_lum_tol=1e-3)
sources.get_merger_time()

# compute the LISA SNR
LISA_snr = sources.get_snr(verbose=True, which_sources=sources.t_merge > 0.1 * u.yr)

# compute the TianQin SNR
sources.update_sc_params({"instrument": "TianQin", "L": np.sqrt(3) * 1e5 * u.km})
TQ_snr = sources.get_snr(verbose=True, which_sources=sources.t_merge > 0.1 * u.yr)

# create a figure
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xscale("log")
ax.set_xlabel(r"Orbital Frequency, $f_{\rm orb} \, [{\rm Hz}]$")
ax.set_ylabel(r"Eccentricity, $e$")

ratio = np.zeros_like(LISA_snr)
nonzero = np.logical_and(LISA_snr > 0, TQ_snr > 0)
ratio[nonzero] = LISA_snr[nonzero] / TQ_snr[nonzero]
ratio = ratio.reshape(F.shape)

# make contours of the ratio of SNR
ratio_cont = ax.contourf(F, E, ratio, cmap="PRGn_r", norm=TwoSlopeNorm(vcenter=1.0),
                         levels=np.arange(0, 3.75 + 0.2, 0.2))

for c in ratio_cont.collections:
    c.set_edgecolor("face")

# add a line when the SNRs are equal
ax.contour(F, E, ratio, levels=[1.0], colors="grey", linewidths=2.0, linestyles="--")

# add a colourbar
cbar = fig.colorbar(ratio_cont, fraction=2/14, pad=0.02,
                    label=r"$\rho_{\rm LISA} / \rho_{\rm TianQin}$",
                    ticks=np.arange(0, 3.5 + 0.5, 0.5))

# annotate which regions suit each detector
ax.annotate("LISA stronger", xy=(0.1, 0.53), xycoords="axes fraction", fontsize=0.7 * fs,
            color=plt.get_cmap("PRGn_r")(1.0),
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="white", alpha=0.5, pad=0.4))
ax.annotate("TianQin stronger", xy=(0.6, 0.73), xycoords="axes fraction", fontsize=0.7 * fs,
            color=plt.get_cmap("PRGn_r")(0.0),
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="white", alpha=0.5, pad=0.4))

# annotate with source details
source_string = r"$m_1 = {{{}}} \, {{ \rm M_{{\odot}}}}$".format(m_1[0].value)
source_string += "\n"
source_string += r"$m_2 = {{{}}} \, {{ \rm M_{{\odot}}}}$".format(m_1[0].value)
source_string += "\n"
source_string += r"$D_L = {{{}}} \, {{ \rm kpc}}$".format(dist[0].value)
ax.annotate(source_string, xy=(0.98, 0.03), xycoords="axes fraction", ha="right", fontsize=0.75*fs,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="white", alpha=0.5, pad=0.4))

ax.set_rasterization_zorder(10000)

plt.savefig(paths.figures / "detector_snr_ratio.pdf", format="pdf", bbox_inches="tight")
import legwork as lw
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.6 * fs,
          'ytick.labelsize': 0.6 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

vbs = lw.source.VerificationBinaries()
vbs.snr = np.array(vbs.true_snr)

fig, ax = lw.visualisation.plot_sensitivity_curve(frequency_range=np.logspace(-4, 0, 1000) * u.Hz,
                                                  show=False)
fig, ax = vbs.plot_sources_on_sc(scatter_s=100, marker="*", snr_cutoff=7, c=vbs.m_1[vbs.snr > 7].to(u.Msun),
                                 fig=fig, ax=ax, show=False, cmap="Oranges", vmin=0.0, vmax=1.0)
cbar = fig.colorbar(ax.get_children()[1])
cbar.set_label(r"Primary Mass, $m_1 \, [{\rm M_{\odot}}]$")

ax.legend(handles=[ax.get_children()[1]], labels=["LISA Verification Binaries (Kupfer+18)"], fontsize=0.7*fs, markerscale=2)

ax.set_rasterization_zorder(10000)

plt.savefig("verification_binaries_on_sc.pdf", format="pdf", bbox_inches="tight")
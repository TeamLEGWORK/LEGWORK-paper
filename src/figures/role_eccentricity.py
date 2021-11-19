import legwork as lw
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.visualization import quantity_support
quantity_support()

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

# set eccentricities
ecc = np.array([1e-6, 0.6, 0.9])
n_binaries = len(ecc)

# use constant values for mass, f_orb and distance
m_1 = np.repeat(0.6, n_binaries) * u.Msun
m_2 = np.repeat(0.6, n_binaries) * u.Msun
f_orb = np.repeat(1.5e-3, n_binaries) * u.Hz
dist = np.repeat(15, n_binaries) * u.kpc

# get the SNR in each harmonic
snr2_n = lw.snr.snr_ecc_evolving(m_1=m_1, m_2=m_2, f_orb_i=f_orb, ecc=ecc, dist=dist,
                                 harmonics_required=100, t_obs=4 * u.yr, n_step=1000,
                                 ret_snr2_by_harmonic=True)

snr = snr2_n.sum(axis=1)**(0.5)

# plot LISA sensitivity curve
fig, ax = lw.visualisation.plot_sensitivity_curve(frequency_range=np.logspace(-3.5, 0, 1000) * u.Hz,
                                                  show=False)

# plot each sources
colours = [plt.get_cmap("plasma")(i) for i in [0.1, 0.5, 0.8]]
for i in range(len(snr2_n)):
    # work out the harmonic frequencies and ASDs
    f_harm = f_orb[i] * range(1, len(snr2_n[0]) + 1)
    y_vals = lw.psd.lisa_psd(f_harm)**(0.5) * np.sqrt(snr2_n)[i]

    # only plot points above the sensitivity curve
    mask = np.sqrt(snr2_n)[i] > 1.0

    # compute the index of the maximal SNR value
    max_index = np.argmax(y_vals[1:]) + 1

    # plot each harmonic
    ax.scatter(f_harm[mask], y_vals[mask],
               s=70, color=colours[i],
               label=r"$e={{{:1.1f}}}$".format(ecc[i]))

    # annotate each source with its SNR at the max SNR value
    ax.annotate(r"$\rho={{{:1.0f}}}$".format(snr2_n[i].sum()**(0.5)),
                xy=(f_harm[max_index].value, y_vals[max_index].value * 1.05),
                ha="center", va="bottom", fontsize=0.9*fs, color=colours[i])
    # plot a dotted line to highlight where the signal is concentrated
    ax.plot([f_harm[max_index].value] * 2, [1e-20, y_vals[max_index].value],
               color="grey", linestyle="dotted", lw=2, zorder=0)

# add a legend and annotate the other source properties
ax.legend(markerscale=2.5, handletextpad=0.0, ncol=3, loc="upper center",
          columnspacing=0.75, fontsize=0.85 * fs)

annotation_string = r"$m_1 = 0.6 \, {\rm M_{\odot}}$"
annotation_string += "\n"
annotation_string += r"$m_2 = 0.6 \, {\rm M_{\odot}}$"
annotation_string += "\n"
annotation_string += r"$D_L = 15 \, {\rm kpc}$"
annotation_string += "\n"
annotation_string += r"$f_{\rm orb} = 1.5 \, {\rm mHz}$"

ax.annotate(annotation_string, xy=(3.5e-1, 1.3e-20), ha="center", va="bottom", fontsize=0.8 * fs,
            bbox=dict(boxstyle="round", fc="white", ec="white", alpha=0.3))

ax.set_ylim(1e-20, 2e-18)

ax.set_rasterization_zorder(10000)

plt.savefig("role_eccentricity.pdf", format="pdf", bbox_inches="tight")
import legwork as lw
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
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

f_range = np.logspace(-5, -1, 100) * u.Hz
e_range = np.linspace(0, 0.99, 500)

m_1 = np.repeat(10, len(f_range) * len(e_range)) * u.Msun
m_2 = np.repeat(10, len(f_range) * len(e_range)) * u.Msun

F, E = np.meshgrid(f_range, e_range)

t_merge = lw.evol.get_t_merge_ecc(ecc_i=E.flatten(), f_orb_i=F.flatten(), m_1=m_1, m_2=m_2,
                                  small_e_tol=0.15, large_e_tol=0.9999).reshape(F.shape)
fig, ax = plt.subplots()

cont = ax.contourf(F, E, np.log10(t_merge.to(u.yr).value), cmap="plasma_r", levels=np.linspace(-6, 10, 17))
cbar = fig.colorbar(cont, label=r"Merger time, $\log_{10} (t_{\rm merge} / {\rm yr})$")
ax.set_xscale("log")

# hide edges that show up in rendered PDFs
for c in cont.collections:
    c.set_edgecolor("face")

mass_string = ""
mass_string += r"$m_1 = {{{}}} \, {{ \rm M_{{\odot}}}}$".format(m_1[0].value)
mass_string += "\n"
mass_string += r"$m_2 = {{{}}} \, {{ \rm M_{{\odot}}}}$".format(m_2[0].value)
ax.annotate(mass_string, xy=(0.5, 0.04), xycoords="axes fraction", fontsize=0.6*fs,
            bbox=dict(boxstyle="round", color="white", ec="white", alpha=0.5), ha="center", va="bottom")

ax.set_xlabel(r"Orbital frequency, $f_{\rm orb} \, [\rm Hz]$")
ax.set_ylabel(r"Eccentricity, $e$")

mission_length = ax.contour(F, E, np.log10(t_merge.to(u.yr).value), levels=np.log10([4]),
                            linestyles="--", linewidths=2)
ax.clabel(mission_length, fmt={np.log10(4): r"$t_{\rm merge} = 4\,{\rm years}$"}, fontsize=0.7*fs, manual=[(1e-2, 0.5)])

ax.set_rasterization_zorder(10000)

plt.savefig(paths.figures / "merger_time.pdf", format="pdf", bbox_inches="tight")
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
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)

# create new figure
fig, ax = plt.subplots()

# define the frequency range of interest
fr = np.logspace(-4, 0, 1000) * u.Hz

# plot a sensitivity curve for different mission lengths
linewidth = 4
for i, t_obs in enumerate([0.5, 2.0, 4.0]):
    lw.visualisation.plot_sensitivity_curve(frequency_range=fr, t_obs=t_obs * u.yr,
                                            fig=fig, ax=ax, show=False, fill=False, linewidth=linewidth,
                                            color=plt.get_cmap("Blues")((i + 1) * 0.3),
                                            label=r"LISA ($t_{{\rm obs}} = {{{}}} \, {{\rm yr}}$)".format(t_obs))

# plot the LISA curve with an approximate response function
lw.visualisation.plot_sensitivity_curve(frequency_range=fr, approximate_R=True,
                                        fig=fig, ax=ax, show=False, fill=False, linewidth=linewidth,
                                        color=plt.get_cmap("Purples")(0.5),
                                        label=r"LISA (approximate $\mathcal{R}(f)$)")

# plot the TianQin curve
lw.visualisation.plot_sensitivity_curve(frequency_range=fr, instrument="TianQin", label="TianQin",
                                        fig=fig, ax=ax, show=False, linewidth=linewidth,
                                        color=plt.get_cmap("Greens")(0.6), fill=False)

ax.legend(fontsize=0.7*fs)

ax.set_rasterization_zorder(10000)

plt.savefig("detector_sc_compare.pdf", format="pdf", bbox_inches="tight")
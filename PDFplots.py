import numpy             as np
import matplotlib.pyplot as plt



plt.close('all')
plt.rcParams.update({
  'figure.max_open_warning': 0,
  'text.usetex':             True,
  'figure.autolayout':       True,
  'font.family':             "STIXGeneral",
  'mathtext.fontset':        "stix",
  'font.size':              10,
  'axes.titlesize':         10,
  'figure.labelsize':       10,
  'figure.titlesize':       10,
  'legend.fontsize':        10,
  'legend.handlelength':     1,
  'legend.handletextpad':    0.5,
  'legend.borderpad':        0.5,
  'legend.borderaxespad':    0.5,
  'legend.columnspacing':    1,
  'legend.framealpha':       1,
  'legend.fancybox':         True,
  'axes.grid':               True,
  'axes.grid.axis':          'both',
  'grid.alpha':              0.2,
  'grid.linewidth':          0.4,
  'xtick.labelsize':        10,
  'ytick.labelsize':        10,
  'lines.linewidth':         1,
  'lines.markersize':        5,
  'savefig.transparent':     True,
  'savefig.pad_inches':      0.01,
  'savefig.format':          'pdf',
  'savefig.bbox':            'tight'
})
plt.rcParams['text.latex.preamble']= r"\usepackage{amsmath}"



# READ FILES

file1 = open('PDF__g_r.txt', "r")
rlin = np.fromstring(file1.readlines()[1], dtype=float, sep=' ')
file1.close()

file2 = open('PDF__g_r_theta.txt', "r")
thetalin = np.fromstring(file2.readlines()[1], dtype=float, sep=' ')
file2.close()

gr      = np.loadtxt('PDF__g_r.txt',       skiprows=4)
grtheta = np.loadtxt('PDF__g_r_theta.txt', skiprows=4)

rgridtheta, thetagrid = np.meshgrid(rlin, thetalin)

rmed = rlin[0:-1] + 0.5*(rlin[1]-rlin[0])



# PLOTS

fig1, ax1 = plt.subplots(1,1, figsize=(16,8))
ax1.set_xlabel(r"$r$")
ax1.set_ylabel(r"$g\left(r\right)$")
ax1.plot(rmed, gr, '--ok')
ax1.set_ylim(bottom=0.)
ax1.grid(True, which="both")

fig2, ax2 = plt.subplots(1,1, figsize=(8,8))
ax2.set_xlabel(r"$r \cdot \rm{cos}\left(\theta\right)$")
ax2.set_ylabel(r"$r \cdot \rm{sin}\left(\theta\right)$")
ax2.set_aspect('equal')
pcolor2 = ax2.pcolormesh(rgridtheta*np.cos(thetagrid), rgridtheta*np.sin(thetagrid), grtheta.transpose(), cmap='viridis')
cbar2 = fig2.colorbar(pcolor2, fraction=0.0455, extend='max')
cbar2.set_label(r"$g\left(r,\theta\right)$")





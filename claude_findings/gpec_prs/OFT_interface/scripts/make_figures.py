"""Figures for the OFT_interface report: GPEC GSE vs file size, and Delta'(2/1) vs mpsi, g-file vs i-file.

usage: python make_figures.py RESULTS_ROOT FIGDIR
RESULTS_ROOT holds m{128,256,512}/ifile_vs_gfile.json from run_ifile_vs_gfile.py (mpsi=...).
"""
import os, sys, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT, FIG = sys.argv[1], sys.argv[2]
os.makedirs(FIG, exist_ok=True)
INK, MUTED = '#0b0b0b', '#52514e'
COL = {'efit': '#1baf7a', 'ldp_i': '#eda100'}
LAB = {'efit': 'g-file (efit)', 'ldp_i': 'i-file (ldp_i)'}
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'font.size': 10, 'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False})
mp = (128, 256, 512)
res = {m: json.load(open(os.path.join(ROOT, f'm{m}', 'ifile_vs_gfile.json'))) for m in mp}

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for t, mk in (('efit', 's'), ('ldp_i', 'D')):
    names = sorted((n for n, v in res[128].items() if v['eq_type'] == t), key=lambda n: res[128][n]['bytes'])
    ax[0].scatter([res[128][n]['bytes'] / 1e6 for n in names], [res[128][n]['int_med'] for n in names],
                  color=COL[t], marker=mk, s=45, label=LAB[t], edgecolor='white', linewidth=1)
    for n in names:
        ax[0].annotate(n[1:], (res[128][n]['bytes'] / 1e6, res[128][n]['int_med']), textcoords='offset points',
                       xytext=(5, 4), fontsize=8, color=MUTED)
    for n, ls in zip(names, (':', '--', '-')):
        ax[1].plot(mp, [res[m][n]['dp_diag'][0] for m in mp], ls=ls, marker=mk, ms=7, color=COL[t],
                   label=f'{LAB[t].split()[0]} {n[1:]}')
ax[0].set_xscale('log'); ax[0].set_yscale('log')
ax[0].set_xlabel('file size [MB]'); ax[0].set_ylabel('GPEC GSE, integrated, median'); ax[0].set_title('mpsi = 128', color=INK, fontsize=10)
ax[0].legend(fontsize=8)
ax[1].set_xscale('log', base=2); ax[1].set_xticks(mp, [str(m) for m in mp]); ax[1].set_xlabel('GPEC mpsi')
ax[1].set_ylabel(r"STRIDE $\Delta'$ (2/1)"); ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'ifile_vs_gfile.png'), dpi=150); plt.close(fig)
print('wrote', FIG)

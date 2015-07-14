"""style.py
Contains standard style for figures.
"""
from matplotlib.colors import colorConverter
from matplotlib import rcParams

# Colors are layered: two types a four layers
# Source: http://colorbrewer2.org/
colors =   [
            "#08519c",
            "#9ecae1",
            "#a63603",
            "#fdae6b",
            "#54278f",
            "#bcbddc",
            "#006d2c",
            "#a1d99b"
            ]

try:
    import seaborn as sns
    sns.set(style='ticks', palette='Set1') 
    sns.despine()
except:
    print('seaborn not installed')

xfactor = 3
rcParams['figure.figsize'] = (xfactor*6.2, xfactor*3.83)  

fontsize_labels         = 20    # size used in latex document
fontsize_labels_axes    = fontsize_labels + 2
fontsize_labels_title   = fontsize_labels + 4
axes_color = "0.8" 

rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage[cmbright]{sfmath}']
rcParams['font.family']= 'sans-serif'
rcParams['font.sans-serif']= 'cmbright'
rcParams['font.weight'] = "light"
rcParams['figure.autolayout']   = True
rcParams['font.size']           = fontsize_labels
rcParams['xtick.labelsize']     = fontsize_labels
rcParams['ytick.labelsize']     = fontsize_labels
rcParams['legend.fontsize']     = fontsize_labels
rcParams['axes.labelsize']      = fontsize_labels_axes
rcParams['axes.titlesize']      = fontsize_labels_title
rcParams['legend.markerscale']  = 4
rcParams['text.color'] = "0.3"
rcParams['xtick.color'] = "0.3"
rcParams['ytick.color'] = "0.3"
rcParams['axes.labelcolor'] = "0.3"
rcParams['axes.edgecolor'] = axes_color
rcParams['axes.grid'] = True

def fixticks(ax):    
    ax.grid(False)      # Turn of grid (distracts!)
    # Set spines to color of axes
    for t in ax.xaxis.get_ticklines(): t.set_color(axes_color)
    for t in ax.yaxis.get_ticklines(): t.set_color(axes_color)
    # Remove top and right axes & spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

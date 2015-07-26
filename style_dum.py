"""style_dum.py
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

rcParams['figure.figsize'] = (6.2, 3.83)  

fontsize_labels         = 12    # size used in latex document
fontsize_labels_axes    = fontsize_labels
fontsize_labels_title   = fontsize_labels
axes_color = "#bdbdbd" 
text_color = "#636363"

params = {
           # Font: Size and type
          'text.usetex':         True,
          'text.latex.preamble': [r'\usepackage[cmbright:{sfmath}'],
          #'text.latex.unicode':   True,
          'font.family':         'sans-serif',
          'font.sans-serif':     'cmbright',
          'font.weight':         "light",
          'figure.autolayout':   True,
          'font.size':           fontsize_labels,
          'xtick.labelsize':     fontsize_labels,
          'ytick.labelsize':     fontsize_labels,
          'legend.fontsize':     fontsize_labels,
          'axes.labelsize':      fontsize_labels_axes,
          'axes.titlesize':      fontsize_labels_title,
           # Color
          'text.color':          text_color, 
          'xtick.color':         text_color, 
          'ytick.color':         text_color,
          'axes.labelcolor':     text_color,
          'axes.edgecolor':      axes_color,
           # Other
          'legend.markerscale':  4,
          'axes.grid':           False
         }
rcParams.update(params) 

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

from matplotlib.colors import colorConverter
from matplotlib import rcParams
import sys, os
sys.path.append(os.path.abspath('../')) # include path with simulation specifications
from imp import reload
import style; reload(style)
from style import *

fontsize_labels = 24    # size used in latex document
rcParams['font.size'] = fontsize_labels
rcParams['axes.labelsize'] = fontsize_labels
rcParams['xtick.labelsize'] = fontsize_labels
rcParams['ytick.labelsize'] = fontsize_labels
rcParams['legend.fontsize'] = fontsize_labels
rcParams['axes.titlesize'] = fontsize_labels

xfactor = 2
rcParams['figure.figsize'] = (xfactor*6.2, xfactor*3.83)  


#try:
    #import seaborn as sns
    #sns.set(style='ticks', palette='Set1') 
    #sns.despine()
    ## These are the colors. Notice how this is programmed:
    ## You initialize your colors by 
    ## colorset = palette()
    ## then you can cycle through the colors:
    ## color = next(colorset)
    ## if you want your set to be reset, just create
    ## a new palette() instance! This way the colors do not interfere.
    #color_names = ['windows blue', "pale red", "faded green", "amber", 
              #'dark green', 'dark fuchsia', 'browny orange', 
              #'puke green', 'dark royal blue', 'dusty purple', 'red orange']
    #colors = sns.xkcd_palette(color_names)
    #palette = lambda: itertools.cycle(sns.xkcd_palette(color_names) )
#except:
    #print('seaborn not installed')


#fontsize_labels = 24    # size used in latex document
##rcParams['text.usetex'] = True
##rcParams['text.latex.preamble'] = [r'\usepackage[cmbright]{sfmath}']
#rcParams['font.family']= 'sans-serif'
##rcParams['font.sans-serif']= 'cmbright'
#rcParams['font.weight'] = "light"
#rcParams['figure.autolayout'] = True
#rcParams['font.size'] = fontsize_labels
#rcParams['axes.labelsize'] = fontsize_labels
#rcParams['xtick.labelsize'] = fontsize_labels
#rcParams['ytick.labelsize'] = fontsize_labels
#rcParams['legend.fontsize'] = fontsize_labels
#rcParams['legend.markerscale'] = 4
#rcParams['axes.titlesize'] = fontsize_labels
#rcParams['text.color'] = "0.3"
#rcParams['xtick.color'] = "0.3"
#rcParams['ytick.color'] = "0.3"
#rcParams['axes.labelcolor'] = "0.3"
#rcParams['axes.edgecolor'] = "0.8"
#rcParams['axes.grid'] = True

#xfactor = 2
#rcParams['figure.figsize'] = (xfactor*6.2, xfactor*3.83)  

#def fixticks(ax):    
    #for t in ax.xaxis.get_ticklines(): t.set_color('0.8')

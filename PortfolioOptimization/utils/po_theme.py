# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:35:16 2017

@author: utilisateur
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

#print(plt.style.available)
List_Style = ['bmh','classic','dark_background','fivethirtyeight','ggplot','grayscale','seaborn-bright','seaborn-colorblind','seaborn-dark',
              'seaborn-dark-palette','seaborn-darkgrid','seaborn-deep','seaborn-muted','seaborn-notebook','seaborn-paper','seaborn-pastel',
              'seaborn-poster','seaborn-talk','seaborn-ticks','seaborn-white','seaborn-whitegrid']

plt.style.use(List_Style[3])

mpl.rcParams['font.size'] = 7
mpl.rcParams['lines.linewidth'] = 1.0
            
mpl.rcParams['axes.edgecolor'] = 'grey'
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.bottom'] = True
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.right'] = True
#mpl.rcParams['axes.prop_cycle'] = cycler(color='')
            
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5
 
mplColorMapSetting = plt.cm.viridis#plasma

'''

=========== Matlab colors ===========
autumn	   #sequential linearly-increasing shades of red-orange-yellow
bone	      #sequential increasing black-white color map with a tinge of blue, to emulate X-ray film
cool	      #linearly-decreasing shades of cyan-magenta
copper	   #sequential increasing shades of black-copper
flag	      #repetitive red-white-blue-black pattern (not cyclic at endpoints)
gray	      #sequential linearly-increasing black-to-white grayscale
hot	      #sequential black-red-yellow-white, to emulate blackbody radiation from an object at increasing temperatures
hsv	      #cyclic red-yellow-green-cyan-blue-magenta-red, formed by changing the hue component in the HSV color space
inferno	   #perceptually uniform shades of black-red-yellow
jet	      #a spectral map with dark endpoints, blue-cyan-yellow-red; based on a fluid-jet simulation by NCSA [1]
magma	      #perceptually uniform shades of black-red-white
pink	      #sequential increasing pastel black-pink-white, meant for sepia tone colorization of photographs
plasma	   #perceptually uniform shades of blue-red-yellow
prism	      #repetitive red-yellow-green-blue-purple-...-green pattern (not cyclic at endpoints)
spring	   #linearly-increasing shades of magenta-yellow
summer	   #sequential linearly-increasing shades of green-yellow
viridis     #perceptually uniform shades of blue-green-yellow
winter	   #linearly-increasing shades of blue-green

=========== Yorick colors ===========
gist_earth	#mapmaker’s colors from dark blue deep ocean to green lowlands to brown highlands to white mountains
gist_heat	#sequential increasing black-red-orange-white, to emulate blackbody radiation from an iron bar as it grows hotter
gist_ncar	#pseudo-spectral black-blue-green-yellow-red-purple-white colormap from National Center for Atmospheric Research [2]
gist_rainbow	#runs through the colors in spectral order from red to violet at full saturation (like hsv but not cyclic)
gist_stern	 #“Stern special” color table from Interactive Data Language software

=========== From ColorBrewer develop by Cynthia Brewer ===========
BrBG	   #brown, white, blue-green
PiYG	   #pink, white, yellow-green
PRGn	   #purple, white, green
PuOr	   #orange, white, purple
RdBu	   #red, white, blue
RdGy	   #red, white, gray
RdYlBu	#red, yellow, blue
RdYlGn	#red, yellow, green
Spectral	#red, orange, yellow, green, blue

=========== Sequential Brewer ===========
Blues	   #white to dark blue
BuGn	   #white, light blue, dark green
BuPu	   #white, light blue, dark purple
GnBu	   #white, light green, dark blue
Greens	#white to dark green
Greys	   #white to black (not linear)
Oranges	#white, orange, dark brown
OrRd	   #white, orange, dark red
PuBu	   #white, light purple, dark blue
PuBuGn	#white, light purple, dark green
PuRd	   #white, light purple, dark red
Purples	#white to dark purple
RdPu	   #white, pink, dark purple
Reds	   #white to dark red
YlGn	   #light yellow, dark green
YlGnBu	#light yellow, light green, dark blue
YlOrBr	#light yellow, orange, dark brown
YlOrRd	#light yellow, orange, dark red

=========== Qualitative data ===========
Accent
Dark2
Paired
Pastel1
Pastel2
Set1
Set2
Set3

=========== Misc ===========
afmhot	     #sequential black-orange-yellow-white blackbody spectrum, commonly used in atomic force microscopy
brg           #blue-red-green
bwr	        #diverging blue-white-red
coolwarm	     #diverging blue-gray-red, meant to avoid issues with 3D shading, color blindness, and ordering of colors [3]
CMRmap	     #“Default colormaps on color images often reproduce to confusing grayscale images. The proposed colormap maintains an aesthetically pleasing color image that automatically reproduces to a monotonic grayscale with discrete, quantifiable saturation levels.” [4]
cubehelix	  #Unlike most other color schemes cubehelix was designed by D.A. Green to be monotonically increasing in terms of perceived brightness. Also, when printed on a black and white postscript printer, the scheme results in a greyscale with monotonically increasing brightness. This color scheme is named cubehelix because the r,g,b values produced can be visualised as a squashed helix around the diagonal in the r,g,b color cube.
gnuplot	     #gnuplot’s traditional pm3d scheme (black-blue-red-yellow)
gnuplot2	     #sequential color printable as gray (black-blue-violet-yellow-white)
ocean	        #green-blue-white
rainbow	     #spectral purple-blue-green-yellow-orange-red colormap with diverging luminance
seismic	     #diverging blue-white-red
nipy_spectral #black-purple-blue-green-yellow-red-white spectrum, originally from the Neuroimaging in Python project
terrain	     #mapmaker’s colors, blue-green-yellow-brown-white, originally from IGOR Pro

'''
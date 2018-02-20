# -*- coding: utf-8 -*-
"""
Created on Tue Nov 07 14:44:29 2017

@author: jb.frisch
"""

from PyQt5 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib

# Ensure using PyQt5 backend
matplotlib.use('QT5Agg')

# Matplotlib canvas class to create figure
class MplSBCanvas(Canvas):
    def __init__(self):
        self.fig = Figure()
        
        self.ax1 = self.fig.add_subplot(311) 
        self.ax2 = self.fig.add_subplot(312, sharex=self.ax1) 
        self.ax3 = self.fig.add_subplot(313, sharex=self.ax1)
        
        self.fig.subplots_adjust(hspace=0)
        
        Canvas.__init__(self, self.fig)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

# Matplotlib widget
class MplSubPlotWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)   # Inherit from QWidget
        self.canvas = MplSBCanvas()                  # Create canvas object
        self.vbl = QtWidgets.QVBoxLayout()         # Set box for plotting
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:41:38 2017

@author: Jean-Baptiste Frisch
"""

from ctypes import WinDLL
from ctypes import byref
from ctypes import POINTER
from ctypes import c_double
from ctypes import c_int
from ctypes import c_long
#import _ctypes

import os.path as pth

import numpy as np
import numpy.ctypeslib as npct


array_1d_double = npct.ndpointer(dtype=c_double, ndim=1, flags='C_CONTIGUOUS')
array_1d_integer = npct.ndpointer(dtype=c_long, ndim=1, flags='C_CONTIGUOUS')

dllabspath = pth.dirname(pth.abspath(__file__)) + pth.sep + "Opt_Histogram.dll"

lib_Opt_Hist = WinDLL(dllabspath)
#lib_Opt_Hist = npct.load_library("Opt_Histogram", ".")
libHandle = lib_Opt_Hist._handle

Opt_bins = lib_Opt_Hist.Opt_Bins
Opt_bins.restype = None
''' Pointer sur:
        - Integer : result
        - Integer : max bins
        - Double : data
        - Integer : taille data
'''
Opt_bins.argtypes = [POINTER(c_int),
                     POINTER(c_int),
                     array_1d_double,
                     POINTER(c_int)]

Edges = lib_Opt_Hist.Edges_hist
Edges.restype = None
''' Pointer sur:
        - Double : result edges
        - Double : data
        - Integer : nb bins
        - Integer : taille data
'''
Edges.argtypes = [array_1d_double,
                  array_1d_double,
                  POINTER(c_int),
                  POINTER(c_int)]

Counts_Edg = lib_Opt_Hist.Counts_hist
Counts_Edg.restype = None
''' Pointer sur:
        - Integer : result counts
        - Double : data
        - Integer : nb bins
        - Integer : taille data
'''
Counts_Edg.argtypes = [array_1d_integer,
                       array_1d_double,
                       POINTER(c_int),
                       POINTER(c_int)]

def Opt_nb_bins(data, max_bins=50):
    res = c_int()
    buff = np.ascontiguousarray(data.values, dtype=c_double)
    
    Opt_bins(byref(res), byref(c_int(max_bins)), buff, byref(c_int(len(buff))))

    return res.value


def Opt_Edges(data, nb_bins=12):
    edges = np.ascontiguousarray(np.empty(nb_bins, dtype=c_double))
    buff = np.ascontiguousarray(data.values, dtype=c_double)
    
    Edges(edges, buff, byref(c_int(nb_bins)), byref(c_int(len(buff))))
    
    return edges

def Opt_Bins(data, nb_bins=12):
    bins = np.ascontiguousarray(np.empty(nb_bins, dtype=c_long))
    buff = np.ascontiguousarray(data.values, dtype=c_double)
    
    Counts_Edg(bins, buff, byref(c_int(nb_bins)), byref(c_int(len(buff))))
    
    return bins

def Opt_Histogram(data, max_bins=50):
    buff = np.ascontiguousarray(data.values, dtype=c_double)
    
    opt_bins = c_int()
    Opt_bins(byref(opt_bins), byref(c_int(max_bins)), buff, byref(c_int(len(buff))))
    
    edges = np.ascontiguousarray(np.empty(opt_bins.value, dtype=c_double))
    bins = np.ascontiguousarray(np.empty(opt_bins.value, dtype=c_long))
    
    Edges(edges, buff, opt_bins, byref(c_int(len(buff))))
    Counts_Edg(bins, buff, opt_bins, byref(c_int(len(buff))))
    
    return [edges, bins]
    
#### Free DLL
#_ctypes.FreeLibrary(lib_Opt_Hist._handle)
#del my_dll    
    
    
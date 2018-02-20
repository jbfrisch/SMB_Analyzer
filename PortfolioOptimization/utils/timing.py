# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 18:40:38 2017

@author: jb.frisch
"""

import time
#from timeit import default_timer as timer

def timing(f):
    def wrap(*args):
        time1 = time.time()
        #str = timer()
        ret = f(*args)
        time2 = time.time()
        #ellapse = timer() - str
        print '----- %s function took:' % (f.func_name)
        print '\t\t- %0.5f ms' % ((time2-time1)*1000.0)
        #print u'\t\t- %0.5f \u03bcs' % ((time2-time1)*1e6)
        return ret
    return wrap
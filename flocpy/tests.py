import flocpy
import os
import numpy as np


__all__ = ['dotest']

def dotest():
    # File name formatting string things
    datadir = flocpy.__path__[0]+os.sep+'testdata'
    print(datadir)
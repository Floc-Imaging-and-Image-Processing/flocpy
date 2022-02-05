
import os
import glob

def filesorter(path, extension, sortby=os.path.basename):
    unsorted_flist = glob.glob(path+os.sep+'*'+extension)
    return sorted(unsorted_flist, key=sortby)

def testdata_paths():
    relpath = os.path.dirname(__file__)
    return glob.glob(os.path.join(relpath, 'testdata', '*/'))

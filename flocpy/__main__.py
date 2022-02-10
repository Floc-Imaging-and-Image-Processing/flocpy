import sys
import os
import flocpy
import glob

def main():
	unsorted_flist = glob.glob(os.getcwd()+os.sep+'*')
	sorted_flist = sorted(unsorted_flist, key=os.path.basename)
	flocpy.identify_flocs(sorted_flist, save=True, return_data=False)

if __name__ == '__main__':
	main()
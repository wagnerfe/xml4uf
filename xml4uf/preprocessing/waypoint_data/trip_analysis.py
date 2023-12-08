import sys, os
import pandas as pd
import skmob as skm
from skmob.measures import individual

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

# import helper functions
from ufo_map.Utils.helpers import *

# define class for cleaning
class Trip_Stats():

    def __init__(self):
        # variables
        self.df = None
        # output
        self.out = None
    
    def load_data(self, path,sample_size=None):
        # load csv
        if sample_size: 
            chunks = pd.read_csv(path, chunksize=sample_size)
            for idx,self.df in enumerate(chunks):
                print('reading in sample of size {}'.format(len(df)))
                if idx==0: break
        else: self.df = pd.read_csv(path)



def main(chunk_size=None):
    
    # define input and output paths
    path_in = ''
    path_out = '/p/projects/eubucco/test_felix/data'

    stats = Trip_Stats()
    # load data
    
    for i in range(1,14):
        stats.load_data(path_in,sample_size=chunk_size)
    

    

    print('closing run.')

if __name__ == "__main__":
    main() 

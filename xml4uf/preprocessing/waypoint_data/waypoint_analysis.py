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
class Waypoint_Stats():

    def __init__(self):
        # tdf
        self.tdf = None
        self.grouped = None
        # variables
        self.n_points = None
        self.l_trip = None
        self.t_trip = None
        self.points_per_s = None
        self.points_per_t = None
        self.v_mean_trip = None
        self.v_max_trip = None
        self.v_min_trip = None
        self.a_mean_trip = None
        self.a_max_trip = None
        self.a_min_trip = None
        # output
        self.out = None
    
    def load_data(self, path,sample_size=None):
        # load csv
        if sample_size: 
            chunks = pd.read_csv(path, chunksize=sample_size)
            for idx,df in enumerate(chunks):
                print('reading in sample of size {}'.format(len(df)))
                if idx==0: break
        else: df = pd.read_csv(path)
        
        # load as trjaectory dataframe
        self.tdf = skm.TrajDataFrame(df, 
                          latitude='lat', longitude='lon', 
                          datetime='CaptureDate')
        
        # load it also as a grouped object
        self.grouped = self.tdf.groupby("TripID")

    def get_v_stats(self):
        # function calculates average velocities [m/s] 
        #self.out['v_mean'] = np.mean(self.out.v_points[0])
        self.out['v_mean'] = self.out.v_points.apply(lambda x: np.mean(x, axis=0))
        self.out['v_min'] = self.out.v_points.apply(lambda x: np.min(x, axis=0))
        self.out['v_max'] = self.out.v_points.apply(lambda x: np.max(x, axis=0))

    def get_a_stats(self):
        # function calculates average accelerations [m/s^2] 
        self.out['a_mean'] = self.out.a_points.apply(lambda x: np.mean(x, axis=0))
        self.out['a_min'] = self.out.a_points.apply(lambda x: np.min(x, axis=0))
        self.out['a_max'] = self.out.a_points.apply(lambda x: np.max(x, axis=0))
    
    def get_n_stats(self):
        self.out['n_points'] = self.out.v_points.str.len()
        self.out['dist'] = self.out.jump_lengths.apply(lambda x: np.sum(x,axis=0))
        self.out['time'] = self.out.waiting_times.apply(lambda x: np.sum(x,axis=0))


def main(chunk_size=None):
    
    # define input and output paths
    path_in = '/p/projects/vwproject/felix_files/data/original_trip_data/waypoints_with_edges.csv'
    path_out = '/p/projects/eubucco/test_felix/data'

    stats = Waypoint_Stats()
    # load data
    stats.load_data(path_in,sample_size=chunk_size)
    
    # intialise dict with stats
    dict_stats = {'ids':[],
                    'waiting_times':[],
                    'jump_lengths':[],
                    'v_points':[],
                    'a_points':[]}


    # calculate waiting times, jump lenghts, velocities and acceleration on all points per group
    list_wt = []
    list_jl = []
    list_vp = []
    list_ap = []
    for id, group in stats.grouped:
        list_wt = individual.waiting_times(group).waiting_times[0]
        list_jl = individual.jump_lengths(group).jump_lengths[0]
        # filter out trips with only one waypoint
        if len(list_wt)!=0:
            list_vp = list_jl*1e3/list_wt
            list_ap = (list_vp[1:]-list_vp[:-1])/list_wt[:-1]
            list_ap = np.insert(list_ap,0,0)
            dict_stats['ids'].append(id)
            dict_stats['waiting_times'].append(list_wt)
            dict_stats['jump_lengths'].append(list_jl*1e3)
            dict_stats['v_points'].append(list_vp)
            dict_stats['a_points'].append(list_ap)
    
    # generate output df
    stats.out=pd.DataFrame(dict_stats)
    print(stats.out.head(3))
    
    # calculate additional measures
    stats.get_v_stats()
    stats.get_a_stats()
    stats.get_n_stats()
    
    # saving output
    n_sample=len(stats.out)
    print("saving stats to .csv for {} trips".format(n_sample))
    stats.out.to_csv(os.path.join(path_out,'waypoint_stats_'+str(n_sample)+'_samples_raw.csv'),index=False)

    print('saving without points')
    stats.out = stats.out.drop(['waiting_times','jump_lengths','v_points','a_points'],axis=1)
    stats.out.to_csv(os.path.join(path_out,'waypoint_stats_'+str(n_sample)+'_samples.csv'),index=False)

    print('closing run.')

if __name__ == "__main__":
    main() 

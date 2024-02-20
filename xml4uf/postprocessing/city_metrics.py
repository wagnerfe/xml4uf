import sys, os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import momepy
import h3


# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)
print(PROJECT_SRC_PATH)

# imports
from ufo_map.Utils.helpers import *
from utils.utils import get_input,load_data,get_crs_local
from utils.utils_h3 import convert_pop_to_parent, convert_od_to_parent, add_h3_geom

# define constants
CRS_UNI = 'epsg:4326'


class CityMetrics():

    def __init__(self,name,run_name,path_root,metrics, buff_dist,to_hex_size):
        self.city_name = name
        self.crs_local = get_crs_local(self.city_name)
        self.run_name = run_name
        self.path_root = path_root
        self.metrics = metrics
        self.buff_dist = buff_dist # buff_size provided in m
        self.to_hex_size = to_hex_size
        self.g = None
        self.nodes = None
        self.gdf_zip = None
        self.gdf_cbd = None
        self.gdf_pop = None
        self.gdf_bound = None
        self.gdf_buff = None 
        self.dict_out = None
        self.df_stats = None
    

    def calc_global_closeness(self):
        print('intiating global closeness calculation...')
        edges = ox.graph_to_gdfs(self.g,nodes=False, edges=True,node_geometry=False, fill_edge_geometry=True)
        primal = momepy.gdf_to_nx(edges, approach='primal')
        
        print('calculating global closeness...')
        primal_gc = momepy.closeness_centrality(primal, name='closeness_global', weight='mm_len')
        self.nodes_gc = momepy.nx_to_gdf(primal_gc, lines=False)
        self.nodes_gc = self.nodes_gc.to_crs(self.crs_local)


    # def cbd_city(self):
    #     # assign city_name to city
    #     dict_cities = {'bos':'Boston',
    #         'lax':'Los Angeles',
    #         'sfo':'San Francisco',
    #         'rio':'Rio',
    #         'lis':'Lisbon'}
    #     return ox.geocode_to_gdf(dict_cities[self.city_name]).centroid


    def get_rings(self,gdf_bound,gdf_cbd,buff_dist=1000):
        """calcuate rings with given ring distance
        in:
        - gdf_bound: gdf of boundary of city
        - gdf_cbd: gdf with cbd geometry point
        - buff_dist: distance of buffer in meter (default 1000m)
        out:
        - gdf_buff: gdf with buffer polygons
        """
        # calc hausdorff distance to determine number of rings
        max_dist = gdf_cbd.iloc[0].geometry.hausdorff_distance(gdf_bound.boundary[0])

        gdf_buff = gpd.GeoDataFrame()
        buff_size = 0
        for i in range(int(max_dist/buff_dist)):
            buff_size = buff_size + buff_dist
            gdf_buff = pd.concat([gdf_buff,gpd.GeoDataFrame(geometry=gdf_cbd.buffer(buff_size)).reset_index(drop=True)])
        return gdf_buff.reset_index(drop=True)


    def get_diff(self,gdf,gdf_buff,n):
        n_1 = n+1
        gdf_n = gdf_buff.iloc[[n]]
        gdf_n_1 = gdf_buff.iloc[[n_1]]
        
        gdf_join_inner = gpd.sjoin(gdf,gdf_n) # join all
        gdf_join_outer = gpd.sjoin(gdf,gdf_n_1) # join inner
        return gdf_join_outer.loc[~gdf_join_outer.hex_id.isin(gdf_join_inner.hex_id)]


    def get_stats_buff(self,gdf,gdf_buff,buff_dist,dist_col,pop_col,point_col):
        df_out = pd.DataFrame()
        for i in range(len(gdf_buff)-1):
            gdf_tmp = self.get_diff(gdf,gdf_buff,i)
            mean_dist = gdf_tmp[dist_col].mean()
            mean_pop = gdf_tmp[pop_col].mean()
            mean_point = gdf_tmp[point_col].mean()
            tot_point = gdf_tmp[point_col].sum()
            df_out = pd.concat([df_out,pd.DataFrame({'buff':[(i+1)*buff_dist],
                                                    'mean_dist':[mean_dist],
                                                    'mean_pop':[mean_pop],
                                                    'mean_point':[mean_point],
                                                    'tot_point':[tot_point]})])
        return df_out.reset_index(drop=True)


    def pop_dense_to_cbd(self):
        self.gdf_buff = self.get_rings(self.gdf_bound,self.gdf_cbd,self.buff_dist)
        gdf_pop_tmp = convert_pop_to_parent(self.gdf_pop, self.to_hex_size)
        gdf_o_tmp = convert_od_to_parent(self.gdf_o, self.to_hex_size)
        print('samples after conversion to h{}: pop dense: {}, trips: {}'.format(self.to_hex_size,len(gdf_pop_tmp),len(gdf_o_tmp)))

        # merge on id with berlin trip data
        df_pop_merge = pd.merge(gdf_pop_tmp,gdf_o_tmp,on='hex_id')
        gdf_pop_merge = add_h3_geom(df_pop_merge,self.crs_local)
        print('{} matches'.format(len(gdf_pop_merge)))
        self.df_stats = self.get_stats_buff(gdf_pop_merge,self.gdf_buff,self.buff_dist,'distance_m','total','points_in_hex')


    def save_to_disk(self):
        print('saving to disk..')
        if 'global_closeness' in self.metrics:
            self.nodes_gc.to_csv(os.path.join(self.path_root,
                                            '6_analysis',
                                            self.city_name,
                                            self.city_name+
                                            'globalcloseness_epsg'+self.crs_local.split(':',1)[1]+
                                            '.csv'),
                                            index=False)

        if 'pop_dense_cbd' in self.metrics:
            self.df_stats.to_csv(os.path.join(self.path_root,
                                            '6_analysis',
                                            self.city_name,
                                            self.city_name+
                                            '_pop_dense_stats_'+self.run_name+
                                            '_h'+str(self.to_hex_size)+
                                            '.csv'),
                                            index=False)


    def calc_metrics(self):
        #self.assign_run_info(**request)
        print('calculating metrics for {self.city_name}')
        
        if 'global_closeness' in self.metrics:
            dict_out = load_data(self.path_root,
                                self.city_name,
                                self.crs_local,
                                graph=True,
                                nx_nodes=True,
                                zip_codes =True)
            
            self.g = dict_out['g']
            self.n = dict_out['n']
            self.gdf_zip = dict_out['gdf_zip']
            
            self.calc_global_closeness()
        
        if 'pop_dense_cbd' in self.metrics:
            dict_out = load_data(self.path_root,
                                self.city_name,
                                self.crs_local,
                                bound = True,
                                od_h3=True,
                                pop_dense_h3=True,
                                cbd = True)
            
            self.gdf_bound = dict_out['gdf_bound']
            self.gdf_pop = dict_out['gdf_pop']
            self.gdf_o = dict_out['gdf_o']
            self.gdf_cbd = dict_out['gdf_cbd']
            
            self.pop_dense_to_cbd()
        
        self.save_to_disk()
        print('Done. Closing run.')


def main():

    request = get_input(PROJECT_SRC_PATH,'postprocessing/city_metrics.yml')

    cm = CityMetrics(**request)
    cm.calc_metrics()

if __name__ == "__main__":
    main()     

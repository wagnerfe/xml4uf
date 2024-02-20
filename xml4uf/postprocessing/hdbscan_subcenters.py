import os,sys
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import random
from hdbscan import HDBSCAN


# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

from utils.utils import get_input, get_crs_local
from utils.utils_h3 import add_h3_geom
from postprocessing.cut_bounds import CutBounds



def get_rio_metrics(df_soc):
    df_soc_rio = df_soc.loc[df_soc.abbrev_muni.isin(['duq','rio','sgo'])]
    return df_soc_rio[['id_hex','abbrev_muni','T001']].rename(columns={'id_hex':'hex_id','T001':'TotEmp'})


class HDBSCANClusters():
    def __init__(self,name,path_root, day_hour,bound,cluster_percentage,sample_percentage, plot_centers):
        self.city_name = name
        self.crs_local = get_crs_local(self.city_name)
        self.path_root = path_root
        self.day_hour = day_hour
        self.bound = bound
        self.cluster_percentage = cluster_percentage
        self.sample_percentage = sample_percentage
        self.plot_centers = plot_centers
                
        self.path_out = None
        self.file_name = None
        self.gdf_emp = None
        self.gdf_points = None
        self.hulls = None


    def _load_data(self):
        if self.city_name in (['bos','lax','sfo']): 
            file_name = os.path.join(self.path_root,
                                    '0_raw_data',
                                    self.city_name,
                                    'employment',
                                    self.city_name+
                                    '_employment_SmartLocationDatabase.geojson')
            gdf = gpd.read_file(file_name).to_crs(self.crs_local)
        elif self.city_name == 'rio':
            file_name = os.path.join(self.path_root,
                                    '0_raw_data',
                                    self.city_name,
                                    'employment/aop_landuse_2019_v2.csv')
            df = pd.read_csv(file_name)
            df = get_rio_metrics(df)
            gdf = add_h3_geom(df,self.crs_local,poly=True)                    
        else: raise ValueError(f'{self.city_name} not supported')
  
        cb = CutBounds(self.city_name,self.path_root,self.day_hour,False,self.bound)
        cb._get_fua_bound()
        return gdf, cb.gdf_bound


    def initialize_run(self):
        gdf, gdf_bound = self._load_data()
        self.gdf_emp = gpd.sjoin(gdf, gdf_bound).reset_index(drop=True)
        
        if self.sample_percentage is not None:
            self.gdf_emp['TotEmp'] = self.gdf_emp['TotEmp']*self.sample_percentage
            self.gdf_emp['TotEmp'] = self.gdf_emp['TotEmp'].astype(int)

        self.path_out = os.path.join(self.path_root,'0_raw_data',self.city_name)
        self.file_name = (self.city_name+
                    '_clusters_cp'+
                    str(self.cluster_percentage)+
                    '_sp'+str(self.sample_percentage)+
                    '.csv')


    def _random_points_in_polygon(self, number, polygon):
        points = []
        min_x, min_y, max_x, max_y = polygon.bounds
        i= 0
        while i < number:
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(point):
                points.append(point)
                i += 1
        return points  # returns list of shapely point


    def assign_points(self):
        list_sampled_points = []
        for i in range(len(self.gdf_emp)):
            if i == len(self.gdf_emp)/2:
                print(f'50 % done.')
            if self.gdf_emp['geometry'].iloc[i].is_valid:
                sampled_points = self._random_points_in_polygon(self.gdf_emp['TotEmp'].iloc[i], self.gdf_emp['geometry'].iloc[i])
                list_sampled_points+=sampled_points
        self.gdf_points = gpd.GeoDataFrame(geometry=list_sampled_points)
        print(f'assigned {len(self.gdf_points)} points from {len(self.gdf_emp)} tracts')


    def _define_cluster_size(self):
        n_jobs = len(self.gdf_points)
        return int(n_jobs*self.cluster_percentage)


    def _plot_clusters(self):
        f, ax = plt.subplots(1, figsize=(9, 9))
        #self.gdf_emp.plot(ax=ax, alpha=0.1)
        self.gdf_points.plot(ax=ax)
        self.hulls.boundary.plot(ax=ax,cmap='cubehelix')
        plt.savefig(os.path.join(self.path_out,
                                'employment',
                                self.file_name[:-3]+'png'))

    
    def cluster_points(self):
        coordinates = np.column_stack((self.gdf_points.geometry.x, self.gdf_points.geometry.y))
        labels = HDBSCAN(min_cluster_size=self._define_cluster_size()).fit(coordinates).labels_
    
        self.gdf_points['num_jobs'] = 1 # each point equals one job location
        self.hulls = self.gdf_points[['geometry','num_jobs']].dissolve(by=labels, aggfunc={'num_jobs':'sum'})
        self.hulls['geometry'] = self.hulls.geometry.convex_hull
        self.hulls = self.hulls.reset_index()
        self.hulls['center_geometry'] = self.hulls.centroid
        
        print(f'found {len(self.hulls)} clusters')
        if self.plot_centers: self._plot_clusters()


    def save_clusters(self):
        #self.hulls.to_csv(os.path.join(self.path_out, 'employment',self.file_name),index=False)
        gdf_center = self.hulls[['index','num_jobs','center_geometry']].rename(columns={'center_geometry':'geometry'})
        gdf_center = gdf_center.loc[gdf_center['index']!=-1]
        gdf_center.to_csv(os.path.join(self.path_out,'streets',self.file_name))


    def run_clustering(self):
        self.initialize_run()
        self.assign_points()
        self.cluster_points()
        self.save_clusters()


def main():
    request = get_input(PROJECT_SRC_PATH,'postprocessing/hdbscan_subcenters.yml')
    clustering = HDBSCANClusters(**request)
    clustering.run_clustering()


if __name__ == "__main__":
    main()
import os,sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import numpy as np

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

from utils.utils import get_input, get_crs_local, read_od
import ufo_map.Utils.helpers as ufo_helpers

CRS_UNI = 4326

class FindSubcenters():

    def __init__(self, name, epsilon, buffer_percentage, save_fig, save_cluster, sample_factor, path_root, day_hour):
        self.city_name = name
        self.epsilon = epsilon
        self.buffer_percentage = buffer_percentage 
        self.save_fig = save_fig
        self.save_cluster = save_cluster
        self.sample_factor = sample_factor
        self.crs_local = get_crs_local(self.city_name)
        self.path_root = path_root 
        self.day_hour = day_hour
        self.gdf = None
        self.gdf_local_cbd = None
        self.centermost_points = None
        self.gdf_cbd = None
       

    def load_data(self):
        print('Loading data in {}...'.format(self.city_name))
        self.gdf = read_od(self.path_root, self.city_name, self.crs_local, self.day_hour, 'destination')
        print('Loaded {} trips for {}h.'.format(len(self.gdf),self.day_hour))
        if self.sample_factor is not None:
            print('Sample {} trips from a total of {}.'.format(int(len(self.gdf)*self.sample_factor),len(self.gdf)))
            self.gdf = self.gdf.sample(n=int(len(self.gdf)*self.sample_factor)).reset_index(drop=True)
        else: self.sample_factor = 1 # set to 1.0 for saving 

    def load_additional_data(self):
        self.gdf_cbd = ufo_helpers.import_csv_w_wkt_to_gdf(os.path.join(self.path_root,'0_raw_data',self.city_name,'streets',
                                                            self.city_name+
                                                            '_cbd_gmaps.csv'),crs=self.crs_local)

    def _get_centermost_point(self,cluster):
        centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
        centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
        return tuple(centermost_point)  


    def dbscan_cluster(self): 
        print('Running dbscan...')        
        min_samples = int(len(self.gdf)*self.buffer_percentage)
        kms_per_radian = 6371.0088 # convert self.epsilonilon from km to radians

        self.gdf = self.gdf.to_crs(4326)
        self.gdf['lat'] = self.gdf.geometry.y
        self.gdf['lon'] = self.gdf.geometry.x
        coords = self.gdf[['lat','lon']].to_numpy()

        dbscan = DBSCAN(
            eps = self.epsilon/kms_per_radian,
            min_samples = min_samples,
            algorithm = 'ball_tree',
            metric = 'haversine')
        print('Fitting the model...')
        db = dbscan.fit(np.radians(coords)) # fit the algorithm

        cluster_labels = db.labels_
        self.gdf['cluster'] = cluster_labels 
        num_clusters = len(set(cluster_labels))
        clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
        print('Num clusters and trips per cluster:')
        print(clusters.str.len())

        clusters = clusters[clusters.str.len() !=0]
        self.centermost_points = clusters.map(self._get_centermost_point)
         
    
    def convert_cbd_points_to_gdf(self):
        print('Cretaing subcenter gdf...')
        df_tmp = pd.DataFrame([[*a] for a in self.centermost_points],columns=('x','y'))
        geometry = gpd.points_from_xy(df_tmp.y,df_tmp.x)
        self.gdf_local_cbd = gpd.GeoDataFrame(geometry = geometry, crs = CRS_UNI)
        self.gdf_local_cbd = self.gdf_local_cbd.to_crs(self.crs_local)   

    def save_local_cbd(self):
        self.gdf_local_cbd.to_csv(os.path.join(self.path_root,
                                        '0_raw_data',
                                        self.city_name,
                                        'streets',
                                        self.city_name+
                                        '_local_cbd_eps'+str(self.epsilon)+
                                        '_n'+str(self.sample_factor)+
                                        '_epsg'+self.gdf_local_cbd.crs.to_authority()[1]+
                                        '.csv'),
                                        index=False)    

        if self.save_cluster:
            self.gdf = self.gdf.to_crs(self.crs_local)
            self.gdf.to_csv(os.path.join(self.path_root,
                                        '6_analysis',
                                        'local_cbd',
                                        self.city_name+
                                        '_cluster_eps'+str(self.epsilon)+
                                        '_n'+str(self.sample_factor)+
                                        '_epsg'+self.gdf_local_cbd.crs.to_authority()[1]+
                                        '.csv'),index=False)

        if self.save_fig:
            self.load_additional_data()          
            
            fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(20,10))            
            plt.title('DBSCAN Clusters, subcenter centroids (orange) and main center from gmaps (red)')
            self.gdf.loc[self.gdf.cluster!=-1].plot(ax=ax1,column='cluster',alpha=0.01)
            self.gdf.loc[self.gdf.cluster==-1].plot(ax=ax1,color='black',alpha=0.005)
            self.gdf_local_cbd.plot(ax=ax1, color='orange', markersize=150)
            self.gdf_cbd.plot(ax=ax1, color='red',markersize=150)
            
            self.gdf[['cluster']].plot.hist(ax=ax2)
            
            plt.savefig(os.path.join(self.path_root,
                                        '6_analysis',
                                        'local_cbd',
                                        self.city_name+'_local_cbd_'+str(self.epsilon)+
                                        '_'+str(self.buffer_percentage)+'.png'), bbox_inches='tight')                                        
        print('Saved data. Closing run.')


    def process_subsenters(self):
        self.load_data()
        self.dbscan_cluster() 
        self.convert_cbd_points_to_gdf()
        self.save_local_cbd()
        

def main():
    
    request = get_input(PROJECT_SRC_PATH,'postprocessing/find_subcenters.yml')

    ps = FindSubcenters(**request)
    ps.process_subsenters()


if __name__ == "__main__":
    main() 

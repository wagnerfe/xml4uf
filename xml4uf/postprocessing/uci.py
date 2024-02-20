import os,sys
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

import utils.utils as utils
import utils.utils_ml as utils_ml
import utils.utils_h3 as uh3


FEATURE_COLS = {'pop_dense':'total_population',
                'employment':'num_jobs'}


class UCI():

    def __init__(self,
                name = None,
                path_root = None,
                hex_size = None,
                metric = None,
                boundary = None,
                ):

        self.city_name = name
        self.path_root = path_root
        self.hex_size = hex_size
        self.metric = metric
        self.boundary = boundary
        self.day_hour = 9


        self.dir_name = os.path.join(self.path_root,'6_analysis','uci')
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        self.file_name = 'h'+str(self.hex_size)+'_'+self.metric+'_'+self.boundary+'_'+self.city_name


    def venables(self,b, dist):
        v = np.dot(np.dot(b.T, dist), b)
        return v


    def location_coef(self,x):
        cl = (np.sum(np.abs(x - (1 / len(x))))) / 2
        return cl


    def calculate_uci(self,gdf, var_name):
        """
        in comparison to original R implementation,
        this version also considers a boundary in the form
        of a geopandas dataframe as input.
        """

        # change projection to UTM
        gdf = gdf.to_crs(epsg=3857)
        boundary = gdf.geometry.unary_union
        gdf_bound = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary),crs=3857)

        # normalize distribution of variable
        var_x = gdf[var_name].values
        var_x_norm = var_x / np.sum(var_x)  # normalization

        # calculate distance matrix
        coords = gdf.centroid.to_crs(epsg=3857)
        dist = coords.geometry.apply(lambda g: coords.distance(g))
        dist = dist.to_numpy()

        # self distance
        n_reg = dist.shape[0]
        poly_areas = gdf.geometry.area.values
        self_dist = np.diag((poly_areas / np.pi) ** (1 / 2))
        dist += self_dist  # Sum dist matrix and self-dist

        # UCI and its components
        LC = self.location_coef(var_x_norm)

        # Spatial separation index (venables)
        v = self.venables(var_x_norm,dist)

        # Determine polygons on border
        boundary = gdf_bound.geometry.boundary[0]
        gdf['border'] = gdf.intersects(boundary).astype(float)
        b = gdf['border'].values
        b[np.isnan(b)] = 0.0
        b[b == 1] = 1 / len(b[b == 1])

        # MAX spatial separation
        # with all activities equally distributed along the border    
        v_max = self.venables(b,dist)

        # Proximity Index PI
        proximity_index = 1 - (v / v_max)

        # UCI
        UCI = LC * proximity_index

        return {'UCI': UCI,
                'location_coef': LC,
                'spatial_separation': v,
                'spatial_separation_max': v_max,
                'proximity_index':proximity_index}


    def select_boundary(self, data, crs_local):
        if self.boundary is not None:
            return utils.read_bound(self.path_root,self.city_name,bound=self.boundary)
        else: 
            bound_geometry = data[self.city_name].geometry.unary_union
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries(bound_geometry),crs=crs_local)
        

    def convert_to_h3(self, data, crs_local):
        gdf_points = utils.read_feature_preproc(self.metric,self.path_root,self.city_name,crs_local) 
        gdf_bound = self.select_boundary(data, crs_local)        
        points_in_boundary = gpd.sjoin(gdf_points, gdf_bound).drop(columns='index_right').reset_index(drop=True)
        self.gdf_h3 = uh3.wrapper_conv_geom_h3_scaled(points_in_boundary,gdf_bound,FEATURE_COLS[self.metric],self.hex_size,crs_local).reset_index()


    def load_data(self):
        path_geoms = os.path.join(self.dir_name,self.file_name+'.gpkg')
        
        if os.path.isfile(path_geoms):
            print(f'Loading h{str(self.hex_size)} file')
            self.gdf_h3 = gpd.read_file(path_geoms)
        else:
            data,_,_ = utils_ml.load_light_cities_sample([self.city_name],path_root=self.path_root)
            crs_local = utils.get_crs_local(self.city_name)
            print(f'Converting to h{str(self.hex_size)}')
            self.convert_to_h3(data, crs_local)
            print('Saving h3 data...')
            self.gdf_h3.to_file(path_geoms)


    def save_index(self,uci_data):
        df = pd.DataFrame([uci_data])
        print(df)
        print('Saving UCI data...')
        df.to_csv(os.path.join(self.dir_name,self.file_name+'_uci.csv'))


    def get_index(self):
        self.load_data()
        uci_data = self.calculate_uci(self.gdf_h3,FEATURE_COLS[self.metric])
        self.save_index(uci_data)
    
    
def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'postprocessing/uci.yml')
    uci = UCI(**request)
    uci.get_index()

if __name__ == "__main__":
    main() 


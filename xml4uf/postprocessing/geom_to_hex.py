# imports
import sys, os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt 
import json
import h3
import yaml

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

# import modules
from ufo_map.Utils.helpers import *
from utils.utils import get_input, get_crs_local, write_hex_od, read_od_preproc_points
from utils.utils_mobility import get_df_x


CRS_UNI = 'epsg:4326'


class ConvertGeomToHex():

    def __init__(self,name,features,day_hour,path_root):
        self.city_name = name
        self.crs_local = get_crs_local(self.city_name)
        self.features = features
        self.day_hour = day_hour
        self.path_root = path_root    
        self.gdf_o = None
        self.gdf_d = None
        self.gdf_hex = None


    def get_data(self, features):
        df_od = read_od_preproc_points(self.path_root, self.city_name, self.crs_local, self.day_hour)
        if 'od_origin' in features:
            df_o = get_df_x(df_od,'origin','distance_m')
            self.gdf_o = gpd.GeoDataFrame(df_o,geometry=df_o['geometry'].apply(wkt.loads),crs=self.crs_local) 
        if 'od_destination' in features:
            df_d = get_df_x(df_od,'destination','distance_m')
            self.gdf_d = gpd.GeoDataFrame(df_d,geometry=df_o['geometry'].apply(wkt.loads),crs=self.crs_local) 
    
    
    def get_h3_points(self,gdf,aperture_size=11,dist_col='distance_m',id_o='id_origin',id_d='id_origin'): # TODO merge with utils_h3. convert_to_h3_points=!
        """
        Function that maps all trip points on a hex grid and normalises trip numbers per
        hexagon.
        Args:
            - gdf: dataframe with cleaned trip origin waypoints
            - dist_col: str that gives the name of dist col
            - id_o: str with name of origin col; f.e. 'id_origin'
            - id_d: str with name of destination col; f.e. 'id_origin'
            - aperture_size: hex raster; for more info see: https://h3geo.org/docs/core-library/restable/ 
        Returns:
            - gdf_out: geodataframe with hexagons containing the average trip lengths 
        """
        
        #convert crs to crs=4326
        gdf = gdf.to_crs(CRS_UNI)
        
        # 0. convert trip geometry to lat long
        gdf['lng']= gdf['geometry'].x
        gdf['lat']= gdf['geometry'].y

        # 0. find hexs containing the points
        gdf['hex_id'] = gdf.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,aperture_size),1)
        
        # 1. group all trips per hexagon and average tripdistancemters
        df_out = gdf.groupby('hex_id')[dist_col].mean().to_frame(dist_col)

        # 3. count number of trips per hex
        df_out['points_in_hex'] = gdf.groupby('hex_id').size().to_frame('cnt').cnt

        # 4. allocate 
        if id_d is not None:
            df_out['ids_origin'] = gdf.groupby('hex_id')[id_o].apply(list).to_frame('ids').ids
            df_out['ids_destination'] = gdf.groupby('hex_id')[id_d].apply(list).to_frame('ids').ids
        else:
            df_out['ids_origin'] = gdf.groupby('hex_id')[id_o].apply(list).to_frame('ids').ids

        # reset index of df_out; keep hex id as col
        df_out = df_out.reset_index()
        
        ## 4. Get center of hex to calculate new features
        df_out['lat'] = df_out['hex_id'].apply(lambda x: h3.h3_to_geo(x)[0])
        df_out['lng'] = df_out['hex_id'].apply(lambda x: h3.h3_to_geo(x)[1])   

        ## 5. Convert lat and long to geometry column 
        gdf_out = gpd.GeoDataFrame(df_out, geometry=gpd.points_from_xy(df_out.lng, df_out.lat),crs=4326)
        
        if id_d is not None: gdf_out = gdf_out[['hex_id',dist_col,'points_in_hex','ids_origin','ids_destination','geometry']]
        else: gdf_out = gdf_out[['hex_id',dist_col,'points_in_hex','ids_origin','geometry']]
        return gdf_out.to_crs(gdf.crs)


    def save_data(self):
        if 'od_origin' in self.features: 
            write_hex_od(self.gdf_hex, self.path_root, self.city_name, self.crs_local, self.day_hour, 'origin')
        if 'od_destination' in self.features: 
            write_hex_od(self.gdf_hex, self.path_root, self.city_name, self.crs_local, self.day_hour, 'destination')
    

    def convert_to_hex(self):
        self.get_data(self.features)
        print('assigning gdf geoms to hex...')
        if 'od_origin' in self.features:
            self.gdf_hex = self.get_h3_points(self.gdf_o)
        if 'od_destination' in self.features:
            self.gdf_hex = self.get_h3_points(self.gdf_d)

        self.save_data()
        print('All done. Closing run.')    


def main():

    request = get_input(PROJECT_SRC_PATH,'postprocessing/geom_to_hex.yml')

    cgh = ConvertGeomToHex(**request)
    cgh.convert_to_hex()

if __name__ == "__main__":
    main()
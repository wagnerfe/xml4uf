import sys, os
import pandas as pd
import geopandas as gpd
from shapely import wkt


# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)
print(PROJECT_SRC_PATH)

import utils.utils as utils
import utils.utils_mobility as utils_mobility


class CleanIdBounds():
    """
    Introduces unique id and cuts data at pre-defined bounds.

    In:
    - preprocessed data: 
        points: 
            ../1_preprocessed_data/../<CITY_NAME>_od_epsg<CRS_LOCAL>_t<DAY_HOUR>.csv
        polygons: 
            .../0_raw_data/../OD0...OD23.fma
            .../0_raw_data/../CITY_NAME.shp 
    
    Out:
    - geoms with unique id: 
        ../2_cleaned_data/../'<CITY_NAME>_od_<RESOLUTION>_unique_id_t<DAY_HOUR>.csv'
    - geoms with unique id and cut at bound:
        ../2_cleaned_data/../'<CITY_NAME>_od_<RESOLUTION>_id_t<DAY_HOUR>_<BOUND>.csv'
    """

    def __init__(
            self,
            name=None,
            path_root=None,
            day_hour=None,
            resolution=None,
            create_id=None,
            bound=None,
            ):
        
        self.city_name = name
        self.path_root = path_root
        self.crs_local = utils.get_crs_local(self.city_name)
        self.day_hour = day_hour
        self.resolution = resolution
        self.create_id = create_id
        self.bound = bound
        
        self.gdf = None
        self.gdf_o = None
        self.gdf_d = None
        self.gdf_bound = None
        self.id_col = 'id'
        self.path_out = None

    
    def initialize(self):
        self.path_out = os.path.join(self.path_root,'2_cleaned_data', self.city_name)
        utils.create_dir(self.path_out)


    def _read_hex(self):
        self.gdf_o = utils.read_od_preproc_hex_preproc(self.path_root,self.city_name,self.crs_local,self.day_hour,od_col='origin')
        self.gdf_d = utils.read_od_preproc_hex(self.path_root,self.city_name,self.crs_local,self.day_hour,od_col='destination') 


    def _read_points(self):
        self.df = utils.read_od_preproc_points(self.path_root, self.city_name, self.crs_local, self.day_hour)


    def _read_polygons(self):
        self.df = utils.read_od_preproc_polygons(self.path_root, self.city_name, self.day_hour)


    def _load_preprocessed_data(self):
        if self.resolution == 'hex': 
            self._read_hex()    
            self.id_col = 'hex_id'
        elif self.resolution == 'points': self._read_points()
        elif self.resolution == 'polygons': self._read_polygons()


    def _assign_geometries_points(self):
        df_o = utils_mobility.get_df_x(self.df,'origin','distance_m')
        df_d = utils_mobility.get_df_x(self.df,'destination','distance_m')
        self.gdf_o = gpd.GeoDataFrame(df_o,geometry=df_o['geometry'].apply(wkt.loads),crs=self.crs_local) 
        self.gdf_d = gpd.GeoDataFrame(df_d,geometry=df_d['geometry'].apply(wkt.loads),crs=self.crs_local) 


    def _assign_geometries_polygons(self):
        df_o = utils.merge_zip_geoms(self.df, self.path_root,self.city_name,self.crs_local,'id_origin')
        df_d = utils.merge_zip_geoms(self.df, self.path_root,self.city_name,self.crs_local,'id_destination')
        
        cols = ['id','p_combined','id_origin','id_destination','geometry']
        self.gdf_o = gpd.GeoDataFrame(df_o[cols],crs=self.crs_local)
        self.gdf_d = gpd.GeoDataFrame(df_d[cols],crs=self.crs_local)


    def _create_id(self):
        if self.resolution !='hex':
            self.df['id'] = utils.create_unique_id(self.df, self.resolution, self.city_name,self.day_hour)


    def _save_id_data(self):
        print('Saving id data to disk...')
        if self.resolution == 'points':
            self.df.to_csv(os.path.join(self.path_out,
                                    self.city_name+
                                    '_od_points_unique_id_t'+
                                    str(self.day_hour)+
                                    '.csv'),index=False)
        elif self.resolution == 'polygons':
            self.df.to_csv(os.path.join(self.path_out,
                                    self.city_name+
                                    '_od_polygons_unique_id_t'+
                                    str(self.day_hour)+
                                    '.csv'),index=False)
        elif self.resolution=='hex':
            pass # hex already has an id from hex system


    def _load_id_data(self):
        if self.resolution=='points':
            self.df = pd.read_csv(os.path.join(self.path_out,self.city_name+
                                            '_od_points_unique_id_t'+
                                            str(self.day_hour)+
                                            '.csv'))
            self.df = utils.id_to_str(self.df)
        elif self.resolution=='polygons':
            self.df = pd.read_csv(os.path.join(self.path_out,self.city_name+
                                            '_od_polygons_unique_id_t'+
                                            str(self.day_hour)+
                                            '.csv'))
            self.df = utils.id_to_str(self.df)
        elif self.resolution=='hex':
            self._read_hex(self)


    def load_data(self):
        if self.create_id:
            self._load_preprocessed_data()
            self._create_id()
            self._save_id_data()
        else:
            self._load_id_data()
        
        if self.resolution == 'points':
            self._assign_geometries_points()
        if self.resolution == 'polygons':
            self._assign_geometries_polygons()
        
        self.gdf_bound = utils.read_bound(self.path_root, self.city_name, bound=self.bound)
        
              
    def od_geoms_in_bounds(self):
        """
        Function to get all trips that start and end within given boundaries.
        Imported from urbanformvmt project: https://github.com/wagnerfe/xml4urbanformanalysis
        """
        print(f'N trips before cut at bound: O:{len(self.gdf_o)}, D:{len(self.gdf_d)}')
        print(f'Bound length {len(self.gdf_bound)}')
        print('---gdf_o:', self.gdf_o.head(2))
        print('---gdf_d:', self.gdf_d.head(2))

        gdf_in_o = gpd.sjoin(self.gdf_o,self.gdf_bound,how='inner',predicate='intersects')
        gdf_in_d = gpd.sjoin(self.gdf_d,self.gdf_bound,how='inner',predicate='intersects')
        gdf_in_o = gdf_in_o.drop(columns="index_right")
        gdf_in_d = gdf_in_d.drop(columns="index_right")
        print(f'N trips after cut at bound: O:{len(gdf_in_o)}, D:{len(gdf_in_d)}')

        gdf_in_d = gdf_in_d.rename(columns={"geometry": "geometry_destination"})
        gdf_in_d = gdf_in_d[[self.id_col,'geometry_destination']]
        gdf_in_o = gdf_in_o.rename(columns={"geometry": "geometry_origin"})
        self.gdf = gdf_in_o.merge(gdf_in_d,left_on = self.id_col, right_on = self.id_col)
        print(f'N trips in self.gdf {len(self.gdf)}')


    def remove_same_orig_dest_trips(self):
        # remove as distance from those trips is only noise
        self.gdf = self.gdf.loc[self.gdf['id_origin']!=self.gdf['id_destination']].reset_index(drop=True)


    def save_data(self):
        print('Saving to disk...')
        if self.resolution=='hex':
            self.gdf.to_csv(os.path.join(self.path_out,
                                    self.city_name+
                                    '_od_h11_epsg'+
                                    self.crs_local.split(':')[1]+                                        
                                    '_t'+str(self.day_hour)+
                                    '_'+self.bound+'.csv'),index=False)

        elif self.resolution == 'points':
            self.gdf.to_csv(os.path.join(self.path_out,
                                    self.city_name+
                                    '_od_points_epsg'+self.crs_local.split(':',1)[1]+
                                    '_t'+str(self.day_hour)+
                                    '_'+self.bound+'.csv'),index=False)
        elif self.resolution == 'polygons':
            self.gdf = self.gdf.drop(columns=['geometry_origin','geometry_destination'])
            self.gdf.to_csv(os.path.join(self.path_out,
                                    self.city_name+
                                    '_od_polygons_epsg'+self.crs_local.split(':',1)[1]+
                                    '_t'+str(self.day_hour)+
                                    '_'+self.bound+'.csv'),index=False)


    def select_resolutions(self):
        if self.resolution == 'polygons':
            return ['points','polygons']
        else: return [self.resolution]


    def clean_data(self):
        self.initialize()
        for self.resolution in self.select_resolutions():
            print(f'Cleaning data for {self.resolution}')
            self.load_data()        
            self.od_geoms_in_bounds()
            self.remove_same_orig_dest_trips
            self.save_data()        
        print('All done. Closing run.')
    

def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'mobility_cleaning/mobility_cleaning.yml')
    
    cb = CleanIdBounds(**request)
    cb.clean_data()


if __name__ == "__main__":
    main()

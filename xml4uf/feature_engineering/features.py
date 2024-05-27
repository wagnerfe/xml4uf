import os,sys
import pandas as pd
import numpy as np
from pathlib import Path

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

import utils.utils as utils
from ufo_map.Feature_engineering.city_level import shortest_distance_graph,city_area_km2, network_length_km, land_use_entropy
from ufo_map.Feature_engineering.socio_econ import feature_in_buffer, trips_per_capita, employment_access, employment_access_v2
from ufo_map.Feature_engineering.streets import ft_intersections_per_buffer

FT_POP_DENSE = 'total_population'
FT_INCOME = 'mean_income'
FT_INCOME_GROUPS = 'income_groups'
FT_JOB_DENSE = 'num_jobs'
FT_TRANSIT_ACCESS = 'score_spatiotemporal'


class Features():
    '''
    Calculate features per geometry. 

    In: 
    - mobility data:
        for points: 
            ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_od_points_epsg<CRS_LOCAL>_<BOUNDARY>.csv
        for polygons:
            ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_od_points_epsg<CRS_LOCAL>_<BOUNDARY>.csv
            ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_od_polygons_epsg<CRS_LOCAL>_<BOUNDARY>.csv
    - preprocessed feature data: 
        ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_<FEATURE_TYPE>_epsg<CRS_LOCAL>.csv
    Out:
    - calculated feature data per bound geometry (when testing we add "test_" as a prefix):
        ../3_features/<CITY_NAME>/<FEATURE_NAME>_<BOUND>.csv
    '''
    def __init__(
            self,
            name = None,
            path_root = None,
            bound = None,
            target = None,
            od_col = None,
            target_time = None,
            features = None,
            testing = None,
            ):
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 
        self.bound = bound
        self.target = target
        self.od_col = od_col
        self.target_time = target_time
        self.features = features
        self.testing = testing

        self.path_out = None
        self.gdf = None
        self.g = None
        self.streets = None
        self.gdf_cbd = None
        self.gdf_local_cbd = None
        self.gdf_pop_dense = None
        self.gdf_pop_dense_meta = None
        self.gdf_income = None
        self.gdf_airports = None
        self.gdf_transit_access = None
        self.df_lu = None

    
    def prepare_target(self):
        df_trips = pd.DataFrame()
        for day_hour in self.target_time:
            print(f"reading in {day_hour} to {day_hour+1} o'clock")
            df_tmp = utils.read_od(self.path_root,
                                self.city_name,
                                self.crs_local,
                                day_hour,
                                self.od_col,
                                resolution='points',
                                bound=self.bound)
            df_trips = pd.concat([df_trips,df_tmp])

        if self.target == 'num_trips':
            df_trips = df_trips.groupby('id_'+self.od_col).size().reset_index(name='num_trips')
        else:
            df_trips = df_trips.groupby('id_'+self.od_col)['distance_m'].mean().to_frame().reset_index()

        df_trips['tractid'] = self.city_name +'-'+ df_trips['id_origin']
        return df_trips[[self.id_col, self.target]]
    

    def calculate_target(self):
        df = self.prepare_target()
        file_name = utils.get_file_name(self.target,self.bound, self.target_time, self.od_col ,self.testing)
        df.to_csv(os.path.join(self.path_out,file_name),index=False)

        
    def _load_feature_data(self,required_data):
        if ('g' in required_data) and (self.g is None):
            self.g = utils.read_graph(self.path_root, self.city_name, 'full')
        if ('gdf_cbd' in required_data): # load again even if self.gdf_cbd is not None for sfo ft_dist_cbd4
            self.gdf_cbd = utils.read_cbd(self.path_root, self.city_name, self.crs_local)
        if ('ft_dist_local_cbd' in required_data[0]): # TODO adjust hack when final data found
            self.gdf_local_cbd = utils.read_local_cbd(required_data[0],self.path_root, self.city_name, self.crs_local)
        if ('pop_dense' in required_data) and (self.gdf_pop_dense is None):
            self.gdf_pop_dense = utils.read_feature_preproc('pop_dense',self.path_root,self.city_name,self.crs_local) 
        if ('pop_dense_meta' in required_data) and (self.gdf_pop_dense_meta is None):
            self.gdf_pop_dense_meta = utils.read_feature_preproc('pop_dense_meta',self.path_root,self.city_name,self.crs_local) 
        if ('income' in required_data) and (self.gdf_income is None):
            self.gdf_income = utils.read_feature_preproc('income',self.path_root,self.city_name,self.crs_local) 
        if ('income_groups' in required_data) and (self.gdf_income is None):
            self.gdf_income_groups = utils.read_feature_preproc('income_groups',self.path_root,self.city_name,self.crs_local) 
        if ('income_groups3' in required_data) and (self.gdf_income is None):
            self.gdf_income_groups = utils.read_feature_preproc('income_groups3',self.path_root,self.city_name,self.crs_local) 
        if ('gdf_airports' in required_data) and (self.gdf_airports is None):
            self.gdf_airports = utils.read_feature_preproc('airports',self.path_root,self.city_name,self.crs_local) 
        if ('employment' in required_data):
            self.gdf_employment = utils.read_feature_preproc('employment',self.path_root,self.city_name,self.crs_local) 
        if ('transit_access' in required_data):
            self.gdf_transit_access = utils.read_feature_preproc('transit_access',self.path_root,self.city_name,self.crs_local) 
        if ('land_use' in required_data):
            self.df_lu = utils.read_feature_preproc('land_use',self.path_root,self.city_name,self.crs_local) 

        

    def _save_feature(self,df, feature_name):
        file_name = utils.get_file_name(feature_name,self.bound, self.testing)
        df = df.rename(columns={feature_name:feature_name})
        
        print(f'Saves {feature_name} at {len(df)} zips:')
        print(df.head(2))
        df.to_csv(os.path.join(self.path_out,file_name),index=False)


    def initialize_run(self):
        self.id_col = 'tractid'
        self.path_out = os.path.join(self.path_root,'3_features',self.city_name)
        utils.create_dir(self.path_out)
        self.gdf = utils.init_geoms(self.path_root,
                                    self.city_name,
                                    self.bound)
        
        print(self.gdf.head(2))
        print(f'len gdf: {len(self.gdf)}')
        

    def calculate_features(self,feature_name):
        if feature_name == 'ft_dist_cbd':
            self._load_feature_data(['gdf_cbd','g'])
            # take only downtown sfo as center like yanyan et al. (2023) Urban Dynamics Through the Lens of Human Mobility
            if self.city_name =='sfo':
                self.gdf_cbd = self.gdf_cbd.iloc[[3]].reset_index(drop=True) 
            df_tmp = shortest_distance_graph(self.gdf, self.gdf_cbd, self.g, feature_name)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_dist_cbd4':
            self._load_feature_data(['gdf_cbd','g'])
            df_tmp = shortest_distance_graph(self.gdf, self.gdf_cbd, self.g, feature_name)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)
        
        if 'ft_dist_local_cbd' in feature_name:
            self._load_feature_data([feature_name,'g'])
            df_tmp = shortest_distance_graph(self.gdf,self.gdf_local_cbd,self.g,feature_name)[[self.id_col,feature_name]] 
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_pop_dense':
            self._load_feature_data(['pop_dense'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_pop_dense, FT_POP_DENSE, feature_name,
                                        id_col=self.id_col,feature_type='total_per_area')[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)
        
        if feature_name == 'ft_pop_total':
            self._load_feature_data(['pop_dense'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_pop_dense, FT_POP_DENSE, feature_name,
                                        id_col=self.id_col,feature_type='total')[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_job_dense':
            self._load_feature_data(['employment'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_employment, FT_JOB_DENSE, feature_name,
                                        id_col=self.id_col,feature_type='total_per_area')[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_jobs_total':
            self._load_feature_data(['employment'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_employment, FT_JOB_DENSE, feature_name,
                                        id_col=self.id_col,feature_type='total')[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_pop_dense_meta':
            self._load_feature_data(['pop_dense_meta'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_pop_dense_meta, FT_POP_DENSE, feature_name,
                                    id_col=self.id_col,feature_type='total_per_area')[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)
        
        if feature_name == 'ft_pop_dense_meta_total':
            self._load_feature_data(['pop_dense_meta'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_pop_dense_meta, FT_POP_DENSE, feature_name,
                                    id_col=self.id_col,feature_type='total')[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_income':
            self._load_feature_data(['income'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_income, FT_INCOME,feature_name,
                                   id_col=self.id_col)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)   

        if feature_name == 'ft_income_groups':
            self._load_feature_data(['income_groups'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_income_groups, FT_INCOME_GROUPS,feature_name,
                                   id_col=self.id_col)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_income_groups3':
            self._load_feature_data(['income_groups3'])
            df_tmp = feature_in_buffer(self.gdf, self.gdf_income_groups, FT_INCOME_GROUPS,feature_name,
                                   id_col=self.id_col)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_beta':
            self._load_feature_data(['g'])
            df_tmp = ft_intersections_per_buffer(self.gdf,self.g,feature_name,
                                                id_col=self.id_col)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)
        
        if feature_name == 'ft_dist_airport':
            self._load_feature_data(['gdf_airports','g'])
            df_tmp = shortest_distance_graph(self.gdf,self.gdf_airports,self.g,feature_name)[[self.id_col,feature_name]] 
            self._save_feature(df_tmp,feature_name)

        if 'ft_employment_access' in feature_name:
            self._load_feature_data(['employment'])
            for threshold in [0.01, 0.05, 0.1, 0.2]:
                feature_name_tmp = feature_name+'_'+str(threshold).split('.')[1]
                df_tmp = employment_access(self.gdf, self.gdf_employment, feature_name_tmp, threshold, self.id_col)[[self.id_col,feature_name_tmp]]
                self._save_feature(df_tmp,feature_name_tmp)
        
        if 'ft_employment_access_v2' in feature_name:
            self._load_feature_data(['employment'])
            for threshold in [0.01, 0.05]:
                feature_name_tmp = feature_name+'_'+str(threshold).split('.')[1]
                df_tmp = employment_access_v2(self.gdf, self.gdf_employment, feature_name_tmp, threshold, self.id_col)[[self.id_col,feature_name_tmp]]
                self._save_feature(df_tmp,feature_name_tmp)

        if feature_name == 'ft_city_area_km2': 
            df_tmp = city_area_km2(self.gdf,feature_name)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_network_length_km': 
            self._load_feature_data(['g'])
            df_tmp = network_length_km(self.gdf,self.g,feature_name)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_transit_access':
            self._load_feature_data(['transit_access'])
            df_tmp = pd.merge(self.gdf[['tractid']], self.gdf_transit_access[['tractid',FT_TRANSIT_ACCESS]], on='tractid',how='left')
            self._save_feature(df_tmp,feature_name)
        
        if feature_name == 'ft_lu_entropy_classic':
            self._load_feature_data(['land_use', 'pop_dense_meta'])
            df_tmp = land_use_entropy(self.gdf,self.df_lu,entropy_type = 'classic', gdf_pop = self.gdf_pop_dense_meta)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)

        if feature_name == 'ft_lu_entropy_normed':    
            self._load_feature_data(['land_use', 'pop_dense_meta'])
            df_tmp = land_use_entropy(self.gdf,self.df_lu,entropy_type = 'normed', gdf_pop = self.gdf_pop_dense_meta)[[self.id_col,feature_name]]
            self._save_feature(df_tmp,feature_name)


    def create_feature_df(self):
        self.initialize_run()
        
        if self.target:
            self.calculate_target()
        
        if self.features is not None:
            for feature in self.features:
                print('Calculating {}...'.format(feature))
                self.calculate_features(feature)
        print('Calculated all features. Closing run.')


def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'feature_engineering/features.yml')
    ft = Features(**request)
    ft.create_feature_df()

if __name__ == "__main__":
    main() 

import sys, os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from random import random
import shapely
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import pickle
import yaml
import json
import h3
import jenkspy


# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

# imports
from ufo_map.Utils.helpers import *
import utils.utils as utils 
import utils.utils as utils_h3

# define constants
CRS_UNI = 'epsg:4326'
BUFFSIZE = 1000 # m buffer around boundary of city
HEX_SIZE = 11
FT_POP_DENSE = 'total_population'
FT_INCOME = 'mean_income'
FT_HH_INCOME = 'mean_hh_income'
FT_INCOME_GROUPS = 'income_groups'
FT_EMPLOYMENT = 'num_jobs'


class PopDense():

    def __init__(
            self,
            name = None,
            path_root = None
            ):

        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 

        self.gdf_pop = None


    def _preproc_pop_dense_ber(self):
        path_file = os.path.join(self.path_root,'0_raw_data','ber/pop_dense/gdf_pop_dens_berlin_2018.csv')
        gdf = import_csv_w_wkt_to_gdf(path_file, crs=self.crs_local, geometry_col='geometry')
        gdf = gdf.rename(columns={'TOT_P_2018':FT_POP_DENSE})
        gdf_pop = gdf[[FT_POP_DENSE,'geometry']]

        print('Merging Berlin local data with meta data...')
        path_file_meta = os.path.join(self.path_root,'1_preprocessed_data/ber/ber_pop_dense_meta_epsg'+str(gdf.crs.to_epsg())+'.csv')
        if os.path.isfile(path_file_meta):
            gdf_pop_meta = utils.import_csv_w_wkt_to_gdf(path_file_meta, crs=self.crs_local)
            
            # get multiploygon bound of berlin pop density data 
            print('Filling holes in local file with 0s...')
            m = gdf_pop.geometry.unary_union
            no_holes = MultiPolygon(Polygon(p.exterior) for p in m.geoms)
            no_holes_bound = gpd.GeoDataFrame(geometry=gpd.GeoSeries(no_holes),crs=self.crs_local)

            # no holes bound is used to fill in holes and to add meta data outside of it
            no_holes_bound_4326 = no_holes_bound.to_crs(4326)
            gdf_h3 = no_holes_bound_4326.h3.polyfill_resample(HEX_SIZE).reset_index().rename(columns={'h3_polyfill':'h3'}).to_crs(self.crs_local)
            # add find geoms of holes, add 0 polygons to gdf_pop
            gdf_sjoin = gpd.sjoin(gdf_h3, gdf_pop,how='left')
            gdf_holes = gdf_sjoin.loc[gdf_sjoin.index_right.isna()]
            gdf_holes['total_population']=0.0
            gdf_pop_no_holes = pd.concat([gdf_pop,gdf_holes[['total_population','geometry']]]).reset_index(drop=True)

            # add meta data outside of no holes bound
            print('Adding meta data at outskirts...')
            gdf_sjoin_meta = gpd.sjoin(gdf_pop_meta,no_holes_bound, how='left')
            gdf_add_meta = gdf_sjoin_meta.loc[gdf_sjoin_meta.index_right.isna()]
            self.gdf_pop = pd.concat([gdf_pop_no_holes,gdf_add_meta[['total_population','geometry']]]).reset_index(drop=True)
        else:
            print('Warning! No data from meta found in 1_preprocessed_data/ber to fill outskirts of Berlin. Continuing with local Berlin data... ')
            self.gdf_pop = gdf_pop


    def _preproc_pop_dense_rio(self):
        path_file = os.path.join(self.path_root,'0_raw_data/rio/aop/hex_metro_rio.gpkg')
        gdf = gpd.read_file(path_file)
        gdf = gdf[['id_hex','pop_total','geometry']]
        gdf = gdf.rename(columns={'id_hex':'hex_id', 'pop_total':FT_POP_DENSE}) 
        self.gdf_pop = gdf.to_crs(self.crs_local)


    def _preproc_pop_dense_bog(self):
        path_dir= os.path.join(self.path_root,'0_raw_data/bog/pop_dense')
        # df = pd.read_csv(os.path.join(path_dir,'feature_data.csv'))
        # gdf_zip = gpd.read_file(os.path.join(path_dir,'BOG.shp')).to_crs(self.crs_local)
        # gdf_pop = pd.merge(gdf_zip[['tractid','geometry']], df[['tractid','population']], on='tractid')
        gdf_pop = gpd.read_file(os.path.join(path_dir,'poblacion-upz-bogota.shp'))
        gdf_pop = gdf_pop.to_crs(self.crs_local)
        self.gdf_pop = gdf_pop[['poblacion_u','geometry']].rename(columns={'poblacion_u':FT_POP_DENSE})
    

    def _preproc_pop_dense_us(self):
        path_dir= os.path.join(self.path_root,'0_raw_data',self.city_name) 
        if self.city_name =='bos':
            file_name_blocks = 'cb_2013_25_bg_500k.shp'
            file_name_data = 'ACSDT5Y2013.B01003-Data.csv'
        else:
            file_name_blocks = 'cb_2013_06_bg_500k.shp'
            file_name_data = 'ACSDT5Y2013.B01003-Data.csv'

        gdf_zip = gpd.read_file(os.path.join(path_dir,'income', file_name_blocks)).to_crs(self.crs_local)
        df = pd.read_csv(os.path.join(path_dir,'pop_dense',file_name_data))
        gdf_pop = pd.merge(gdf_zip[['AFFGEOID','geometry']],df,left_on='AFFGEOID',right_on='GEO_ID')
        gdf_pop = gdf_pop.rename(columns={'B01003_001E':FT_POP_DENSE})
        self.gdf_pop = gdf_pop[[FT_POP_DENSE,'geometry']]


    def preproc(self):
        print('Preprocessing pop dense...')
        if self.city_name in ['ber','ber_inrix']:
            self._preproc_pop_dense_ber()
        elif self.city_name == 'rio':
            self._preproc_pop_dense_rio()
        elif self.city_name == 'bog':
            self._preproc_pop_dense_bog()
        elif self.city_name in ['bos','sfo','lax']:
            self._preproc_pop_dense_us()
        return self.gdf_pop



class PopDenseMeta():
    # Warning! Bogota is not yet supported
    def __init__(
        self,
        name = None,
        path_root = None,
        ):

        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 

        self.gdf_pop = None
        self.gdf_bound = None
        if self.city_name in ['bos','lax','sfo','ber','ber_inrix']:
            self.lat_col = 'Lat'
            self.lon_col = 'Lon'
            self.pop_col = 'Population'
        elif self.city_name in ['rio']:
            self.lat_col = 'latitude'
            self.lon_col = 'longitude'
            self.pop_col = 'population_2020'
        elif self.city_name in ['lis']:
            self.lat_col = 'latitude'
            self.lon_col = 'longitude'
            self.pop_col = 'prt_general_2020'
        elif self.city_name == 'bog':
            self.lat_col = 'latitude'
            self.lon_col = 'longitude'
            self.pop_col = 'col_general_2020'


    def get_bound(self):
        # returns bound of a city incl. buffer of buff_size: BUFFSIZE
        gdf_bound = utils.read_bound(self.path_root,self.city_name, bound ='fua')
        # add buffer around bound
        gdf_buff_tmp = gdf_bound.buffer(5000)
        gdf_buff_tmp = gdf_buff_tmp.to_crs(CRS_UNI)
        self.gdf_buff = gpd.GeoDataFrame(geometry=gdf_buff_tmp)


   # get all relevant parquet paths of downlaoded dir
    def _get_paths(self):
        if self.city_name in ['lax','sfo']:
            self.path_pop = os.path.join(self.path_root,'0_raw_data','bos','pop_dense/pop_dense_meta/')
        else:
            self.path_pop = os.path.join(self.path_root,'0_raw_data',self.city_name,'pop_dense/pop_dense_meta/')

        all_paths = os.listdir(self.path_pop)
        # return only .csvs
        return [p for p in all_paths if p[-3:]=='csv']
    

    def _transform_to_h3(self):
        # find hexs containing the points
        self.gdf_pop['hex_id'] = self.gdf_pop.apply(lambda x: h3.geo_to_h3(x[self.lat_col],x[self.lon_col],HEX_SIZE),1)
        # group all trips per hexagon and average
        df_hex = self.gdf_pop.groupby('hex_id')[self.pop_col].mean().to_frame(FT_POP_DENSE).reset_index()
        self.gdf_pop = utils_h3.add_h3_geom(df_hex,self.crs_local,poly=True)


    def _clean_output(self):
        self.gdf_pop = self.gdf_pop[[FT_POP_DENSE,'geometry']]


    def preproc(self):
        print('Preprocessing population density meta...')
        all_paths = self._get_paths()
        self.get_bound()
        self.gdf_pop = gpd.GeoDataFrame()                                                  
        
        # Loop over all csv files of a country  
        for i,path in enumerate(all_paths):
            print('loading path {} of {}...'.format(i+1,len(all_paths)))
            gdf_out_tmp = gpd.GeoDataFrame()
            chunks = pd.read_csv(os.path.join(self.path_pop,path),chunksize=int(1E6))
            for idx, chunk in enumerate(chunks):
                print('looping through chunk {}'.format(idx))
                gdf_tmp = gpd.GeoDataFrame(chunk,geometry = gpd.points_from_xy(chunk[self.lon_col], chunk[self.lat_col]), crs=CRS_UNI)
                gdf_sjoin = gdf_tmp.sjoin(self.gdf_buff)
                gdf_out_tmp = pd.concat([gdf_out_tmp,gdf_sjoin])
            
            self.gdf_pop = pd.concat([self.gdf_pop,gdf_out_tmp])
        
        self.gdf_pop = self.gdf_pop.reset_index(drop=True)
        self._transform_to_h3()
        self._clean_output()
        return self.gdf_pop



class Income():
    
    def __init__(
            self,
            name = None,
            path_root = None,
            ):
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 

        self.df = None
        self.gdf_plz = None
        self.gdf_income = None


    def _ber_read_raw_files(self):    
        df_h = pd.read_csv(os.path.join(self.path_root,'0_raw_data',self.city_name,'income','hh_status_PLZ_2019.csv'),header=None, sep='\t')
        df_h = df_h[0].str.split(';', expand=True)
        col_names = df_h.iloc[0].values
        self.df = df_h.iloc[1:len(df_h)].copy() 
        self.df.columns = col_names
        self.gdf_plz = gpd.read_file(os.path.join(self.path_root,'0_raw_data',self.city_name,'mobility','zip_grid','plz-5stellig.shp'))


    def _ber_clean_df(self):
        # replace non ints in ph_to col
        self.df['ph_to'].str.contains(',')
        self.df.loc[self.df.ph_to.str.contains(',')]
        self.df =  self.df.replace(",", ".", regex=True)
        self.df = self.df.astype(float)
        self.df['plz']= self.df['PLZ'].astype(int)
        self.gdf_plz['plz'] = self.gdf_plz['plz'].astype(int)


    def _ber_calculate_weighted_mean(self):
        df_stat = self.df[[col for col in self.df.columns if 'stat' in col]]
        df_stat.columns = range(1,1+len(df_stat.columns))
        df_data = pd.DataFrame()
        for col in df_stat:
            df_data['mean_{}'.format(col)] = (df_stat[col]*col)

        self.df['mean'] =df_data[[col for col in df_data.columns if 'mean' in col]].sum(axis=1) 
        # get weigthed average
        self.df['weigthed_mean'] = self.df['mean']/self.df['ph_to']
        # assign income values for whole Germany
        self.gdf_income = pd.merge(self.gdf_plz[['plz','geometry','einwohner','note']],self.df,on='plz')


    def _ber_sjoin_bound(self):
        gdf_bound = utils.read_bound(self.path_root,self.city_name)        
        self.gdf_income = self.gdf_income.to_crs(self.crs_local)
        self.gdf_income = gpd.sjoin(self.gdf_income,gdf_bound)
        self.gdf_income = self.gdf_income.rename(columns={'weigthed_mean':FT_INCOME})
        self.gdf_income = self.gdf_income[['plz','einwohner',FT_INCOME,'geometry']]


    def _ber_preproc_income(self):
        self._ber_read_raw_files()
        self._ber_clean_df()
        self._ber_calculate_weighted_mean()
        self._ber_sjoin_bound()

    
    def _us_scale_income(self,df):
        df = df.iloc[1:]
        df_ = df[[col for col in df.columns if col[-1]=='E']].drop(columns='NAME')
        df_ = df_.astype(int)

        income_dict = {1:5.0, 2:12.5, 3:17.5, 4:22.5, 5:27.5, 6:32.5, 7:37.5, 8:42.5, 9:47.5, 10: 55.0, 11: 67.5, 12: 87.5, 13: 112.5, 14: 137.5, 15:175.0, 16:200.0}
        list_cols = df_.columns.to_list()
        col_items = {v: k for v, k in enumerate(list_cols[1:])}
        col_income = dict((value, income_dict[key+1]) for (key, value) in col_items.items())

        df_['sum_income'] = df_[[col for col in df_.columns if col in col_income.keys()]].mul(col_income).sum(axis=1)
        df_[FT_INCOME] = df_['sum_income']/df_['B19001_001E']
        return pd.merge(df[['GEO_ID','NAME']],df_, left_index=True, right_index=True)[['GEO_ID','NAME',FT_INCOME]].reset_index(drop=True)


    def _us_preproc_income(self):
        if self.city_name == 'bos':
            path_blocks = os.path.join(self.path_root,
                                '0_raw_data/bos/income/cb_2013_25_bg_500k.shp') 
            path_income_agg = os.path.join(self.path_root,
                                    '0_raw_data/bos/income/ACSDT5Y2013.B19001-Data.csv')
        elif self.city_name == 'lax':
            path_blocks = os.path.join(self.path_root,
                                '0_raw_data/lax/income/cb_2013_06_bg_500k.shp') 
            path_income_agg = os.path.join(self.path_root,
                                    '0_raw_data/lax/income/ACSDT5Y2013.B19001-Data.csv')            
        elif self.city_name == 'sfo':
            path_blocks = os.path.join(self.path_root,
                                '0_raw_data/sfo/income/cb_2013_06_bg_500k.shp') 
            path_income_agg = os.path.join(self.path_root,
                                    '0_raw_data/sfo/income/ACSDT5Y2013.B19001-Data.csv')            
            
        gdf_blocks = gpd.read_file(path_blocks)
        df_income_raw = pd.read_csv(path_income_agg)
        
        df_income = self._us_scale_income(df_income_raw)
        self.gdf_income = pd.merge(gdf_blocks[['AFFGEOID','geometry']],df_income,left_on='AFFGEOID',right_on='GEO_ID')
        self.gdf_income = self.gdf_income.to_crs(self.crs_local)


    def _rio_preproc_income(self):
        path_rio_income= os.path.join(self.path_root,'0_raw_data', self.city_name,'aop','hex_metro_rio.gpkg')
        gdf_rio = gpd.read_file(path_rio_income)
        gdf_rio = gdf_rio[['id_hex','income_per_capita','income_decile', 'income_quintile','geometry']]
        gdf_rio = gdf_rio.rename(columns={'id_hex':'hex_id', 'income_per_capita':FT_INCOME}) 
        self.gdf_income = gdf_rio.to_crs(self.crs_local)


    def _bog_preproc_income(self):
        path_data = os.path.join(self.path_root,'0_raw_data/bog/mobility')
        df_income = pd.read_csv(os.path.join(path_data,'feature_data.csv'))
        gdf_zips = gpd.read_file(os.path.join(path_data,'BOG.shp'))
        gdf_income = pd.merge(gdf_zips,df_income,on='tractid')
        gdf_income = gdf_income.rename(columns={'median_income':FT_INCOME})
        self.gdf_income = gdf_income.to_crs(self.crs_local)

    
    def preproc(self):
        print('Preprocessing income...')
        if self.city_name in ['ber','ber_inrix']:
            self._ber_preproc_income()
        elif self.city_name in ['bos','lax','sfo']:
            self._us_preproc_income()
        elif self.city_name == 'rio':
            self._rio_preproc_income()
        elif self.city_name=='bog':
            self._bog_preproc_income()
        return self.gdf_income



class IncomeBrackets():
    
    def __init__(
            self,
            name = None,
            path_root = None,
            feature = None
            ):
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 
        self.feature = feature

        self.df = None
        self.gdf_plz = None
        self.gdf_income_groups = None
        self.num_brackets = None


    def apply_jenks_breaks(self):
        self.gdf_income_groups = self.gdf_income_groups.dropna(subset=[FT_HH_INCOME])
        breaks = jenkspy.jenks_breaks(self.gdf_income_groups[FT_HH_INCOME], n_classes=self.num_brackets) 
        self.gdf_income_groups[FT_INCOME_GROUPS] = pd.cut(self.gdf_income_groups[FT_HH_INCOME],
                                                        bins=breaks,
                                                        labels=list(range(self.num_brackets)),
                                                        include_lowest=True).astype(int)

    def _ber_load_geoms(self):
        return gpd.read_file(os.path.join(self.path_root,'0_raw_data/ber/mobility/zip_grid/plz-5stellig.shp'))

    def _ber_load_city_survey(self):
        path_city_survey = os.path.join(self.path_root,'0_raw_data/ber/employment/mobility-survey-Berlin.csv')
        return pd.read_csv(path_city_survey)
    
    def _ber_load_outskirt_data(self):
        path_outskirt_data = os.path.join(self.path_root,'0_raw_data/ber/income/Berlin-Umland-Gemeinden.csv')
        gdf = utils.import_csv_w_wkt_to_gdf(path_outskirt_data,crs=4326)
        gdf = gdf[['OBJECTID','Income','geometry']].rename(columns={'OBJECTID':'plz','Income':FT_HH_INCOME})
        return gdf.to_crs(utils.CRS_BER)
    
    def _ber_add_geoms_to_survey(self,survey,gdf_zip):
        gdf_zip['plz'] = gdf_zip.plz.astype(int)
        gdf_zip = gdf_zip.loc[gdf_zip.plz.isin(survey.Postcode)]
        return pd.merge(gdf_zip[['plz','geometry']],survey,right_on='Des_Plz',left_on='plz')
    
    def _ber_clean_append_zips(self,gdf_income):
        gdf_income = gdf_income.drop_duplicates('HHNR') # drop duplictae households
        gdf_income.loc[gdf_income.IncomeDetailed=='Over5600', 'IncomeDetailed_Numeric']=6000
        df_tmp = gdf_income.groupby('plz')['IncomeDetailed_Numeric'].mean().to_frame(FT_HH_INCOME).reset_index()
        gdf_mean_income = gdf_income[['plz','geometry']].drop_duplicates().reset_index(drop=True)
        return pd.merge(gdf_mean_income, df_tmp).to_crs(utils.CRS_BER)
    
    def _ber_preproc_income_groups(self):    
        gdf_zip = self._ber_load_geoms()
        city_income = self._ber_load_city_survey()
        gdf_city_income = self._ber_add_geoms_to_survey(city_income,gdf_zip)
        gdf_city_income = self._ber_clean_append_zips(gdf_city_income)
        gdf_outskirt_income = self._ber_load_outskirt_data()
        self.gdf_income_groups = pd.concat([gdf_city_income, gdf_outskirt_income],axis=0)
        self.apply_jenks_breaks()


    def _us_scale_income(self,df):
        df = df.iloc[1:]
        df_ = df[[col for col in df.columns if col[-1]=='E']].drop(columns='NAME')
        df_ = df_.astype(int)

        income_dict = {1:5.0, 2:12.5, 3:17.5, 4:22.5, 5:27.5, 6:32.5, 7:37.5, 8:42.5, 9:47.5, 10: 55.0, 11: 67.5, 12: 87.5, 13: 112.5, 14: 137.5, 15:175.0, 16:200.0}
        list_cols = df_.columns.to_list()
        col_items = {v: k for v, k in enumerate(list_cols[1:])}
        col_income = dict((value, income_dict[key+1]) for (key, value) in col_items.items())

        df_['sum_income'] = df_[[col for col in df_.columns if col in col_income.keys()]].mul(col_income).sum(axis=1)
        df_[FT_HH_INCOME] = df_['sum_income']/df_['B19001_001E']
        return pd.merge(df[['GEO_ID','NAME']],df_, left_index=True, right_index=True)[['GEO_ID','NAME',FT_HH_INCOME]].reset_index(drop=True)


    def _us_preproc_income_groups(self):
        if self.city_name == 'bos':
            path_blocks = os.path.join(self.path_root,
                                '0_raw_data/bos/income/cb_2013_25_bg_500k.shp') 
            path_income_agg = os.path.join(self.path_root,
                                    '0_raw_data/bos/income/ACSDT5Y2013.B19001-Data.csv')
        elif self.city_name == 'lax':
            path_blocks = os.path.join(self.path_root,
                                '0_raw_data/lax/income/cb_2013_06_bg_500k.shp') 
            path_income_agg = os.path.join(self.path_root,
                                    '0_raw_data/lax/income/ACSDT5Y2013.B19001-Data.csv')            
        elif self.city_name == 'sfo':
            path_blocks = os.path.join(self.path_root,
                                '0_raw_data/sfo/income/cb_2013_06_bg_500k.shp') 
            path_income_agg = os.path.join(self.path_root,
                                    '0_raw_data/sfo/income/ACSDT5Y2013.B19001-Data.csv')            
            
        gdf_blocks = gpd.read_file(path_blocks)
        df_income_raw = pd.read_csv(path_income_agg)
        
        df_income = self._us_scale_income(df_income_raw)
        self.gdf_income_groups = pd.merge(gdf_blocks[['AFFGEOID','geometry']],df_income,left_on='AFFGEOID',right_on='GEO_ID')
        self.gdf_income_groups = self.gdf_income_groups.to_crs(self.crs_local)
        self.apply_jenks_breaks()

    
    def _rio_preproc_income_groups(self):
        path_rio_income= os.path.join(self.path_root,'0_raw_data', self.city_name,'aop','hex_metro_rio.gpkg')
        gdf_rio = gpd.read_file(path_rio_income)
        gdf_rio = gdf_rio[['id_hex','income_per_capita','income_decile', 'income_quintile','geometry']]
        gdf_rio = gdf_rio.rename(columns={'id_hex':'hex_id', 'income_per_capita':FT_HH_INCOME}) 
        self.gdf_income_groups = gdf_rio.to_crs(self.crs_local)
        self.apply_jenks_breaks()


    def _bog_preproc_income_groups(self):
        path_data = os.path.join(self.path_root,'0_raw_data/bog/mobility')
        df_income = pd.read_csv(os.path.join(path_data,'feature_data.csv'))
        gdf_zips = gpd.read_file(os.path.join(path_data,'BOG.shp'))
        gdf_income = pd.merge(gdf_zips,df_income,on='tractid')
        gdf_income = gdf_income.rename(columns={'median_income':FT_INCOME_GROUPS})
        self.gdf_income_groups = gdf_income.to_crs(self.crs_local)


    def preproc(self):
        print('Preprocessing income groups...')
        
        # get num of brackets from last digit of feature name, default is 7 if non provided 
        try: self.num_brackets = int(self.feature[-1])
        except: self.num_brackets = 7
        
        if self.city_name in ['ber','ber_inrix']:
            self._ber_preproc_income_groups()
        elif self.city_name in ['bos','lax','sfo']:
            self._us_preproc_income_groups()
        elif self.city_name == 'rio':
            self._rio_preproc_income_groups()
        elif self.city_name=='bog':
            self._bog_preproc_income_groups()
        return self.gdf_income_groups


class Airports():
    def __init__(
        self,
        name = None,
        path_root = None,
        path_airport_file = None,
        ):
    
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 
        self.path_airport_file = path_airport_file

        self.gdf_airports = None


    def _load_airports(self):
        with open(os.path.join(self.path_root,self.path_airport_file), 'r') as stream:
            self.airport_locations = yaml.safe_load(stream)
        

    def preproc(self):
        print('Preprocessing airports...')
        self._load_airports()        
        self.gdf_airports = gpd.GeoDataFrame()
        for airport in self.airport_locations[self.city_name].keys():
            df = pd.DataFrame({'airport':[airport]})
            gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(x=[self.airport_locations[self.city_name][airport][1]], y=[self.airport_locations[self.city_name][airport][0]]),crs=4326)
            self.gdf_airports = pd.concat([self.gdf_airports,gdf])
        self.gdf_airports = self.gdf_airports.to_crs(self.crs_local).reset_index(drop=True)
        return self.gdf_airports



class Employment():
    def __init__(
        self,
        name = None,
        path_root = None,
        ):
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root 

        self.gdf_emp = None
    

    def _preprocess_employment_bog(self):
        def translate(df):
            esp_names = ['ID Encuesta',
                        'Motivo Viaje',
                        'Tiempo Camino',
                        'Medio Predominante',
                        'Latitud Origen',
                        'Latitud Destino',
                        'Longitud Origen',
                        'Longitud Destino',
                        'Diferencia Horas',
                        'Factor Ajuste',
                        'Ponderador Calibrado',
                        'heures',
                        'minutes',
                        'Duración',
                        'Estrato']
            eng_names = ['id',
                        'purpose',
                        'route_timing',
                        'mode',
                        'lat_origin',
                        'lat_destination',
                        'lon_origin',
                        'lon_destination',
                        'time_diff',
                        'adjustment_factor',
                        'calibrated_weighting',
                        'hours',
                        'minutes',
                        'duration',
                        'stratum']
            translations = dict(zip(esp_names, eng_names))
            df = df.rename(columns=translations)

            esp_modes = ['PEATON',
                        'TPC-SITP',
                        'Transmilenio',
                        'AUTO',
                        'BICICLETA,BICICLETA CON MOTOR',
                        'MOTO',
                        'ESPECIAL',
                        'INTERMUNICIPAL',
                        'TAXI',
                        'ALIMENTADOR',
                        'ILEGAL',
                        'OTROS']
            eng_modes = ['walk',
                        'minibus',
                        'brt',
                        'auto',
                        'bike',
                        'motorbike',
                        'special',
                        'regional_bus',
                        'taxi',
                        'bus',
                        'illegal',
                        'other']
            translations = dict(zip(esp_modes, eng_modes))
            df['mode'] = df['mode'].replace(translations)

            esp_purpose = ['Volver a casa',
                        'Trabajar',
                        'Estudiar',
                        'Otra cosa',
                        'Compras',
                        'Tramites',
                        'Recibir atencion en salud',
                        'Buscar / Dejar alguien bajo su cuidad',
                        'Ir a ver a alguien',
                        'Asuntos de Trabajo',
                        'Recreacion',
                        'Buscar/dejar dejar algo',
                        'Comer / Tomar algo',
                        'Buscar / dejar a alguien que no esta bajo su c',
                        'Buscar trabajo']
            eng_purpose = ['return_home',
                        'work',
                        'study',
                        'other',
                        'shopping',
                        'paperwork',
                        'health_care',
                        'care_work',
                        'companion',
                        'work_matters',
                        'recreation',
                        'find_leave_something',
                        'eat_drink',
                        'find_leave_someone_not_under_your_care',
                        'look_for_a_job']
            translations = dict(zip(esp_purpose, eng_purpose))
            df['purpose'] = df['purpose'].replace(translations)
            return df

        def group_mode_purpose(df):
            group_purpose = {
                'work_matters': 'work',
                'find_leave_someone_not_under_your_care': 'companion',
                'find_leave_someone_under_your_care': 'care_work',
                'eat_drink': 'recreation',
                'look_for_a_job': 'other',
                'find_leave_something': 'other'
            }

            aggregate_purpose = {
                'care_work': 'other',
                'health_care': 'other',
                'paperwork': 'other',
                'recreation': 'other',
                }

            aggregate_mode = {
                'minibus': 'transit',
                'regional_bus': 'transit',
                'brt': 'transit',
                'bus': 'transit',
                'auto': 'motorized',
                'taxi': 'motorized',
                'motorbike': 'motorized',
                'illegal': 'other',
                'special': 'other',
                }


            df['purpose'] = df['purpose'].replace(group_purpose)
            df['aggregated_mode'] = df['mode'].replace(aggregate_mode)
            df['aggregated_purpose'] = df['purpose'].replace(aggregate_purpose)
            return df

        def parse_lat_lon(df, lon_col, lat_col):
            df[lat_col] = df[lat_col].apply(lambda x: x / (10 ** (len(str(x)) - 3)))
            df[lon_col] = df[lon_col].apply(lambda x: x / (10 ** (len(str(x)) - 5)))
            return df
        
        def clean_coords(gdf):
            gdf = gdf.loc[~gdf['lat_destination'].isna()]
            gdf = gdf.loc[~gdf['lon_destination'].isna()]
            gdf = gdf.loc[gdf['lat_destination']!=0.0]
            gdf = gdf.loc[gdf['lon_destination']!=0.0]
            return gdf.copy(deep=True)

        def filter_trips_by_purpose(gdf, purpose = None, od_col ='destination'):
            gdf_tmp = gpd.GeoDataFrame(gdf, crs=4326, geometry=gdf[od_col])
            gdf_purpose = gdf_tmp.loc[gdf_tmp['purpose']==purpose]
            return gdf_purpose.drop_duplicates(['lat_origin','lat_destination','lon_origin','lon_destination']).reset_index(drop=True)

        def cluster_in_h3(gdf,aperture_size=11,col_name='num_jobs'):     
            gdf['lng']= gdf['geometry'].x
            gdf['lat']= gdf['geometry'].y
            gdf['hex_id'] = gdf.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,aperture_size),1)
            df_out = gdf.groupby('hex_id').size().to_frame(col_name)
            return df_out.reset_index()
        
        df = pd.read_csv(os.path.join(self.path_root,
                                    '0_raw_data',
                                    'bog',
                                    'employment',
                                    'bogota-mobility-survey-2015.csv'),
                                    sep=';')
        df = translate(df)
        df = group_mode_purpose(df)
        df = parse_lat_lon(df, 'lon_origin', 'lat_origin')
        df = parse_lat_lon(df, 'lon_destination', 'lat_destination')
        df['origin'] = gpd.points_from_xy(df['lon_origin'], df['lat_origin'])
        df['destination'] = gpd.points_from_xy(df['lon_destination'], df['lat_destination'])
        gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry='origin')
        gdf = clean_coords(gdf)
        gdf_emp = filter_trips_by_purpose(gdf, purpose = 'work', od_col='destination')
        df_emp_hex = cluster_in_h3(gdf_emp,aperture_size=11, col_name = FT_EMPLOYMENT)
        self.gdf_emp = utils.add_h3_geom(df_emp_hex,self.crs_local,poly=True)


    def _preprocess_employment_ber(self):
        def load_preprocess_survey():
            path_survey = os.path.join(self.path_root,'0_raw_data/ber/employment/mobility-survey-Berlin.csv')
            survey = pd.read_csv(path_survey)
            survey['Start'] = survey['Trip_Purpose'].apply(lambda x: x.split('-')[0])
            survey['End'] = survey['Trip_Purpose'].apply(lambda x: x.split('-')[-1])
            return survey

        def add_geoms_to_emp(survey,gdf_zip):
            # preprocess zip
            gdf_zip['plz'] = gdf_zip.plz.astype(int)
            gdf_zip = gdf_zip.loc[gdf_zip.plz.isin(survey.Postcode)]
            gdf_emp = pd.merge(survey,gdf_zip[['plz','geometry']],left_on='Des_Plz',right_on='plz')
            return gdf_emp 

        def filter_by_purpose(gdf_emp,gdf_zip):
            # add filter: Work & HH_PNR (to avoid work trips to same location from same person)
            gdf_emp_cleaned = gdf_emp.loc[gdf_emp.End=='Work']
            gdf_emp_cleaned = gdf_emp_cleaned.drop_duplicates(subset='HH_PNR')
            gdf_emp_cleaned = gdf_emp_cleaned.groupby('Des_Plz').size().to_frame(FT_EMPLOYMENT).reset_index()

            # add a geometry column
            gdf_zip_emp = pd.merge(gdf_zip[['plz','geometry']],gdf_emp_cleaned,left_on='plz',right_on='Des_Plz')
            gdf_zip_emp = gdf_zip_emp.to_crs(utils.get_crs_local(self.city_name))
            return gdf_zip_emp

        survey = load_preprocess_survey()
        gdf_zip = gpd.read_file(os.path.join(self.path_root,
                                '0_raw_data/ber/mobility/zip_grid/plz-5stellig.shp'))
        gdf_emp = add_geoms_to_emp(survey,gdf_zip)
        self.gdf_emp = filter_by_purpose(gdf_emp,gdf_zip)

        
    def _preproc_employment_us(self):
        gdf_emp = gpd.read_file(os.path.join(self.path_root,
                            '0_raw_data',
                            self.city_name,
                            'employment',
                            self.city_name+
                            '_employment_SmartLocationDatabase.geojson'))
        gdf_emp = gdf_emp.to_crs(self.crs_local)
        self.gdf_emp = gdf_emp[['TotEmp','geometry']].rename(columns={'TotEmp':FT_EMPLOYMENT})


    def _preproc_employment_rio(self):
        gdf_emp = gpd.read_file(os.path.join(self.path_root,
                            '0_raw_data',
                            self.city_name,
                            'aop',
                            'hex_metro_rio.gpkg'))
        gdf_emp = gdf_emp.to_crs(self.crs_local)
        self.gdf_emp = gdf_emp[['jobs_total','geometry']].rename(columns={'jobs_total':FT_EMPLOYMENT})


    def preproc(self):
        print('Preprocessing employment...')
        if self.city_name == 'bog':
            self._preprocess_employment_bog()
        elif self.city_name in ['ber','ber_inrix']:
            self._preprocess_employment_ber()
        elif self.city_name in ['bos','lax','sfo']:
            self._preproc_employment_us()
        elif self.city_name == 'rio':
            self._preproc_employment_rio()
        return self.gdf_emp


class CorrectArea():
    def __init__(
        self,
        name = None,
        ):
        
        self.city_name = name
        self.buffersize=50
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = utils.get_path_root() 


    def initalize(self):
        g = utils.read_graph(self.path_root, self.city_name, 'full')
        self.gdf = utils.init_geoms(self.path_root,self.city_name,'fua')

        ox_graph = check_adjust_graph_crs(self.gdf,g)
        _, gdf_edges = ox.utils_graph.graph_to_gdfs(ox_graph)
        self.gdf_edges = gdf_edges.reset_index()

    
    def get_street_area(self, buffersize=50):
        print(f'Getting buffered streets for {self.city_name}')
        buff_streets = self.gdf_edges.geometry.buffer(buffersize)
        buff_streets = gpd.GeoDataFrame(geometry=buff_streets)
        gdf_buff = gpd.GeoDataFrame(geometry=[buff_streets.geometry.unary_union], crs=self.gdf.crs)
        
        self.gdf['inter'] = self.gdf.geometry.apply(lambda x: gdf_buff.geometry.iloc[0].intersection(x))
        self.gdf['intersection_area'] = self.gdf['inter'].area
        
    
    def preproc(self):
        self.initalize()
        self.get_street_area()
        return self.gdf[['tractid','intersection_area','geometry']]


class PreprocessFeatures:
    """
    Preprocesses raw feature data.

    In:
    - feature data:
        ../0_raw_data/<CITY_NAME>/<FEATURE_TYPE>
    Out:
    - preprocessed feature data with unified cols across cities in .csv format:
        ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_<FEATURE_TYPE>_epsg<CRS_LOCAL>.csv
    """

    def __init__(
        self,
        name=None,
        path_root=None,
        features=None,
        path_airport_file=None,
        ):
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root
        self.path_airport_file = path_airport_file
        self.features = features


    def initialize(self):
        preprocess_dir = os.path.join(self.path_root, "1_preprocessed_data", self.city_name)
        utils.create_dir(preprocess_dir)


    def save_feature(self,gdf,feature):
        gdf.to_csv(os.path.join(self.path_root,
                                    '1_preprocessed_data',
                                    self.city_name,
                                    self.city_name+'_'+
                                    feature+
                                    '_epsg'+
                                    str(gdf.crs.to_epsg())+
                                    '.csv'),
                                    index=False)


    def _select_class(self, feature):
        if feature == "pop_dense":
            return PopDense(self.city_name, self.path_root)
        if feature == "pop_dense_meta":
            return PopDenseMeta(self.city_name, self.path_root)
        if feature == "income":
            return Income(self.city_name, self.path_root)
        if "income_groups" in feature:
            return IncomeBrackets(self.city_name, self.path_root, feature)
        if feature == "airports":
            return Airports(self.city_name, self.path_root, self.path_airport_file)
        if feature == "employment":
            return Employment(self.city_name, self.path_root)
        if feature == "correct_area":
            return CorrectArea(self.city_name)


    def preprocess(self):
        self.initialize()
        for feature in self.features:
            feature_class = self._select_class(feature)
            gdf = feature_class.preproc()
            self.save_feature(gdf, feature)
        print("All features preprocessed and saved. Closing run.")


def main():
    request = utils.get_input(PROJECT_SRC_PATH, "preprocessing/feature_preproc.yml")
    p = PreprocessFeatures(**request)
    p.preprocess()


if __name__ == "__main__":
    main()
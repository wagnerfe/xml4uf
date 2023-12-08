## Utils function for xml4uf project ##
#Author:     wagnerfe (wagner@mcc-berlin.net)
#Date:       05.09.22
import sys,os
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import json
import osmnx as ox
import yaml,json
import h3
import glob

from shapely import wkt
from shapely.ops import cascaded_union

from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf
from utils.utils_h3 import add_h3_geom, convert_od_to_parent
from utils.utils_mobility import get_df_x, id_to_str, read_od_txt

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))

CRS_UNI = 'epsg:4326'
CRS_US = 'epsg:26914'
CRS_RIO = 'epsg:5641'
CRS_LIS = 'epsg:20790'
CRS_BER = 'epsg:25833'
CRS_BOG = 'EPSG:21897'

ID_COL = {
        'ber':'tractid',
        'ber_inrix':'plz',
        'bos':'tractid',
        'lax':'GEOID',
        'sfo':'TRACT',
        'rio':'Zona',
        'lis':'ID',
        'bog':'tractid',
        }

DEFAULT_FEATURES = ['ft_dist_cbd',
                    'ft_employment_access_01',
                    'ft_income',
                    'ft_beta',
                    'ft_dist_airport',
                    'ft_pop_dense_meta',
                    'ft_pop_dense',
                    'ft_job_dense']

DEFAULT_CITIES = ['ber','bos','lax','sfo','rio','bog']


def load_config():
    with open(os.path.join(PROJECT_SRC_PATH.rsplit('/',1)[0],'bin','env_config.yml'), 'r') as stream:
        config_file = yaml.safe_load(stream)
    return config_file


def get_path_root():
    path_mount, path_data = load_config()
    if path_mount is not None:
        return os.path.join(path_mount,path_data)
    else: 
        return path_data


PATH_ROOT = get_path_root()


def check_if_mounted():
    path_mount = load_config()['path_mount']
    if PROJECT_SRC_PATH.split('/',2)[1] == path_mount.split('/',2)[1]: 
        return True, path_mount
    else: return False, path_mount


def save_dict_to_txt(param_dict, path):
    f = open(path, "w")
    f.write("{\n")
    for k in param_dict.keys():        
        f.write(F"'{k}': '{param_dict[k]}',\n")  # add comma at end of line
    f.write("}")
    f.close()


def get_input(path_source, param_file):
    #  check if on cluster or locally mounted
    mounted, path_mount = check_if_mounted()
    if mounted: 
        with open(os.path.join(path_source.rsplit('/',1)[0],'bin',param_file), 'r') as stream:
            request = yaml.safe_load(stream)[0]
        if 'path_root' in request.keys():
            request['path_root'] = path_mount + request['path_root'] # add mounted path to path_root
    else:
        request = json.load(sys.stdin)
    print_parameter(request)
    return request


def load_pickle(path):
    with open(path, 'rb') as file:
        d = pickle.load(file)
    return d


def load_several_pickles(path):
    list_files = glob.glob(path+'/*.pkl')
    several_pickles = {}
    for i,file_path in enumerate(list_files):
        one_pickle = load_pickle(file_path)
        several_pickles = several_pickles | one_pickle
    return several_pickles


def save_pickle(path, file):
    with open(path, 'wb') as f: 
        pickle.dump(file, f)


def print_parameter(request):
    print('----Parameter-----')
    for x in request:
        if type(request[x])==list:
            if isinstance(request[x][0],str):
                print (x+':')
                [print(' - '+el) for el in request[x]]
        else: print (x+':',request[x])
    print('------------------')


def clean_tractid(gdf,city_name):
    # clean and rename tract id col as they are inconsitent across files
    if city_name in ['ber','ber_inrix']:
        gdf['tractid'] = gdf[ID_COL[city_name]].astype(str)
    elif city_name in ['bos','sfo','lis']:
        gdf['tractid'] = gdf[ID_COL[city_name]].astype(str)
    elif city_name in ['lax']:
        gdf['tractid'] = gdf[ID_COL[city_name]].astype(str).str[1:]
    elif city_name in ['rio']:
        gdf['tractid'] = gdf[ID_COL[city_name]].astype(str).str[:-2]
    elif city_name in ['bog']:
        gdf['tractid'] = gdf[ID_COL[city_name]].astype(str)
    
    gdf['tractid'] = gdf.tractid.apply(lambda x : x[1:] if x.startswith("0") else x)
    return gdf


def get_graph(path_graph):
    #loads graph file via pickle so that we can calculate graph functions
    with open(path_graph, 'rb') as f: 
        g_nx = pickle.load(f)
    return g_nx


def nxg_to_gdf(g):
    # creates gdf representing the streetnetwork with edges
    return ox.utils_graph.graph_to_gdfs(g, 
                                       nodes=False, 
                                       edges=True,
                                       node_geometry=False, 
                                       fill_edge_geometry=True).reset_index(drop=True)


def get_nodes(g):
    # get nodes from graph
    return ox.graph_to_gdfs(g, edges=False)   


def get_crs_local(city):
    # returns based on city name the crs
    if city in ['bos','lax','sfo']: return CRS_US
    elif city in ['ber','ber_inrix']: return CRS_BER
    elif city in ['lis']: return CRS_LIS
    elif city in ['rio']: return CRS_RIO
    elif city in ['bog']: return CRS_BOG


def clean_df_col(df,target_col):
    # remove nans, infs and -infs from any col
    return df[~df[target_col].isin([np.nan, np.inf, -np.inf])].reset_index(drop=True)


def read_graph(path_root, city_name, filter_col):
    crs_local = get_crs_local(city_name)
    crs_ = crs_local.split(':')[1]

    if filter_col =='full':
        return get_graph(os.path.join(path_root,
                                '0_raw_data',
                                city_name,
                                'streets',
                                city_name+'_graph_epsg'+crs_+'_full.pickle'))
    elif filter_col=='filtered':
        return get_graph(os.path.join(path_root,
                                '0_raw_data',
                                city_name,
                                'streets',
                                city_name+'_graph_epsg'+crs_+'_filtered.pickle'))
    else: print('Warning! Graph not found.')



def read_streets(path_root, city_name):
    g = get_graph(os.path.join(path_root,
                                '0_raw_data',
                                city_name,
                                'streets',
                                city_name+'_graph_full.pickle'))
    return ox.utils_graph.graph_to_gdfs(g,
                                        nodes=False,
                                        edges=True,
                                        node_geometry=False,
                                        fill_edge_geometry=True)



def read_od(path_root, city_name, crs_local, day_hour, od_col,resolution='points',bound='fua'):
    """reads in od data from 2_cleaned_data as gdf with origin or destination col"""
    if 'hex' in resolution:
        df_od = pd.read_csv(os.path.join(path_root,
                                            '2_cleaned_data',
                                            city_name,
                                            city_name+
                                            '_od_h11_epsg'+
                                            crs_local.split(':')[1]+                                        
                                            '_t'+str(day_hour)+
                                            '_'+bound+'.csv'))
        df_od = convert_od_to_parent(df_od,int(resolution[3:]))
        return add_h3_geom(df_od,crs_local)

    elif resolution=='points':
        df_od = pd.read_csv(os.path.join(path_root,
                                    '2_cleaned_data',
                                    city_name,
                                    city_name+
                                    '_od_'+
                                    resolution+
                                    '_epsg'+
                                    crs_local.split(':',1)[1]+
                                    '_t'+str(day_hour)+
                                    '_'+bound+'.csv'))        
        df_od = clean_df_col(df_od,'distance_m')
        df_od = id_to_str(df_od)
        df = get_df_x(df_od,od_col,'distance_m')
        return gpd.GeoDataFrame(df,geometry=df['geometry'].apply(wkt.loads),crs=crs_local) 

    elif resolution=='polygons': 
        df_od = pd.read_csv(os.path.join(path_root,
                                    '2_cleaned_data',
                                    city_name,
                                    city_name+
                                    '_od_'+
                                    resolution+
                                    '_epsg'+
                                    crs_local.split(':',1)[1]+
                                    '_t'+str(day_hour)+
                                    '_'+bound+'.csv'))        
        df_od = clean_df_col(df_od,'p_combined')
        df_od = id_to_str(df_od)
        df = merge_zip_geoms(df_od,path_root,city_name,crs_local,'id_'+od_col)
        df = df.drop(columns='tractid')
        return gpd.GeoDataFrame(df,crs=crs_local)


def read_raw_zip_geoms(path_root,city_name, crs_local):
    path_raw_data = os.path.join(path_root,'0_raw_data',city_name,'mobility')
        
    if city_name == 'ber_inrix':
        gdf_zip = gpd.read_file(os.path.join(path_raw_data,'zip_grid/plz-5stellig.shp')).to_crs(crs_local)
    else:
        gdf_zip = gpd.read_file(os.path.join(path_raw_data,
                                            city_name.upper()+'.shp')).to_crs(crs_local)
    gdf_zip = clean_tractid(gdf_zip,city_name)
    return gdf_zip[['tractid','geometry']]


def merge_zip_geoms(df_od, path_root,city_name,crs_local,od_col):
    gdf_zip = read_raw_zip_geoms(path_root,city_name, crs_local)
    return pd.merge(df_od,gdf_zip,left_on=od_col,right_on='tractid',how='left')


def create_geom_id(gdf,city_name):
    return (city_name+'-'+gdf.tractid)


def _get_inter_area(row,gdf_data):
    try:
        # calc intersection area
        out = (row.geometry.intersection(gdf_data.geometry[row.index_right])).area
    except BaseException:
        # in rows which don't intersect with a raster of the density data (NaN)
        out = np.nan
    return out 


def remove_boundary_zips(gdf, gdf_bound, iou=0.5):
    gdf['area'] = gdf.geometry.area
    gdf_joined = gpd.sjoin(gdf[['area','geometry']],gdf_bound, how="left")
    gdf_joined['intersecting_area'] = gdf_joined.apply(lambda row: _get_inter_area(row,gdf_bound), axis=1)
    gdf_joined = gdf_joined[gdf_joined['intersecting_area'].notna()]
    gdf_joined['share'] = gdf_joined['intersecting_area']/gdf_joined['area']
    gdf_out = gdf_joined.loc[gdf_joined['share']<iou]
    gdf = gdf.loc[~gdf.index.isin(gdf_out.index)].reset_index(drop=True)
    return gdf.drop(columns='area')


def init_geoms(path_root, city_name, bound, iou=None):
    path_out = os.path.join(path_root,'3_features',city_name)
    crs_local = get_crs_local(city_name)

    if bound is not None:
        path_geom_file = os.path.join(path_out,city_name+'_'+bound+'_geoms.gpkg')
        gdf_bound = read_bound(path_root,city_name,bound) # we still take fua, but remove outliers
    else: 
        path_geom_file = os.path.join(path_out,city_name+'_geoms.gpkg')
    
    if not os.path.isfile(path_geom_file):
        print('Creating geom file...')
        gdf_geoms = read_raw_zip_geoms(path_root,city_name,crs_local)        
        gdf_geoms = gpd.sjoin(gdf_geoms, gdf_bound)
        
        gdf_geoms['tractid'] = create_geom_id(gdf_geoms, city_name)
        gdf_geoms[['tractid','geometry']].to_file(path_geom_file, driver="GPKG")
    else:
        gdf_geoms = gpd.read_file(path_geom_file)

    if (bound is not None) & (iou is not None):
        len_pre = len(gdf_geoms)
        gdf_geoms = remove_boundary_zips(gdf_geoms,gdf_bound,iou).reset_index(drop=True)
        print(f'Removing {len_pre-len(gdf_geoms)} zips due to iou = {iou} at bounds...')
    
    return gdf_geoms[['tractid','geometry']]


def create_unique_id(df,resolution, city_name, day_hour):
    if resolution=='polygons':
        return (city_name+'-'+str(day_hour)+'-'+df['id_origin']+'-'+df['id_destination'])
    elif resolution == 'points':
        return (city_name+'-'+str(day_hour)+'-'+df['id_origin']+'-'+df['id_destination']+'-'+df.index.map(str))


def read_od_preproc_polygons(path_root,city_name,day_hour):
    path_raw_data = os.path.join(path_root,'0_raw_data',city_name,'mobility')
    
    if city_name == 'ber_inrix':
        df_od = pd.read_csv(os.path.join(path_raw_data,'berlin_od_polygons_raw.csv'))
    elif city_name == 'ber':
        df_od = pd.read_csv(os.path.join(path_raw_data,'OD_t'+str(day_hour)+'.csv'))
    elif city_name == 'bog':
        df_od = pd.read_csv(os.path.join(path_raw_data,'OD_t'+str(day_hour)+'.csv'))
    else:
        df_od = read_od_txt(os.path.join(path_raw_data,'OD'+str(day_hour)+'.fma'))
    df_od = df_od.rename(columns={'origin':'id_origin','destination':'id_destination'})
    return id_to_str(df_od)
    

def read_od_preproc_points(path_root,city_name,crs_local,day_hour):
    df_od = pd.read_csv(os.path.join(path_root,
                                    '1_preprocessed_data',
                                    city_name,
                                    city_name+
                                    '_od_epsg'+crs_local.split(':',1)[1]+
                                    '_t'+str(day_hour)+'.csv'))
    df_od = id_to_str(df_od)
    return clean_df_col(df_od,'distance_m')
    

def read_od_preproc_hex(path_root,city_name,crs_local,day_hour,od_col):
    df_od = pd.read_csv(os.path.join(path_root,
                                        '1_preprocessed_data',
                                        city_name,
                                        city_name+
                                        '_'+od_col+'_epsg'+
                                        crs_local.split(':')[1]+                                        
                                        '_h11'+
                                        '_t'+str(day_hour)+'.csv'))
    return add_h3_geom(df_od,crs_local)

# TODO adjust to 1_preprocessed_data and include in read_feature_preproc
def read_cbd(path_root, city_name, crs_local):
    return import_csv_w_wkt_to_gdf(os.path.join(path_root,
                                            '0_raw_data',
                                            city_name,
                                            'streets',
                                            city_name+
                                            '_cbd_gmaps.csv'),
                                            crs=crs_local)                                         


def read_local_cbd(feature,path_root, city_name, crs_local):
    feature = feature[8:]
    print(feature) 
    gdf_local_cbd = import_csv_w_wkt_to_gdf(os.path.join(path_root,
                                                '1_preprocessed_data',
                                                city_name,'local_cbd',
                                                city_name+
                                                '_'+feature+'_epsg'+
                                                crs_local.split(':')[1]+
                                                '.csv'),    
                                                crs=crs_local)   
    return gdf_local_cbd[['geometry']]


def write_hex_od(gdf_tmp,path_root, city_name, crs_local, day_hour,od_col):
    gdf_tmp.to_csv(os.path.join(path_root,
                                '1_preprocessed_data',
                                city_name,
                                city_name+
                                '_'+od_col+'_epsg'+
                                crs_local.split(':')[1]+
                                '_h11'+
                                '_t'+str(day_hour)+'.csv'),index=False)


def read_feature_preproc(feature,path_root,city_name,crs_local):
    if feature == 'transit_access':
        return gpd.read_file(os.path.join(path_root,
                                          '0_raw_data',
                                          'transit-access-score',
                                          city_name+'-access-score.gpkg'))
    elif feature == 'land_use':
        return load_pickle(os.path.join(path_root,
                                          '0_raw_data',
                                          'land_use',
                                          city_name+'_df_top100.pkl'))
    else:
        return import_csv_w_wkt_to_gdf(os.path.join(path_root,
                                '1_preprocessed_data',
                                city_name,
                                city_name+
                                '_'+feature+'_epsg'+
                                crs_local.split(':',1)[1]+
                                '.csv'),crs=crs_local)


def _remove_unnamed0(df):
    if 'Unnamed: 0' in df.columns:
        return df.drop(columns='Unnamed: 0')
    else: return df


def get_file_name(feature_name,bound = None, target_time = None, od_col = None, testing = False): 
    if feature_name in ['distance_m','num_trips']:
        file_name_loc = 'ft_'+feature_name
    else: file_name_loc = feature_name

    if testing: file_name_loc = 'test_'+file_name_loc

    if feature_name in ['distance_m','num_trips']:
        file_name_loc += '_'+od_col[0:4]
        lst_times = [str(l) for l in target_time]
        file_name_loc = file_name_loc+'_h'+''.join(lst_times)

    if bound is not None:file_name_loc += '_'+bound
    
    return file_name_loc + '.csv'


def load_features(city_name,
                features = DEFAULT_FEATURES,
                target = 'distance_m',
                target_time = [6,7,8,9], 
                bound = 'fua',
                add_geoms = True, 
                iou = None,
                path_root = PATH_ROOT, 
                testing=False):

    # load geoms
    dir_name = os.path.join(path_root,'3_features',city_name)        
    
    df = init_geoms(path_root, city_name, bound, iou)
    if not add_geoms:
        df = df.drop(columns='geometry')        
        
    # load target
    if target is not None:
        target_file = get_file_name(target,bound,target_time = target_time, od_col='origin', testing = testing)
        df_tmp = pd.read_csv(os.path.join(dir_name,target_file))
        df = pd.merge(df,df_tmp)
    
    # load features
    if features is not None:
        for feature in features:
            file_name = get_file_name(feature,bound,testing = testing)
            if os.path.isfile(os.path.join(dir_name,file_name)):
                df_tmp = pd.read_csv(os.path.join(dir_name,file_name))
                if 'Unnamed: 0' in df_tmp.columns: df_tmp = df_tmp.drop(columns='Unnamed: 0')
                df = pd.merge(df,df_tmp)
        
        # check if ft data found
        if not any('ft_' in col for col in df.columns):
            raise ValueError(f'Error no feature data found in \n {dir_name}')

    return df


def get_full_name(city_name):
    dict_names = {
                'ber':'Berlin',
                'ber_inrix':'Berlin',
                'bos':'Boston',
                'lax':'Los Angeles',
                'sfo':'San Jose',
                'rio':'Rio de Janeiro',
                'lis':'Lisbon',
                'bog':'Bogota',
                }
    return dict_names[city_name]


def read_bound(path_root,city_name,bound=None):
    full_name = get_full_name(city_name)
    crs_local = get_crs_local(city_name)
    if city_name=='rio':
        return gpd.read_file(os.path.join(path_root,
                                        '0_raw_data',
                                        'boundaries',
                                        'aop_rio_boundary.gpkg'))
    else:
        if bound is None:
            return import_csv_w_wkt_to_gdf(os.path.join(path_root,
                                                '0_raw_data',
                                                city_name,
                                                'streets',
                                                city_name+'_bound_epsg'+crs_local.split(':')[1]+'.csv'),
                                                crs=crs_local,geometry_col='0').drop(columns='0')
        elif bound=='fua':
            file_name = 'GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg'
            col_name = 'eFUA_name'
            sfo_index = [8485] # as San Francisco is at index 8485 in fua (as there are several San Joses)
        elif bound=='ucdb':
            file_name = 'GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg'
            col_name = 'UC_NM_MN'
            sfo_index = [9] # as San Francisco is at index 9 in ucdb (as there are several San Joses)
        
        gdf_bounds_tmp = gpd.read_file(os.path.join(path_root,
                                                    '0_raw_data',
                                                    'boundaries',
                                                    file_name))
        gdf_bound = gdf_bounds_tmp.loc[gdf_bounds_tmp[col_name]==full_name].to_crs(crs_local)
        if city_name == 'sfo': gdf_bound = gdf_bound.loc[sfo_index]
        return gdf_bound[['geometry']].reset_index(drop=True)


def list_in_dict(dict_):
    return any([isinstance(dict_[k], list) for k in dict_.keys()])


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        print("created folder : ", path)


def read_old_od(path_root, city_name, crs_local, day_hour, od_col,resolution='points',bound='fua'):
    if resolution=='polygons': 
        dir_name = os.path.join(path_root,'3_features',city_name,'t'+str(day_hour),resolution)   
        return gpd.read_file(os.path.join(dir_name, city_name+'_geoms.gpkg'))



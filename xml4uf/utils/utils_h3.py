## h3 utils function for xml4uf project ##
#Author:     wagnerfe (wagner@mcc-berlin.net)
#Date:       15.09.22

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import h3
import h3pandas


CRS_UNI = 'epsg:4326'


def add_h3_poly_geom(row):
    """ add polygon geometry to each h3 row """
    # TODO: atm h3 needs to be in row 0?!
    points = h3.h3_to_geo_boundary(row[0], True)
    return Polygon(points)


def add_h3_geom(df,crs_local,poly=False):
    """ function to assign a geometry to h3 data
    in:
    - df: dataframe with h3 col
    - crs_local: local crs that geom should be projected to
    - poly: if False (default) return geom as points, if True return geom as polygon boundary of h3 

    out:
    - gdf with addtional column 'geometry' in crs_local projection

    """
    if poly:
        df['geometry'] = (df.apply(add_h3_poly_geom,axis=1))    
        return gpd.GeoDataFrame(df,geometry='geometry', 
                                crs='epsg:4326').to_crs(crs_local) 
    else:
        df['lat'] = df['hex_id'].apply(lambda x: h3.h3_to_geo(x)[0])
        df['lng'] = df['hex_id'].apply(lambda x: h3.h3_to_geo(x)[1])   
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat),crs=4326).to_crs(crs_local).drop(columns=['lat','lng'])


def convert_to_h3_points(gdf,dist_col:str,id_o:str=None,id_d:str=None, APERTURE_SIZE:int=8): #TODO rename to make more clear this is for mobility (groupby distance, sum trips) and add same func for pop dense from process_pop_dense.transform_to_h3
    """
    Function that maps all trip points on a hex grid and normalises trip numbers per
    hexagon.
    Args:
        - gdf: dataframe with cleaned trip origin waypoints
        - APERTURE_SIZE: hex raster; for more info see: https://h3geo.org/docs/core-library/restable/ 
        - crs: crs of input gdf
    Returns:
        - gdf_out: geodataframe with hexagons containing the average trip lengths & duration per hexagon
    Last update: 15/04/21. By Felix.
    """
    # define hex col
    hex_col = 'hex_id'
    # before grouping remove and infs, -infs, nans in distance col
    gdf = gdf[~gdf[dist_col].isin([np.nan, np.inf, -np.inf])]
    #convert crs to crs=4326
    gdf = gdf.to_crs(epsg=4326)
    # 0. convert trip geometry to lat long
    gdf['lng']= gdf['geometry'].x
    gdf['lat']= gdf['geometry'].y
    # 0. find hexs containing the points
    gdf[hex_col] = gdf.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,APERTURE_SIZE),1)
    # 1. group all trips per hexagon and average tripdistancemters
    df_out = gdf.groupby(hex_col)[dist_col].mean().to_frame(dist_col)
    # 3. count number of trips per hex
    df_out['points_in_hex'] = gdf.groupby(hex_col).size().to_frame('cnt').cnt
    # 4. allocate 
    if id_d is not None:
        df_out['ids_origin'] = gdf.groupby(hex_col)[id_o].apply(list).to_frame('ids').ids
        df_out['ids_destination'] = gdf.groupby(hex_col)[id_d].apply(list).to_frame('ids').ids
    else:
        df_out['ids_origin'] = gdf.groupby(hex_col)[id_o].apply(list).to_frame('ids').ids
    # reset index of df_out; keep hex id as col
    df_out = df_out.reset_index()
    ## 4. Get center of hex to calculate new features
    df_out['lat'] = df_out[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    df_out['lng'] = df_out[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])   
    ## 5. Convert lat and long to geometry column 
    gdf_out = gpd.GeoDataFrame(df_out, geometry=gpd.points_from_xy(df_out.lng, df_out.lat),crs=4326)
    if id_d is not None: gdf_out = gdf_out[[hex_col,dist_col,'points_in_hex','ids_origin','ids_destination','geometry']]
    else: gdf_out = gdf_out[[hex_col,dist_col,'points_in_hex','ids_origin','geometry']]
    ## 6. Convert to crs
    gdf_out = gdf_out.to_crs(gdf.crs)
    return gdf_out


def get_h3_polygons(gdf,gdf_bound,APERTURE_SIZE):
    """
    Function to transform polygon data into h3 grid with given aperture size
    
    Args:
        - gdf: dataframe with data provided in polygons
        - APERTURE_SIZE: hex raster; for more info see: https://h3geo.org/docs/core-library/restable/ 
        - crs: crs of input gdf

    Returns:
        - gdf_out: geodataframe 'gdf' and 2 additional columns, containing hexagon IDs and point geometry

    Last update: 15/04/21. By Felix.
    """
    # transform input gdf to right crs
    gdf_4326 = gdf.to_crs(4326) 
    gdf_bound = gdf_bound.to_crs(4326)
    
    # fill up gdf bound with hex
    dfh = gdf_bound.h3.polyfill_resample(APERTURE_SIZE).reset_index().drop(columns='index').rename(columns={'h3_polyfill':'hex_id'})  
    dfh = dfh.drop(columns='geometry')

    # add lat & lng of center of hex 
    dfh['lat']=dfh['hex_id'].apply(lambda x: h3.h3_to_geo(x)[0])
    dfh['lng']=dfh['hex_id'].apply(lambda x: h3.h3_to_geo(x)[1])
    dfh = gpd.GeoDataFrame(dfh,geometry=gpd.points_from_xy(dfh.lng, dfh.lat),crs="epsg:4326")
    #return gdf_4326, dfh
    
    # Intersect Hex Point with gdf Polygons 
    gdf_out = gpd.sjoin(gdf_4326,dfh, how="right")
    gdf_out = gdf_out.drop(columns={"index_left","lat","lng"})

    # transform into predefined crs
    gdf_out = gdf_out.to_crs(gdf.crs)

    return gdf_out


def geom_to_h11_scaled(gdf_in, gdf_bound,col):
    """ gets parent of provided h3 hexagons and sums up the values to new area size
    in:
        - gdf_in: geopandas df
            gdf with equally sized polygon geoms as grid and values in col
        - gdf_bound: geopandas df
            gdf with boundary of analysis
        - col: str
            colname with grid values of interest (values area provided as sum over grid area)
    
    out:
        - hdf_h1: gepandas df
            same as gdf_in but with hex cols and geoms in hex11 and additional col with scaled values on h11 grid
    """
    # get hex per cells (only works with equally sized cells) and transform this as global var for our setup
    n_hex_per_cell = len(gdf_in.iloc[[0]].to_crs(CRS_UNI).h3.polyfill_resample(11))
    # get h3 polys for area
    gdf_h11 = get_h3_polygons(gdf_in,gdf_bound,11) # TODO import this?!
    # take only rowss that are not nan
    gdf_h11 = gdf_h11.loc[~gdf_h11.iloc[:,0].isna()]
    # get pop dense per h11 cell
    gdf_h11[col+'_h11']=gdf_h11[col]/n_hex_per_cell
    return gdf_h11


def h11_to_parent_scaled(df_in_h3,hex_size,col,crs_local=None):
    """ gets parent of prived h3 hexagons and sums up the values to new area size
    in:
        - df: pd.DataFrame  
            df with h3 data as input; h3 must be index of df and col names: 'h_<hex_size>'
        - hex_size: str
            hex_size to convert to
        - crs_local: str
            crs of output gdf; if given functions returns gdf
    
    out:
        - df_pop_h3: dataframe/geopandas dataframe
            contains df_in_h3 but converted to hex_size 
    """
    # depending on aperture size, get parents and sum up over area
    df_pop_h3 = df_in_h3.h3.h3_to_parent(hex_size).reset_index(drop=True)
    if hex_size>9: hex_col = 'h3_'+str(hex_size)
    else: hex_col = 'h3_0'+str(hex_size)
    df_pop_h3 = df_pop_h3.groupby(hex_col)[col].sum().to_frame()
    
    # if crs given, return gdf with geometries
    if crs_local is not None:
        gdf_pop_h3 = df_pop_h3.h3.h3_to_geo_boundary()
        return gdf_pop_h3.to_crs(crs_local)
    else: return df_pop_h3


def wrapper_conv_geom_h3_scaled(gdf,gdf_bound,pop_col,hex_size,crs_local):
    """Takes raw pop dense data as input gdf and returns in any hex resolution
    """
    if 'hex_id' in gdf.columns: gdf = gdf.rename(columns={'hex_id':'id'})
    # get h11 fro geom
    gdf_pop = geom_to_h11_scaled(gdf,gdf_bound,pop_col)
    # set index
    gdf_pop=gdf_pop.rename(columns={'hex_id':'h3_11'})
    gdf_pop = gdf_pop.set_index('h3_11')
    # scale to desired hex_size
    gdf_pop = h11_to_parent_scaled(gdf_pop, hex_size,pop_col,crs_local)
    return gdf_pop


def convert_pop_to_parent(gdf,to_hex_size):
    """converts hex od data to parent hex. 
    Function takes sum of pop dense per hex.
    In and out hex data contains no geoms"""
    
    if to_hex_size>9: str_hex='h3_'+str(to_hex_size) 
    else: str_hex='h3_0'+str(to_hex_size) 
    
    if gdf.duplicated(subset='hex_id').any():
        print('found duplicated hex cols in pop data, removing dupls')
        gdf = gdf.drop_duplicates(subset='hex_id')
    
    gdf_tmp = gdf.rename(columns={'hex_id':'h3_11'})
    gdf_tmp = gdf_tmp.set_index('h3_11')

    if to_hex_size<=11:
        gdf_tmp = gdf_tmp.h3.h3_to_parent(to_hex_size).reset_index(drop=True)
        gdf_out = gdf_tmp.groupby(str_hex)['total'].sum().to_frame().reset_index()
        return gdf_out.rename(columns={str_hex:'hex_id'})
    else: raise ValueError('Error hex size > h3_11, which is not supported!')    


def convert_od_to_parent(gdf,to_hex_size):
    """converts hex od data to parent hex. 
    Function takes average tripdistance and sum of points per hex.
    In and out hex data contains no geoms"""

    if to_hex_size>9: str_hex='h3_'+str(to_hex_size) 
    else: str_hex='h3_0'+str(to_hex_size) 
    
    if gdf.duplicated(subset='hex_id').any():
        print('found duplicated hex cols in od data, removing dupls')
        gdf = gdf.drop_duplicates(subset='hex_id')
    
    gdf_tmp = gdf.rename(columns={'hex_id':'h3_11'})
    gdf_tmp = gdf_tmp.set_index('h3_11')

    if to_hex_size<=11:
        gdf_tmp = gdf_tmp.h3.h3_to_parent(to_hex_size).reset_index(drop=True)
        gdf_tmp1 = gdf_tmp.groupby(str_hex)['distance_m'].mean().to_frame().reset_index()
        gdf_tmp2 = gdf_tmp.groupby(str_hex)['points_in_hex'].sum().to_frame().reset_index()
        gdf_out = pd.merge(gdf_tmp1,gdf_tmp2,on=str_hex)
        return gdf_out.rename(columns={str_hex:'hex_id'})
    else: raise ValueError('Error hex size > h3_11, which is not supported!')    


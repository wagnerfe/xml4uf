## Mobility data utils function for xml4uf project ##
#Author:     wagnerfe (wagner@mcc-berlin.net)
#Date:       05.09.22


import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
import datetime
import os
from geovoronoi import voronoi_regions_from_coords

from ufo_map.Utils.helpers import import_csv_w_wkt_to_gdf
import utils.utils as utils


def import_trip_csv_to_gdf(path,crs):
    '''
    Import trip csv file from Inrix data with WKT geometry column into a GeoDataFrame

    Last modified: 25/02/2020. By: Felix

    '''
    df = pd.read_csv(path)
    gdf_origin = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.startloclon, df.startloclat),crs=crs)
    gdf_origin = gdf_origin[['tripid','tripdistancemeters','lengthoftrip','startdate','enddate','providertype','geometry'] ]
    gdf_dest = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.endloclon, df.endloclat),crs=crs)
    gdf_dest = gdf_dest[['tripid','tripdistancemeters','lengthoftrip','startdate','enddate','providertype','geometry'] ]
    return (gdf_origin, gdf_dest)


def trip_inside_bounds(gdf_o,gdf_d,gdf_bound):
    """
    Function to get all trips that start and end within given boundaries

    Args:
        - gdf_o: geodataframe with trip origin waypoint
        - gdf_d: geodataframe with trip destination waypoints
        - gdf_bounds: geodataframe with boundaries

    Returns:
        - gdf_out: geodataframe with all trips that start and end inside bounds

    Last update: 14/04/21. By Felix.
    """
    # Get only trips that start / end in berlin
    gdf_in_o = gpd.sjoin(gdf_o,gdf_bound,how='inner',op='within')
    gdf_in_d = gpd.sjoin(gdf_d,gdf_bound,how='inner',op='within')
    gdf_in_o = gdf_in_o.drop(columns="index_right")
    gdf_in_d = gdf_in_d.drop(columns="index_right")

    # Merge to one gdf
    gdf_in_d = gdf_in_d.rename(columns={"geometry": "geometry_dest"})
    gdf_in_d = gdf_in_d[['tripid','geometry_dest']]
    gdf_out = gdf_in_o.merge(gdf_in_d,left_on = 'tripid', right_on = 'tripid')
    return gdf_out


def tripdistance_bounds(gdf,lmin,lmax):
	"""
	Function to set upper and lower bounds on tripdistance.
 
    Args:
        - gdf: geodataframe with trip origin waypoint
        - lmin,lmax: min max bounds for trip length

    Returns:
        - gdf_out: geodataframe with lower and upper bounds

    Last update: 13/04/21. By Felix.
	"""
	gdf_out = gdf[gdf['tripdistancemeters'].between(lmin, lmax)]
	gdf_out = gdf_out.reset_index(drop=True)
	return gdf_out


def set_weektime(gdf, weekend, start_hour, end_hour):
    """
    Function to filter for weekdays or weekends.

    Args:
        - gdf: geodataframe with trip origin waypoint
        - weekend (bool): 
            0 := no weekend (Mo,...,Fr)
            1 := weekend (Sat, Sun)
        - start_hour, end_hour (datetime format):
            f.e. 07:00:00 -> datetime.time(7, 0, 0)

    Returns:
        - gdf_out: geodataframe with trips only on either weekdays or weekends
        and only starting between start_hour and end_hour

    Last update: 13/04/21. By Felix.
    """

    ## in whole function
    if weekend:
        gdf['startdate'] = pd.to_datetime(gdf['startdate'])
        gdf = gdf[((gdf['startdate']).dt.dayofweek) >= 5]
        df_hour = gdf.startdate.dt.time
        gdf_out = gdf[(start_hour<=df_hour)&(df_hour<=end_hour)]
    else:
        gdf['startdate'] = pd.to_datetime(gdf['startdate'])
        gdf = gdf[((gdf['startdate']).dt.dayofweek) < 5]
        df_hour = gdf.startdate.dt.time
        gdf_out = gdf[(start_hour<=df_hour)&(df_hour<=end_hour)]

    return gdf_out


def clean_date(df):
    startdate_cleaned = df['startdate'].str.split('+',expand=True)[0]
    df['startdate'] = startdate_cleaned.str.split('.',expand=True)[0]
    
    enddate_cleaned = df['enddate'].str.split('Z',expand=True)[0]
    df['enddate'] = enddate_cleaned.str.split('.',expand=True)[0]
    return df


def get_h3_waypoints(gdf, colname:str, APERTURE_SIZE:int=8, crs:int=25833):
    """
    !!! CHANGED FUNCTIONALITY IN COMPARISON TO URBANFORMVMT_V1 !!!
    
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
    #hex_col = 'hex'+str(APERTURE_SIZE)
    hex_col = 'hex_id'

    # take colname as geometry col (default:'geometry', else:'geometry_dest') 
    if not colname == 'geometry':
        #print(gdf[colname].head())
        #print(type(gdf[colname]))
        #gdf['geometry'] = gpd.GeoSeries.from_wkt(gdf[colname]).set_crs(crs)
        gdf['geometry'] = gpd.GeoSeries(gdf[colname].apply(wkt.loads),crs=crs)
        print('transfered geometry_dest successfully')

    #convert crs to crs=4326
    gdf = gdf.to_crs(epsg=4326)
    
    # 0. convert trip geometry to lat long
    gdf['lng']= gdf['geometry'].x
    gdf['lat']= gdf['geometry'].y

    # 0. find hexs containing the points
    gdf[hex_col] = gdf.apply(lambda x: h3.geo_to_h3(x.lat,x.lng,APERTURE_SIZE),1)

    return gdf


def discretize_space_time(gdf,
                            id_col = 'TripID',
                            datetime_col = 'CaptureDate',
                            hex_size=8,
                            time_step=30):
    """Function to convert waypoints to discrete space-time points on the hour
        in: 
        gdf:=gdf with one waypoint per row
        hex_size:= aperture_size
        time_step:=time in min
        out: gdf_out:= gdf[id,'t0','t1',...,'tn'], per t we allocate a position marked by hex_id or np.nan
    """
    print('...preparing spatial discretization for hex{}...'.format(hex_size))
    # create additonal column with hex_ids for each waypoint
    gdf_h3 = get_h3_waypoints(gdf,'geometry',hex_size)
    
    # intialise output df
    gdf_out = pd.DataFrame(data=gdf_h3[id_col].unique(), columns=[id_col])

    print(gdf_out.head())

    print('...preparing temporal discretization...')
    # convert to datetime 
    gdf_h3[datetime_col] = pd.to_datetime(gdf_h3[datetime_col])
    
    # group data into x min time steps
    gdf_h3['time_floored']=gdf_h3.CaptureDate.dt.floor(str(time_step)+'min').dt.time

    for t, group in gdf_h3.groupby('time_floored'):
        print('discretizing for t = {}'.format(t))        
        
        # intialise discret time intervalls on trip id
        tx = 't_'+str(t.strftime('%H'))+str(t.strftime('%M'))
        gdf_out[tx] = np.nan
        
        # drop waypoints (duplictae case is when within time step, we have several waypoints) and take only first
        group.drop_duplicates(subset=[id_col], keep = False, inplace=True)
        
        # merge on id
        gdf_out = pd.merge(gdf_out,group[[id_col,'hex_id']], on=id_col,how='left')
        
        # add hex_id to t[x] where apliccable
        gdf_out[tx] = gdf_out.hex_id
        
        # drop hex_id col
        if 'hex_id' in gdf_out.columns: gdf_out = gdf_out.drop(columns='hex_id')

    # return gdf_out
    return gdf_out


def get_df_x(df,od_marker,target_col):
    if od_marker in ['origin']:
        return df[['id','id_origin','id_destination',target_col,'geometry_origin']].rename(columns={'geometry_origin':'geometry'})
    elif od_marker in ['destination']:
        return df[['id','id_origin','id_destination',target_col,'geometry_destination']].rename(columns={'geometry_destination':'geometry'})


def read_od_txt(path):
    # reads od data from txt format
    return pd.read_csv(path,sep=" ",names=['origin','destination','p_combined'],skiprows=6)


def id_to_str(df,raw_col=False):
    o_col = 'id_origin'
    d_col = 'id_destination'
    if raw_col:
        o_col = 'origin'
        d_col = 'destination'

    df[o_col]=df[o_col].astype(str)
    df[d_col]=df[d_col].astype(str)
    return df.reset_index(drop=True)


def sample_on_p(df):
    # sample trips of df based on assigned trip percentages
    # we divide float number to split total number and percentages
    # based on random number between [0,1] we add 1 or 0 to total number for percentages
    df['fix_num'],df['p'] = df['p_combined'].divmod(1)
    df['random'] = np.random.uniform(0,1,len(df))
    df['num_trips'] = df.fix_num
    df.loc[(df.p > df.random),'num_trips'] += 1  
    return df.drop(columns=['fix_num','p','random'])



def clean_veroni_df(df):
    df['trip_cont'] = df[[str(col) for col in list(range(23))]].sum(axis=1)
    df = df.rename(columns={'O_Antenna1':'origin','D_Antenna2':'destination'})
    df['geometry_origin'] = gpd.points_from_xy(df['lon1'],df['lat1'],crs=4326)
    df['geometry_destination'] = gpd.points_from_xy(df['lon2'],df['lat2'],crs=4326)
    return df


def veroni_of_gdf(gdf_o_uni,gdf_boundary):
    # calc veroni tesselation
    coords = np.array([[geom.xy[0][0], geom.xy[1][0]] for geom in gdf_o_uni.geometry])
    region_polys, region_pts = voronoi_regions_from_coords(coords, gdf_boundary.geometry.iloc[0])
    # create geopandas df with ids
    geoms_indices = np.array(list(region_polys.keys()))
    geoms_values = list(region_polys.values())
    geoms = gpd.GeoDataFrame(geometry=geoms_values,crs=4326)
    return pd.merge(geoms,gdf_o_uni.drop(columns='geometry'),left_index=True,right_index=True)


def select_grid_berlin(zip_grid,path_root,city_name):
    if zip_grid=='lor':
        path_grid = os.path.join(path_root,'0_raw_data',city_name,'mobility/zip_grid/Planungsraum_EPSG_25833.shp')
        berlin_lor = gpd.read_file(path_grid).set_crs(utils.CRS_BER)
        berlin_lor = berlin_lor.loc[~berlin_lor['SCHLUESSEL'].isna()]
        return berlin_lor[['SCHLUESSEL','geometry']].rename(columns={'SCHLUESSEL':'tractid'})
    else: 
        path_grid = os.path.join(path_root,'0_raw_data',city_name,'mobility/zip_grid/plz-5stellig.shp')
        berlin_plz = gpd.read_file(os.path.join(path_grid)).to_crs(utils.CRS_BER)
        return berlin_plz[['plz','geometry']].rename(columns={'plz':'tractid'})
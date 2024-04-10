import sys, os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely import wkt
import datetime

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

# imports
from ufo_map.Utils.helpers import *
import utils.utils as utils
import utils.utils_mobility as utils_mobility

ZIPS_AIRPORT_BOG = [170,238,33,195,627,505,245,608]


class PreProc_OD():
    '''
    Preprocesses raw OD mobility data. Exceptions in input data are
    handled by ExceptionManager class.

    In: 
    - OD data: 
        ../0_raw_data/CITY_NAME/mobility/OD0...OD23.fma
    - zip geometries: 
        ../0_raw_data/CITY_NAME/mobility/CITY_NAME.shp 
    - graph:
        ../0_raw_data/CITY_NAME/streets/<CITY_NAME>_graph_epsg<CRS_LOCAL>_full.pickle
    Out:
    - OD data, incl. origin_geometry (Points), destination geometry (Points), distance_m: 
        1_preprocessed_data/../<CITY_NAME>_od_epsg<CRS_LOCAL>_t<DAY_HOUR>.csv
    '''
    def __init__(
            self, 
            name = None,
            path_root = None,
            day_hour = None,
            filters = None,
            sample_on_network = None,
            calc_distance = None,
            ): 
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root
        self.day_hour = day_hour
        self.filters = filters
        self.sample_on_network = sample_on_network
        self.calc_distance = calc_distance

        self.run_info = None
        self.g = None
        self.edges = None
        self.nodes = None
        self.gdf_zip = None
        self.gdf_zip_raw = None # only relevant for Bogota case
        self.df_raw = None # only relevant for Bogota
        self.df_od = None
        self.gdf_points_o = None
        self.gdf_points_d = None
        self.edges_filtered = None


    def load_network(self):
        print('loading edges and zip codes...')
        self.g = utils.read_graph(self.path_root, self.city_name, 'full')
        self.nodes,self.edges = ox.graph_to_gdfs(self.g) # we need nodes when calc shortest dist in shortest_path_od()
        self.edges_filtered = self.edges.loc[self.edges.highway.isin(self.filters)].reset_index(drop=True)


    def prepare_paths(self):
        self.path_data = os.path.join(self.path_root,
                                    '0_raw_data',
                                    self.city_name,
                                    'mobility')
        
        path_out = os.path.join(self.path_root,
                                    '1_preprocessed_data',
                                    self.city_name)

        self.file_path = os.path.join(path_out,
                                    self.city_name+
                                    '_od_epsg'+self.crs_local.split(':',1)[1]+
                                    '_t'+str(self.day_hour))
        
        utils.create_dir(path_out)


    def initialize(self, exep_manager):
        self.load_network()
        self.prepare_paths()

        if self.city_name == 'bog': 
            self.gdf_zip, self.df_od_raw = exep_manager.veroni_bogota()
            self.edges_filtered = exep_manager.add_airport_streets(self.gdf_zip, self.edges, self.edges_filtered)
        elif self.city_name == 'ber':
            self.gdf_zip = exep_manager.get_berlin_zips()
        else:
            self.gdf_zip = gpd.read_file(os.path.join(self.path_data,
                                                    self.city_name.upper()+
                                                    '.shp')).to_crs(self.crs_local)

        self.gdf_zip = utils.clean_tractid(self.gdf_zip,self.city_name)        
        
        if (self.calc_distance)&(not self.sample_on_network):
            self.gdf_points = pd.read_csv(os.path.join(self.file_path+'_no_dist.csv'))


    def num_points_per_zip(self,locx,zip):
        return self.df_od.groupby([locx]).num_trips.sum().loc[zip]


    def sample_points(self, day_hour):
        print('read OD data, clean and sample it...')
        
        if self.city_name=='bog':
            df = self.df_od_raw
        else:   
            df = utils_mobility.read_od_txt(os.path.join(self.path_data,'OD'+str(day_hour)+'.fma'))
        
        df = utils_mobility.id_to_str(df,raw_col=True)
        self.df_od = utils_mobility.sample_on_p(df)


    def sample_on_edges(self,gdf_edges, n):
        gdf_edges = gdf_edges[['geometry', 'length']]
        weights = gdf_edges['length'] / gdf_edges['length'].sum()
        idx = np.random.choice(gdf_edges.index, size=n, p=weights)
        lines = gdf_edges.loc[idx, 'geometry']
        return gpd.GeoDataFrame(geometry=lines.interpolate(lines.length * np.random.rand())).reset_index(drop=True)

    
    def get_edges_sub_graph(self,id):
        gdf_bound = self.gdf_zip[self.gdf_zip.tractid==id]
        for polygon in gdf_bound['geometry']:
            intersecting_nodes = self.nodes[self.nodes.intersects(polygon)].index
            G_sub = self.g.subgraph(intersecting_nodes)
        _,edges_sub = ox.graph_to_gdfs(G_sub)
        return edges_sub.loc[edges_sub.highway.isin(self.filters)].reset_index(drop=True)


    def assign_points_to_zip(self, gdf_sjoin, id, locx):
        #edges_one_zip = gdf_sjoin.loc[gdf_sjoin.tractid == id]
        edges_one_zip = self.get_edges_sub_graph(id)
        
        if len(edges_one_zip)>0:
            num_points = self.num_points_per_zip(locx,id)
            return self.sample_on_edges(edges_one_zip,int(num_points)).rename(columns={'geometry':'geometry_'+locx}) 
        else: return None


    def get_partner_ids(self,df,locx):
        df_tmp = df.loc[df.num_trips!=0]
        df_tmp.loc[df_tmp.num_trips>1,locx] = df_tmp.loc[df_tmp.num_trips>1][locx]+','

        ls = (df_tmp[locx]*df_tmp.num_trips).to_list()
        ls = [l.split(',') for l in ls]
        ls = [item for sublist in ls for item in sublist]
        ls = [l for l in ls if l !='']
        return ls 


    def allocate_per_zip(self,loc1):
        print(f'------- \n allocate points at {loc1}')
        
        if loc1 == 'origin': loc2 = 'destination'
        else: loc2 = 'origin' 

        self.df_od['num_trips']=self.df_od['num_trips'].astype(int)
        df_od_tmp = self.df_od.loc[self.df_od.num_trips!=0]
        
        gdf_sjoin = gpd.sjoin(self.edges_filtered,self.gdf_zip)
        
        gdf = gpd.GeoDataFrame()        
        list_zip_wo_edges = []
        
        for i, id in enumerate(self.gdf_zip.tractid):
            if id in list(df_od_tmp[loc1]):
                num_points = self.num_points_per_zip(loc1,id)
                print('part {} of {}; id: {}, n: {}'.format(i,len(self.gdf_zip),id,int(num_points)))
                dest_id = self.get_partner_ids(df_od_tmp.loc[df_od_tmp[loc1]==id],loc2)
                
                gdf_points_zip = self.assign_points_to_zip(gdf_sjoin, id, loc1)
                
                if gdf_points_zip is not None: # only add if subgraph in assign_points_random has edges (=streets)
                    gdf_points_zip['tractid'] = id
                    gdf_points_zip['id_'+loc2] = dest_id
                    gdf = pd.concat([gdf,gdf_points_zip])
                else: 
                    list_zip_wo_edges.append(id)
                    print('Warning! {} has no edges!'.format(id))
        
        if list_zip_wo_edges: print(f'Zips with no streets: {list_zip_wo_edges}')
        return gdf.reset_index(drop=True)  


    def find_od_pairs(self):
        self.gdf_points_o['id']=(self.gdf_points_o.tractid+self.gdf_points_o.id_destination)
        self.gdf_points_d['id']=(self.gdf_points_d.id_origin+self.gdf_points_d.tractid)
        self.gdf_points_o['key'] = self.gdf_points_o.groupby('id').cumcount()
        self.gdf_points_d['key'] = self.gdf_points_d.groupby('id').cumcount()
        self.gdf_points = self.gdf_points_o.merge(self.gdf_points_d, on=['id', 'key'])
        self.gdf_points = self.gdf_points[['id_origin','id_destination','geometry_origin','geometry_destination']]

        num_loss_o = len(self.gdf_points_o)-len(self.gdf_points)
        num_loss_d = len(self.gdf_points_d)-len(self.gdf_points)
        print('when merging, we loose {} trips from origin and {} from destination'.format(num_loss_o,num_loss_d))


    def shortest_path_od(self,exep_manager):
        self.gdf_points['id_temp'] = self.gdf_points.index

        df_o = self.gdf_points.drop(columns=('geometry_destination')).rename(columns={'geometry_origin':'geometry'})
        df_d = self.gdf_points.drop(columns=('geometry_origin')).rename(columns={'geometry_destination':'geometry'})
        
        if not self.sample_on_network:
            gdf_o = gpd.GeoDataFrame(df_o, geometry=df_o['geometry'].apply(wkt.loads),crs=self.crs_local)
            gdf_d = gpd.GeoDataFrame(df_d, geometry=df_d['geometry'].apply(wkt.loads),crs=self.crs_local)
        else:
            gdf_o = gpd.GeoDataFrame(df_o, geometry='geometry',crs=self.crs_local)
            gdf_d = gpd.GeoDataFrame(df_d, geometry='geometry',crs=self.crs_local)

        print('calculating nearest neighbours of starting points and graph nodes...')
        gdf_o2 = nearest_neighbour(gdf_o, self.nodes)
        gdf_d2 = nearest_neighbour(gdf_d, self.nodes)
        gdf_merge = gdf_o2.merge(gdf_d2, how='left', on='id_temp')
        graph_ig, list_osmids = convert_to_igraph(self.g)

        # call get shortest dist func, where gdf_merge_3426.osmid_x is nearest node from starting point and osmid_y is
        # nearest node from end destination (one of the neighbourhood centers)
        print('calculating shortest distance in network...')
        self.gdf_points['distance_m'] = gdf_merge.apply(lambda x: get_shortest_dist(graph_ig,
                                                    list_osmids,
                                                    x.osmid_x,
                                                    x.osmid_y,
                                                    'length'),
                                                    axis=1)

        dist_start = gdf_o2['distance'][self.gdf_points.distance_m != np.inf]
        dist_end = gdf_d2['distance'][0]
        self.gdf_points.loc[self.gdf_points.distance_m != np.inf,'distance_m'] += dist_start + dist_end 
        self.gdf_points[['id_origin','id_destination','geometry_origin','geometry_destination','distance_m']]
        
        if self.city_name=='bog': 
            self.gdf_points = exep_manager.veroni_to_zip_bogota(self.gdf_points)
        if self.city_name == 'ber': 
            exep_manager.save_berlin_od_polygons(self.gdf_points, self.gdf_zip)


    def save_points(self,with_distance=False):
        print('saving to disk...')        
        if with_distance:
            self.gdf_points.to_csv(os.path.join(self.file_path+'.csv'),index=False)
        else:
            self.gdf_points.to_csv(os.path.join(self.file_path+'_no_dist.csv'),index=False)


    def points_to_nxg_random(self):
        exep_manager = ExceptionManager(self.city_name,self.path_root,self.day_hour)        
        self.initialize(exep_manager) 
        
        if self.sample_on_network: 
            if self.city_name !='ber':   
                self.sample_points(self.day_hour) 
                self.gdf_points_o = self.allocate_per_zip('origin')
                self.gdf_points_d = self.allocate_per_zip('destination')
                self.find_od_pairs()
            else: 
                self.gdf_points = exep_manager.od_pairs_berlin(self.gdf_zip)
            
            print(self.gdf_points)
            self.save_points(with_distance=False)
        
        if self.calc_distance:
            self.shortest_path_od(exep_manager)
            self.save_points(with_distance=True)
        
        print('run finished. closing.')



class ExceptionManager():
    """
    Manages exceptions in mobility input data.
    For Bogota case:
        In:
        - OD data:
            0_raw_data/../'OD_antenna-useSurvey_24h.csv'
        - zip geometries:
            0_raw_data/../bogota_zat_info.geojson
        Out: 
        - zip code matches: id_origin, id_destination, p_combined
            1_preprocessed_data/../bog_od_t<DAY_HOUR>_raw.csv

    """
    def __init__(
            self,
            city_name = None,
            path_root = None,
            day_hour = None,
            ):
        
        self.city_name = city_name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.day_hour = day_hour
        self.path_root = path_root 
        
        self.gdf_zip = None
        self.gdf_zip_raw = None


    def veroni_bogota(self):
        self.gdf_zip_raw = gpd.read_file(os.path.join(self.path_root,'0_raw_data/bog/mobility/',
                                                'bogota_zat_info.geojson'))
        df_od_raw = pd.read_csv(os.path.join(self.path_root,'0_raw_data/bog/mobility/',
                                                'OD_antenna-useSurvey_24h.csv'))
        df_od_raw = utils_mobility.clean_veroni_df(df_od_raw)
        
        gdf_boundary = gpd.GeoDataFrame(geometry=[self.gdf_zip_raw.geometry.unary_union], crs=self.gdf_zip_raw.crs)
        gdf_boundary = gdf_boundary.convex_hull
        gdf_o = gpd.GeoDataFrame(df_od_raw, geometry=df_od_raw['geometry_origin'])
        gdf_o_uni = gdf_o.drop_duplicates(subset='origin').reset_index(drop=True)
        gdf_veroni = utils_mobility.veroni_of_gdf(gdf_o_uni,gdf_boundary)
        gdf_veroni = gdf_veroni[['origin','geometry']].rename(columns={'origin':'tractid'})
        
        df_od_raw = df_od_raw[['origin','destination', str(self.day_hour)]].rename(columns={str(self.day_hour):'p_combined'})
        df_od_raw = df_od_raw.loc[df_od_raw.p_combined!=0].reset_index(drop=True)
        return gdf_veroni.to_crs(self.crs_local), df_od_raw


    def add_airport_streets(self, gdf_veroni, edges, edges_filtered):
        # in bogota, there is only a primary road accessing the airport, which we exclude
        # with filtering in PreProc_OD.load_network(). As an exception we include them again.
        gdf_zips_airport = gdf_veroni.loc[gdf_veroni.tractid.isin(ZIPS_AIRPORT_BOG)]
        edges_airport = gpd.sjoin(edges, gdf_zips_airport[['geometry']]).drop(columns='index_right')
        print(len(edges_filtered))
        print(f'adding {len(edges_airport)} streets')
        return pd.concat([edges_filtered, edges_airport]).reset_index(drop=True)

    
    def _prepare_bogota_data(self,gdf_points):
        df_preproc = gdf_points.rename(columns={'id_origin':'veroni_origin','id_destination':'veroni_destination'})
        df_preproc['id_tmp'] = 'id-'+df_preproc.veroni_origin+'-'+df_preproc.veroni_destination+'-'+df_preproc.index.map(str)

        df_o = df_preproc[['id_tmp','distance_m','geometry_origin']]
        df_d = df_preproc[['id_tmp','geometry_destination']]
        gdf_o = gpd.GeoDataFrame(df_o, geometry=df_o.geometry_origin,crs=self.crs_local)
        gdf_d = gpd.GeoDataFrame(df_d, geometry=df_d.geometry_destination,crs=self.crs_local)

        gdf_zip = self.gdf_zip_raw.to_crs(self.crs_local).reset_index(drop=True)
        gdf_zip['tractid'] = gdf_zip.index
        gdf_zip['geometry'] = gdf_zip.geometry.buffer(3) # we add a 3m buffer to each zip code, due to wholes in the zip code grid data
        return gdf_o,gdf_d,gdf_zip
    

    def _save_bogota_od_polygons(self,df_zip_trips, gdf_zip):
        path_out = os.path.join(self.path_root,'0_raw_data',self.city_name,'mobility')
        df_zip_trips.to_csv(os.path.join(path_out,'OD_t'+str(self.day_hour)+'.csv'),index=False)
        
        # we don't save gdf_zip as it has topology issues when inserting a 3m buffer in _prepare_bogota_data
        print('Saving Bogota geoms without topology issues')
        self.gdf_zip_raw = self.gdf_zip_raw.to_crs(self.crs_local).reset_index(drop=True)
        self.gdf_zip_raw['tractid'] = self.gdf_zip_raw.index
        self.gdf_zip_raw[['tractid','geometry']].to_file(os.path.join(path_out,'BOG.shp'),driver='ESRI Shapefile')
        self.gdf_zip_raw[['tractid','population','median_income']].to_csv(os.path.join(path_out,'feature_data.csv'),index=False)


    def veroni_to_zip_bogota(self, gdf_points):
        gdf_o,gdf_d,gdf_zip = self._prepare_bogota_data(gdf_points)
        gdf_sjoin_o = gpd.sjoin(gdf_zip[['tractid','geometry']], gdf_o).drop_duplicates(subset='id_tmp') # drop points that got allocated more than once due to intersection on bound
        gdf_sjoin_d = gpd.sjoin(gdf_zip[['tractid','geometry']], gdf_d).drop_duplicates(subset='id_tmp')

        gdf_sjoin_o = gdf_sjoin_o.rename(columns={'tractid':'id_origin'})
        gdf_sjoin_d = gdf_sjoin_d.rename(columns={'tractid':'id_destination'})

        gdf_points = pd.merge(gdf_sjoin_o[['id_tmp','geometry_origin','distance_m','id_origin']],gdf_sjoin_d[['id_tmp','id_destination','geometry_destination']],on='id_tmp')
        df_zip_trips = gdf_points.groupby(['id_origin','id_destination']).size().to_frame('p_combined').reset_index()
        self._save_bogota_od_polygons(df_zip_trips, gdf_zip)
        return gdf_points


    def get_berlin_zips(self):
        self.gdf_zip = utils_mobility.select_grid_berlin('plz',self.path_root,self.city_name)
        return self.gdf_zip


    def _filter_day_hour(self):
        if self.day_hour is not None:
            print('Cleaning date time cols in df...')
            self.df = utils_mobility.clean_date(self.df)

            len_1 = len(self.df)
            if type(self.day_hour) == list:
                tstart = datetime.time(int(self.day_hour[0]), 0, 0) # 0am
                tend = datetime.time(int(self.day_hour[-1]), 0, 0)  # 24pm
            else:
                tstart = datetime.time(int(self.day_hour), 0, 0) # 0am
                tend = datetime.time(int(self.day_hour+1), 0, 0)  # 24pm

            self.df['startdate'] = pd.to_datetime(self.df['startdate'])
            df_hour = self.df.startdate.dt.time
            self.df = self.df[(tstart<=df_hour)&(df_hour<=tend)].reset_index(drop=True)            
            print(f'Reduced sample from {len_1} to {len(self.df)} via t{self.day_hour} filter')


    def _sjoin_zip(self,df, gdf_zip, od_col):
        if od_col=='origin': geom_col = 'geometry'
        else: geom_col = 'geometry_dest'
        gdf = gpd.GeoDataFrame(df,geometry=df[geom_col].apply(wkt.loads),crs=self.crs_local)
        return gpd.sjoin(gdf,gdf_zip[['tractid','geometry']]).reset_index(drop=True)
    

    def _load_trips(self):
        path_trips = os.path.join(self.path_root,
                                '0_raw_data/ber/mobility',
                                'ber_trips_fua_buff1000m_weekday_t0_24_consumer.csv') ## WARNING! hardcoded filename
        self.df = pd.read_csv(path_trips)
        self._filter_day_hour() 


    def od_pairs_berlin(self, gdf_zip):
        self._load_trips()
        gdf_o = self._sjoin_zip(self.df,gdf_zip,'origin')
        gdf_o = gdf_o.rename(columns={'tractid':'id_origin','geometry':'geometry_origin'})
    
        gdf_d = self._sjoin_zip(self.df,gdf_zip,'destination')
        gdf_d = gdf_d.rename(columns={'tractid':'id_destination','geometry':'geometry_destination'})
        
        gdf_points = pd.merge(gdf_o[['tripid','id_origin','geometry_origin']],
                        gdf_d[['tripid','id_destination','geometry_destination']],
                        on='tripid')    
        return gdf_points
    

    def save_berlin_od_polygons(self, gdf_points, gdf_zip):
        df_zip_trips = gdf_points.groupby(['id_origin','id_destination']).size().to_frame('p_combined').reset_index()
        path_out = os.path.join(self.path_root,'0_raw_data/ber/mobility')
        df_zip_trips.to_csv(os.path.join(path_out,'OD_t'+str(self.day_hour)+'.csv'),index=False)
        gdf_zip.to_file(os.path.join(path_out,'BER.shp'),driver='ESRI Shapefile')



class ParseRaw_Inrix():
    
    def __init__(
            self,
            name = None,
            process_raw_data = None, # only needed in preproc_inrix
            preproc_inrix = None, # only needed in preproc_inrix
            path_root = None,
            day_hour = None,
            file_name = None,
            zip_grid = None, # only needed in preproc_inrix
            raw_file_name = None,
            boundary_file_name = None,
            buffersize = None,
            lmin = None,
            lmax = None,
            weekend = None,
            starttime = None,
            endtime = None,
            consumer = None,
            ):
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = path_root
        self.file_name = file_name        
        self.raw_file_name = raw_file_name
        self.boundary_file_name = boundary_file_name
        self.buffersize = buffersize
        self.lmin = lmin
        self.lmax = lmax
        self.weekend = weekend
        self.starttime = starttime
        self.endtime = endtime
        self.consumer = consumer
                
        self.path_out = os.path.join(self.path_root, '0_raw_data',self.city_name,'mobility',self.file_name)
        self.df = None
        self.gdf_o = None
        self.gdf_d = None
        self.gdf_bound = None


    def read_data(self):
        print('Reading in raw trip data...')
        path_raw_trips = os.path.join(self.path_root,'0_raw_data',self.city_name,'mobility',self.raw_file_name)
        path_boundary = os.path.join(self.path_root, '0_raw_data','ber','streets',self.boundary_file_name)
        self.gdf_o, self.gdf_d = utils_mobility.import_trip_csv_to_gdf(path_raw_trips, self.crs_local)
        self.gdf_bound = import_csv_w_wkt_to_gdf(path_boundary,self.crs_local)


    def save_data(self, gdf):
        gdf.to_csv(self.path_out,index=False)


    def preprocess_raw_data(self):
        self.read_data()
        if self.buffersize is not None: self.gdf_bound['geometry'] = self.gdf_bound['geometry'].buffer(self.buffersize)
        gdf_trips = utils_mobility.trip_inside_bounds(self.gdf_o,self.gdf_d,self.gdf_bound)
        print(f'N trips after bound filter: {len(gdf_trips)}')

        if self.lmin is not None: 
            if not self.lmax:
                self.lmax = max(gdf_trips.tripdistancemeters)	# (no bound) in meter
            gdf_trips = utils_mobility.tripdistance_bounds(gdf_trips,self.lmin,self.lmax)
            print(f'N trips after triplength filter: {len(gdf_trips)}')

        if self.weekend is not None:
            tstart = datetime.time(int(self.starttime), 0, 0) # 0am
            tend = datetime.time(int(self.endtime), 0, 0)  # 24pm
            
            gdf_trips = utils_mobility.clean_date(gdf_trips)
            gdf_trips = utils_mobility.set_weektime(gdf_trips,self.weekend, tstart, tend)
            print(f'Num trips after weektime filter: {len(gdf_trips)}')

        if self.consumer is not None:
            gdf_trips = gdf_trips[gdf_trips['providertype'].str.match('1: consumer')]
            gdf_trips = gdf_trips.reset_index(drop=True)
            print(f'Num trips after commercial filter: {len(gdf_trips)}')

        self.save_data(gdf_trips)



class PreProc_Inrix():

    def __init__(
            self, 
            request = None,
            ):
        
        self.city_name = request['name']
        self.crs_local = utils.get_crs_local(self.city_name)
        self.path_root = request['path_root']
        self.day_hour = request['day_hour']
        self.file_name = request['file_name']
        self.process_raw_data = request['process_raw_data']
        self.zip_grid = request['zip_grid']
        self.request = request
        
        self.df = None
        self.gdf_o = None
        self.gdf_d = None
        self.gdf_zip = None
        self.df_trip_polygons = None
        self.df_polygons = None
    

    def _read_sjoin(self,df, od_col):
        if od_col=='origin': geom_col = 'geometry'
        else: geom_col = 'geometry_dest'
        gdf = gpd.GeoDataFrame(df,geometry=df[geom_col].apply(wkt.loads),crs=self.crs_local)
        gdf_sjoin = gpd.sjoin(self.gdf_zip[['tractid','geometry']], gdf)
        return gdf_sjoin[['tripid','tractid','tripdistancemeters','geometry']].reset_index(drop=True)


    def _filter_day_hour(self):
        # this function should be part of preprocess_ber.py as daytime filter
        # can and should be specified there. For convenience it is kept here
        # to avoid loading again and again the large raw trip file.
        if self.day_hour is not None:
            len_1 = len(self.df)
            if type(self.day_hour) == list:
                tstart = datetime.time(int(self.day_hour[0]), 0, 0) # 0am
                tend = datetime.time(int(self.day_hour[-1]), 0, 0)  # 24pm
            else:
                tstart = datetime.time(int(self.day_hour), 0, 0) # 0am
                tend = datetime.time(int(self.day_hour+1), 0, 0)  # 24pm

            self.df['startdate'] = pd.to_datetime(self.df['startdate'])
            df_hour = self.df.startdate.dt.time
            self.df = self.df[(tstart<=df_hour)&(df_hour<=tend)].reset_index(drop=True)            
            print(f'Reduced sample from {len_1} to {len(self.df)} via t{self.day_hour} filter')


    def load_data(self):
        print('Loading data...')
        self.gdf_zip = utils_mobility.select_grid_berlin(self.zip_grid,self.path_root,self.city_name)

        path_trips = os.path.join(self.path_root,'0_raw_data',self.city_name,'mobility',self.file_name)
        self.df = pd.read_csv(path_trips)
        self._filter_day_hour() 
        self.gdf_o = self._read_sjoin(self.df,'origin')
        self.gdf_d = self._read_sjoin(self.df,'destination')


    def create_raw_polygons(self):
        self.gdf_o = self.gdf_o.rename(columns={'tractid':'origin'})
        self.gdf_d = self.gdf_d.rename(columns={'tractid':'destination'})
        self.df_trip_polygons = pd.merge(self.gdf_o[['tripid','origin']],self.gdf_d[['tripid','destination']], on='tripid')
        self.df_trip_polygons['p_combined'] = 1
        self.df_trip_polygons = self.df_trip_polygons.reset_index(drop=True)
        # groupby origin and dest gives OD polygon data with p_combined
        self.df_polygons = self.df_trip_polygons.groupby(['origin','destination']).agg({'p_combined': 'sum'}).reset_index()


    def unify_inrix_cols(self):
        self.df = pd.merge(self.df_trip_polygons[['tripid','origin','destination']],self.df,on='tripid')
        self.df = self.df.rename(columns={'tripdistancemeters':'distance_m',
                                        'origin':'id_origin',
                                        'destination':'id_destination',
                                        'geometry':'geometry_origin',
                                        'geometry_dest':'geometry_destination'})


    def save_data(self):
        print('saving to disk...')
        self.df.to_csv(os.path.join(self.path_root,
                                '1_preprocessed_data',
                                self.city_name,
                                self.city_name+
                                '_od_epsg'+self.crs_local.split(':',1)[1]+
                                '_t'+str(self.day_hour)+'.csv'),index=False)

        self.df_polygons.to_csv(os.path.join(self.path_root,
                                    '0_raw_data',
                                    self.city_name,
                                    'mobility',
                                    'OD_t'+str(self.day_hour)+'.csv'),
                                    index=False)


    def clean_data(self):
        self.load_data()
        self.create_raw_polygons()
        self.unify_inrix_cols()
        self.save_data()        
        print('All done. Closing run.')


def main():

    request = utils.get_input(PROJECT_SRC_PATH,'preprocessing/mobility_preproc.yml')
    
    if 'preproc_inrix' in request.keys():
        if request['process_raw_data']==True: 
            print('Parsing raw inix OD data')
            parse_inrix = ParseRaw_Inrix(**request)
            parse_inrix.preprocess_raw_data()
        
        if request['preproc_inrix'] == True:
            preproc = PreProc_Inrix(request)
            preproc.clean_data() 
        
    else:
        preproc = PreProc_OD(**request)
        preproc.points_to_nxg_random()
    
        


if __name__ == "__main__":
    main()    
import sys, os
import pandas as pd
import geopandas as gpd
import osmnx as ox

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

from ufo_map.Utils.helpers import *
import utils.utils as utils


class DownloadStreets():

    def __init__(
            self,
            name = None,
            filter_streets = None,
            full_graph = None,
            filtered_graph = None,
            csv_format = None,
            save_bound = None,
            path_root = None,
            ):
        
        self.city_name = name
        self.crs_local = utils.get_crs_local(self.city_name)
        self.filter  = filter_streets
        self.full_graph = full_graph
        self.filtered_graph = filtered_graph
        self.save_bound = save_bound
        self.path_root = path_root
        self.csv_format=csv_format
        
        self.g = None
        self.g_filtered = None
        self.gdf_boundary = None
        self.run_info = None
        self.path_out = os.path.join(self.path_root,'0_raw_data',self.city_name,'streets')
        

    def get_city_boundary(self):
        print('Get convex hull of city boundary')
        if self.city_name in ['ber','ber_inrix']:
            # take fua as it is larger than berlin area & bound
            self.gdf_boundary = utils.read_bound(self.path_root,self.city_name,bound='fua')
            self.gdf_boundary = self.gdf_boundary.to_crs(utils.CRS_UNI) 
        else:    
            gdf = gpd.read_file(os.path.join(self.path_root,
                                            '0_raw_data',
                                            self.city_name,
                                            'mobility',
                                            self.city_name.upper()+'.shp'))
            gdf = gpd.GeoDataFrame(geometry=[gdf.geometry.unary_union], crs=gdf.crs)
            self.gdf_boundary = gdf.convex_hull
        
            if self.save_bound:
                print('Saving boundary...')
                gdf_boundary_crs = self.gdf_boundary.to_crs(self.crs_local)
                gdf_boundary_crs.to_csv(os.path.join(self.path_root,
                                                '0_raw_data',
                                                self.city_name,
                                                'streets',
                                                self.city_name+'_bound_epsg'+gdf_boundary_crs.crs.to_authority()[1]+'.csv'),
                                                index=False)


    def download_nxgraph(self):
        if self.full_graph:
            print('Downloading full graph...')
            self.g = ox.graph_from_polygon(self.gdf_boundary.geometry.iloc[0],simplify=True,network_type='drive')
            self.g = ox.project_graph(self.g, to_crs=self.crs_local)
        
        if self.filtered_graph:
            print('Downloading filtered graph...')
            self.g_filtered = ox.graph_from_polygon(self.gdf_boundary.geometry.iloc[0],simplify=True,network_type='drive',custom_filter = self.filter)
            self.g_filtered = ox.project_graph(self.g_filtered, to_crs=self.crs_local)            


    def _save_street_types(self, gdf_streets, ending):
        l_index = gdf_streets[[isinstance(val, list) for val in gdf_streets.highway.values]].index
        gdf_streets['highway'].iloc[l_index]=gdf_streets.iloc[l_index].highway.astype(str)

        print('Saving street types...')
        with open(os.path.join(self.path_out,self.city_name+'_road_types'+ending+'.txt'), "w") as output:
            output.write(str(set(gdf_streets.highway)))


    def save_graph(self):
        filename=self.city_name+'_graph_epsg'+self.crs_local.split(':')[1]
        
        if self.full_graph:
            print('Saving full graph...')
            utils.save_pickle(os.path.join(self.path_out,filename+'_full'+'.pickle'),self.g)

        if self.filtered_graph: 
            print('Saving filtered graph...')
            utils.save_pickle(os.path.join(self.path_out,filename+'_filtered'+'.pickle'),self.g_filtered)

        if self.csv_format: 
            if self.full_graph:
                gdf_streets = ox.graph_to_gdfs(ox.get_undirected(self.g), 
                                                                nodes=False, 
                                                                edges=True,
                                                                node_geometry=False, 
                                                                fill_edge_geometry=True).reset_index(drop=True)
                self._save_street_types(gdf_streets, '_full')
                gdf_streets.to_csv(os.path.join(self.path_out,filename+'_full'+'.csv'),index=False)
            
            if self.filtered_graph:
                gdf_streets = ox.graph_to_gdfs(ox.get_undirected(self.g_filtered), 
                                                                nodes=False, 
                                                                edges=True,
                                                                node_geometry=False, 
                                                                fill_edge_geometry=True).reset_index(drop=True)
                
                self._save_street_types(gdf_streets, '_filtered')
                gdf_streets.to_csv(os.path.join(self.path_out,filename+'_filtered'+'.csv'),index=False)
                                                

    def download_street_network(self):
        self.get_city_boundary()
        self.download_nxgraph()
        self.save_graph()
        

def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'downloading/download_streets.yml')

    streets = DownloadStreets(**request)
    streets.download_street_network()
        

if __name__ == "__main__":
    main() 

    
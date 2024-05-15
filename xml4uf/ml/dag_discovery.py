import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

import utils.utils as utils
import utils.utils_ml as utils_ml
import utils.utils_causal as utils_causal

import tigramite.data_processing as pp
import tigramite.plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.robust_parcorr import RobustParCorr


class CausalGraphDiscovery():
    '''
    Run graph discovery based on calculated features. 

    In: 
    - mobility data:
        for points: 
            ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_od_points_epsg<CRS_LOCAL>_<BOUNDARY>.csv
        for polygons:
            ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_od_points_epsg<CRS_LOCAL>_<BOUNDARY>.csv
            ../2_preprocessed_data/<CITY_NAME>/<CITY_NAME>_od_polygons_epsg<CRS_LOCAL>_<BOUNDARY>.csv
    - calculated feature data: 
        ../3_features/<CITY_NAME>/t<DAY_HOUR>/<RESOLUTION>/<FEATURE_NAME>
    Out:
    - DAG results:
        ../4_causal_inference/<RUN_NAME>/<FILE_NAME>.pkl
        ../4_causal_inference/<RUN_NAME>/plots/<FILE_NAME>_dag.png
        ../4_causal_inference/<RUN_NAME>/plots/<FILE_NAME>_scatter.png
        
    '''
    def __init__(self = None,
                name = None,
                folder = None,
                experiment = None,
                path_root = None,
                day_hour = None,
                resolution = None,
                bound = None,
                feature_sample = None,
                fold = None,
                max_folds = None,
                random_fold = None,
                seed=None,
                normalize = None,
                features = None,
                clean_kwargs = None,
                target = None,
                sample_size = None,
                scatter_plot = None,
                density_plot = None,
                save_plots = None,
                dag_plot = None,
                cond_test_name = None,
                tau_max = None,
                pc_alpha = None,
                verbosity = None,
                assumptions = {},
                ):
        
        # file params
        if type(name) == list: self.city_name = name
        else: self.city_name = [name]
        self.folder = folder
        self.experiment = experiment
        self.path_root = path_root 
        self.day_hour = day_hour
        self.resolution = resolution
        self.bound = bound
        self.feature_sample = feature_sample
        self.fold = fold
        self.max_folds = max_folds
        self.random_fold = random_fold
        self.seed = seed
        self.normalize = normalize
        # run params 
        self.features = features
        self.clean_kwargs = clean_kwargs
        self.target = target
        self.sample_size = sample_size
        self.scatter = scatter_plot
        self.density = density_plot
        self.dag = dag_plot
        self.save_plots = save_plots
        # tigramite params
        self.cond_test_name = cond_test_name
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.verbosity = verbosity
        self.assumptions = assumptions
        # vars
        self.path_out = None
        self.file_name = None
        self.df = None
        self.df_tmp = None
        self.tigramite_df = None
        self.id_col = 'tractid'
        self.link_assumptions = None


    def _prepare_file_name(self):
        res_name = self.resolution[0:3]
        
        if self.fold is not None: self.file_name = 'f'+str(self.fold)+'-'+str(self.max_folds)+'_ci_'
        else: self.file_name = 'ci_'

        if self.seed is not None: self.file_name += 's'+str(self.seed)+'_'

        if len(self.city_name) > 1: 
            cities_abbrev = ''
            for c in sorted(self.city_name): cities_abbrev += c[0:2]
        else: cities_abbrev = self.city_name[0]

        self.file_name = (self.file_name+
                        cities_abbrev+
                        '_'+self.bound+
                        '_'+res_name+ 
                        '_'+self.cond_test_name+
                        '_ns'+str(self.sample_size))
        
        # controls for cases when pc alpha vals are provided as list;
        if self.pc_alpha is not None: 
            if isinstance(self.pc_alpha, float): self.file_name = self.file_name+'_alp'+str(self.pc_alpha)
        
        if self.feature_sample: self.file_name = self.file_name +'_nf'+str(self.feature_sample)
        self.file_name+='.pkl'


    def _prepare_path_out(self):
        self.path_out = os.path.join(self.path_root,'4_causal_inference',self.folder)
        if self.experiment is not None: self.path_out = os.path.join(self.path_out,self.experiment)

        Path(self.path_out).mkdir(parents=True, exist_ok=True)
        if self.scatter or self.dag or self.density:
            Path(os.path.join(self.path_out,'plots')).mkdir(parents=True, exist_ok=True)
        self._prepare_file_name()  


    def _prepare_tigramite(self):
        self.df_tmp = self.df.drop(columns=[self.id_col,'city_name'])
        self.var_names = list(self.df_tmp.columns.values)        
        self.tigramite_df = pp.DataFrame(self.df_tmp.values, var_names=self.var_names)


    def _sample_data(self):
        len_w_nan = len(self.df)
        self.df = self.df.dropna()
        print(f'Dropped {len_w_nan-len(self.df)} NaN. Num samples for analysis: {len(self.df)}')

        if self.random_fold == 'balance_city_samples':
            self.df = utils_causal.balance_city_samples(self.df,self.seed)    

        if self.sample_size is not None:
            if self.sample_size<len(self.df):
                self.df = self.df.sample(n=self.sample_size).reset_index(drop=True)
                print(f'Num samples after sampling: {len(self.df)}')
            else: print('Warning! Required sample size larger than available data. No sample created.')
                
        if self.fold is not None:
            self.df = utils_causal.split_folds(self.df ,self.max_folds, self.fold, self.random_fold)
        
        self.sample_size = len(self.df)
        print('Using a {} sample for run...'.format(len(self.df)))


    def load_data(self):
        self.df, _ = utils_ml.load_cities_sample(city_names=self.city_name,
                                                target=self.target,
                                                path_root=self.path_root,
                                                day_hour = self.day_hour,
                                                bound = self.bound, # TODO if necessary add to yml
                                                features = self.features,
                                                norm_cent = self.normalize,
                                                verbose = True,
                                                clean_kwargs=self.clean_kwargs) 


    def initialize_run(self):
        print('Initialising DAG discovery for {}...'.format(self.city_name))
        self._sample_data()
        self._prepare_tigramite()
        self._prepare_path_out()


    def select_cond_ind_test(self):
        if self.cond_test_name == 'ParCorr':
            self.cond_test = ParCorr(significance='analytic')
        if self.cond_test_name == 'RobustParCorr':
            self.cond_test = RobustParCorr(significance='analytic')
        if self.cond_test_name == 'GPDC':
            self.cond_test = GPDC(significance='analytic')
        if self.cond_test_name == 'GPDCtorch':
            self.cond_test = GPDCtorch(significance='analytic')
        if self.cond_test_name == 'CMIknn':
            self.cond_test = CMIknn(significance='shuffle_test')


    def set_assumptions(self):
        if self.assumptions: 
            T,N = self.df_tmp.shape
            self.link_assumptions = utils_causal.init_link_assumptions(N)
            
            if 'ft_dist_cbd' in self.assumptions:
                node_num = 1+self.features.index('ft_dist_cbd')
                self.link_assumptions = utils_causal.set_node_links_soft(node_num,self.link_assumptions,into_node=False) # distance to center can only cause other urban form features

            if 'ft_dist_cbd4' in self.assumptions:
                node_num = 1+self.features.index('ft_dist_cbd4')
                self.link_assumptions = utils_causal.set_node_links_soft(node_num,self.link_assumptions,into_node=False) # distance to center can only cause other urban form features
            
            if 'distance_m' in self.assumptions:
                node_num = 0
                self.link_assumptions = utils_causal.set_node_links_soft(node_num,self.link_assumptions,into_node=True) # VKT can only be caused by urban form
                
            if 'ft_income' in self.assumptions:
                node_num = 1+self.features.index('ft_income')
                self.link_assumptions = utils_causal.set_node_links_soft(node_num,self.link_assumptions,into_node=False) # income can only be a confounder

            if 'ft_income_groups' in self.assumptions:
                node_num = 1+self.features.index('ft_income_groups')
                self.link_assumptions = utils_causal.set_node_links_soft(node_num,self.link_assumptions,into_node=False) # income can only be a confounder

            if 'ft_income_groups3' in self.assumptions:
                node_num = 1+self.features.index('ft_income_groups3')
                self.link_assumptions = utils_causal.set_node_links_soft(node_num,self.link_assumptions,into_node=False) # income can only be a confounder
        
        else: self.link_assumptions=None


    def _create_scatter_plot(self):
        #correlations = self.pcmci.run_bivci(tau_max=self.tau_max, val_only=True)['val_matrix']
        if self.save_plots: 
            print('Saving scatter plot...')
            file_scatter = os.path.join(self.path_out,'plots',self.file_name[:-4]+'_scatter.png')
        else: file_scatter = None
        tp.plot_scatterplots(dataframe=self.tigramite_df,
                            name = file_scatter,
                            setup_args={'figsize':(15, 10)}, 
                            add_scatterplot_args={'matrix_lags':None}); 


    def _create_density_plot(self):
        if self.save_plots:
            print('Saving density plot...')
            file_density = os.path.join(self.path_out,'plots',self.file_name[:-4]+'_density.png')
        else: file_density = None
        tp.plot_densityplots(dataframe=self.tigramite_df, 
                            name = file_density,
                            setup_args={'figsize':(15, 10)}, 
                            add_densityplot_args={'matrix_lags':None})


    def _create_dag_plot(self):
        var_names = utils_causal.translate_var_names(self.var_names)
        if utils.check_if_mounted(): # when mounted locally print plot
            tp.plot_graph(val_matrix=self.results['val_matrix'],
                    graph=self.results['graph'],
                    var_names=var_names,
                    save_name = None,
                    link_colorbar_label='cross-MCI (edges)',
                    node_colorbar_label='auto-MCI (nodes)',
                    ); 
        
        if self.save_plots: 
            file_dag = os.path.join(self.path_out,'plots',self.file_name[:-4]+'_dag.png')
            print('Saving dag plot...')
            tp.plot_graph(val_matrix=self.results['val_matrix'],
                    graph=self.results['graph'],
                    var_names=var_names,
                    save_name = file_dag,
                    link_colorbar_label='cross-MCI (edges)',
                    node_colorbar_label='auto-MCI (nodes)',
                    ); 


    def _add_optimal_alpha_to_file_name(self):
        if self.pc_alpha is None:
            self.pc_alpha = self.results['optimal_alpha']
            self._prepare_file_name()


    def run_pcmci(self):
        self.pcmci = PCMCI(dataframe=self.tigramite_df, 
                            cond_ind_test=self.cond_test,
                            verbosity=self.verbosity)
        self.results = self.pcmci.run_pcmciplus(tau_min=0, 
                                                tau_max=self.tau_max, 
                                                pc_alpha=self.pc_alpha,
                                                link_assumptions=self.link_assumptions)
        self._add_optimal_alpha_to_file_name()
        if self.scatter:
            self._create_scatter_plot()
        if self.density:
            self._create_density_plot()
        if self.dag:
            self._create_dag_plot()


    def save_results(self):
        print('Saving results...')
        self.results['var_names'] = self.var_names
        self.results['df'] = self.df
        with open(os.path.join(self.path_out,self.file_name), 'wb') as f: # TODO define path out
            pickle.dump(self.results, f)        

    def causal_discovery(self):
        self.load_data()
        self.initialize_run()
        self.select_cond_ind_test()
        self.set_assumptions()
        self.run_pcmci()
        self.save_results()
        

def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'ml/dag_discovery.yml')

    cgd = CausalGraphDiscovery(**request)
    cgd.causal_discovery()
    utils.save_dict_to_txt(request, os.path.join(cgd.path_out,cgd.file_name+'_params.txt'))


if __name__ == "__main__":
    main() 

import os,sys
import numpy as np
import pickle 
import glob
import shutil
from pathlib import Path
from collections import Counter

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

#from utils.utils import get_input
import utils.utils as utils
import utils.utils_causal as utils_causal
import tigramite.plotting as tp


def _get_file_name(filepath):
    return filepath.rsplit('/',1)[1]


class SummariseCiRuns():

    def __init__(self= None,
                folder = None,
                experiment = None,
                path_root = None,
                summary_dag_plot = None,
                summary_bar_plot = None
                ):
        self.path_root = path_root
        self.folder = folder
        self.experiment = experiment
        self.dag = summary_dag_plot
        self.bar = summary_bar_plot
        
        self.conf_lev = 0.9
        self.summary_results = None
        self.file_name = None


    def initalize_run(self):
        self.path_out = os.path.join(self.path_root,'4_causal_inference',self.folder)
        if self.experiment is not None: self.path_run = os.path.join(self.path_out, self.experiment) 
        self.path_out = self.path_run.rsplit('/',1)[0]
        list_paths = glob.glob(self.path_run+'/*.pkl')        
        self.lst_files = [_get_file_name(path) for path in list_paths]
        
        print(f'Summarizing {len(self.lst_files)} files for {self.folder}, incl:')
        for f in self.lst_files: print(f)


    def _read_matrix(self,dict_results, matrix):
        len_dict = len(dict_results.keys())
        return np.mean([dict_results[i][matrix] for i in range(len_dict)],axis=0)


    def get_window_result(self,dict_results):
        self.window_results = {}
        len_dict = len(dict_results.keys())
        self.window_results['val_matrix'] = [dict_results[i]['val_matrix'] for i in range(len_dict)]
        self.window_results['graph'] = [dict_results[i]['graph'] for i in range(len_dict)]


    def get_most_common(self,dict_window,j,k):
        list_arrows = [dict_window[i][j][k][0] for i in range(len(dict_window))]
        occurence_count = Counter(list_arrows)
        # 5 is chosen as we can diff between 5 different: 'o-o', '-->', '<--', '', 'x-x'
        most_common = occurence_count.most_common(5)
        most_common_occurences = [x[1] for x in most_common]
        # in case we have several most occurences, set to 'o-o'
        if len(most_common_occurences)==1:
            return [most_common[0][0]], [most_common_occurences[0]]
        else:
            if (most_common_occurences[0]==most_common_occurences[1]): 
                return ['o-o'], [most_common_occurences[0]]
            else:
                return [most_common[0][0]],[most_common_occurences[0]]


    def get_most_frequent_links(self,window_results):
        num_vars = len(window_results['graph'][0])
        most_frequent_links=[]
        counts = []
        
        for j in range(num_vars):
            print(f'---{j}---')
            node_entries = []
            node_counts = []
            for k in range(num_vars):
                most_common, entry_counts = self.get_most_common(window_results['graph'],j,k)
                node_entries +=[most_common]
                node_counts +=[entry_counts]
            
            most_frequent_links+=[node_entries]
            counts+=[node_counts]

        return np.array([most_frequent_links]), np.array([counts])


    def create_dag_plot(self):
        print('Saving dag plot...')
        var_names = utils_causal.translate_var_names(self.results[0]['var_names'])
        
        Path(os.path.join(self.path_out,'plots')).mkdir(parents=True,exist_ok=True)
        file_dag = os.path.join(self.path_out,'plots',self.file_name+'_dag.png')
        tp.plot_graph(graph=self.summary_results['most_frequent_links'], 
                    val_matrix=self.summary_results['val_matrix_mean'],
                    var_names=var_names,
                    save_name = file_dag,
                    link_width=self.summary_results['link_frequency'])


    def save_results(self):
        utils.save_pickle(os.path.join(self.path_out,self.file_name+'.pkl'),self.results)


    def summarize_run(self):
        self.results = {}
        i = 0
        for i,run_file in enumerate(self.lst_files):
            path = os.path.join(self.path_run,run_file)
            self.results[i] = utils.load_pickle(path)
        
        self.summary_results = {} # Generate summary results - copied from tigramite
        self.p_matrix = self._read_matrix(self.results,'p_matrix')
        self.val_matrix = self._read_matrix(self.results,'val_matrix')
        self.get_window_result(self.results)
        self.f, self.c = self.get_most_frequent_links(self.window_results)
        
        self.summary_results['val_matrix_mean'] = np.mean(self.window_results['val_matrix'], axis=0)
        c_int = (1. - (1. - self.conf_lev)/2.)
        self.summary_results['val_matrix_interval'] = np.stack(np.percentile(self.window_results['val_matrix'], axis=0,
                                                        q = [100*(1. - c_int), 100*c_int]), axis=3)
        self.summary_results['most_frequent_links'] = self.f[0]  #.squeeze()
        self.summary_results['link_frequency'] = self.c[0]/float(10) 
        
        if self.experiment is not None:
            self.file_name = 'sum_'+self.experiment+'_'+self.lst_files[0][8:-4]
        else:
            self.file_name = 'sum_'+self.lst_files[0][8:-4]
    

    def create_summary_bar(self):
        data={}
        for i,run_file in enumerate(self.lst_files):
            path = os.path.join(self.path_run,run_file)
            city_name = run_file.split('_')[1]
            data[city_name] = utils.load_pickle(path)
            print(city_name)

        if self.experiment is not None:
            file_name = self.experiment+'_bar.png'
        else:
            file_name = 'bar.png'

        self.path_bar = os.path.join(self.path_out, file_name)
        utils_causal.create_bar_plot(data,self.path_bar)


    def summarize_folder(self):
        self.initalize_run()        
        if self.dag: 
            self.summarize_run()
            self.create_dag_plot()
            self.save_results()
        if self.bar:
            self.create_summary_bar()
        


def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'postprocessing/sum_causal.yml')

    sr = SummariseCiRuns(**request)
    sr.summarize_folder()


if __name__ == "__main__":
    main() 

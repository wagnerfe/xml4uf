import os,sys
import pandas as pd
import numpy as np
import pickle
import glob

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

from utils.utils import get_input, load_pickle
import utils.plotting as plotting
import utils.utils_ml as utils_ml


class SummarizeMl():
    def __init__(self,
                run_name=None,
                file_name=None,
                path_root=None,
                several_folders=None,
                summarise_run=None,
                day_hour=None,
                shap_type=None,
                figures=None,
                title = None,
                rescale_shap = None
                ):      
        
        self.run_name = run_name
        self.file_name = file_name
        self.path_root = path_root
        self.several_folders = several_folders  
        self.summarise_run = summarise_run
        self.day_hour = day_hour
        self.shap_type = shap_type
        self.figures = figures
        self.title = title
        self.rescale_shap = rescale_shap
        
        self.df = None
        self.main_metrics = ['r2_model', 'r2_pred', 'mae_pred', 'rmse_pred', 'mean_sample', 'std_sample']


    def get_city_name(self,file_path):
        return file_path.rsplit('/',1)[1][3:6]


    def get_resolution(self,file_path=None):
        if file_path:
            return file_path.rsplit('/',1)[1][7:10]
        else:
            return self.file_name.split('_')[2]
        

    def get_run_name(self,file_path=None):
        if file_path:
            filename = file_path.rsplit('/',1)[1]
            return filename.rsplit('_')[3:]
        else:
            return self.file_name.rsplit('_')[3:]
        

    def clean_df(self,df):
        df = df.rename(columns={'r2_model':'r2_train',
                        'r2_pred':'r2_test',
                        'mae_pred':'mae_test',
                        'rmse_pred':'rmse_test'})
        df = df[['city', 'resolution','r2_train', 'r2_test', 'mae_test', 'rmse_test', 'mean_sample','std_sample']]
        if self.several_folders:
            return df
        else:
            return df.sort_values(by=['resolution','city']).reset_index(drop=True)


    def summarize_ml_folder(self):
        list_files = glob.glob(self.folder_path+'/*.pkl')

        lst=[]
        for i,file_path in enumerate(list_files):
            cities_results = load_pickle(file_path)
            city = list(cities_results.keys())[0]
            results_dict = cities_results[city]
            dict_metrics = {key: np.round(results_dict[key],2) for key in self.main_metrics}
            dict_metrics['run'] = self.get_run_name(file_path)
            dict_metrics['city'] = self.get_city_name(file_path)
            dict_metrics['resolution'] = self.get_resolution(file_path) 
            lst.append(dict_metrics)
        self.df = pd.DataFrame(lst)        


    def summarize_ml_file(self,folder=None):
        if folder is None: folder = self.folder_path
        else: folder = os.path.join(self.folder_path,folder)
        
        path = os.path.join(folder,self.file_name)
        cities_results = load_pickle(path)
        
        lst=[]
        for i,city in enumerate(cities_results.keys()):
            results_dict = cities_results[city]
            dict_metrics = {key: np.round(results_dict[key],2) for key in self.main_metrics}
            dict_metrics['run_spec'] = self.get_run_name()
            dict_metrics['city'] = city
            dict_metrics['resolution'] = self.get_resolution() 
            lst.append(dict_metrics)
        self.df = pd.DataFrame(lst)


    def summarise_several_folders(self):
        folders =glob.glob(self.folder_path+'/*')
        folders = [f.split('/')[-1] for f in folders]
        
        df_out = pd.DataFrame()
        print('Summarising folders:')
        for folder in folders:
            print(folder)
            self.summarize_ml_file(folder)
            self.df['run'] = folder
            df_out = pd.concat([df_out,self.df])
            self.df=None
        self.df = df_out


    def save_ml_summary(self):
        print(self.df)
        if self.file_name:
                self.df.to_csv(os.path.join(self.folder_path,'0_sum_'+self.file_name[:-4]+'.csv'),index=False)
        else:
            self.df.to_csv(os.path.join(self.folder_path,'0_main_metrics.csv'),index=False)
        print('Saving summary. Closed run.')


    def get_run_data(self):
        list_files = glob.glob(self.folder_path+'/*.pkl')
        data={}
        for i,file_path in enumerate(list_files):
            city_results = load_pickle(file_path)
            
            if i==0: data = city_results
            else: data.update(city_results)
        return data
    
    
    def create_figures(self):
        if self.file_name:
            data = load_pickle(os.path.join(self.folder_path,self.file_name))
        else: 
            data = self.get_run_data()

        if self.rescale_shap:
            data = utils_ml.get_rescaled_explainer(data, self.shap_type)

        plotter = plotting.ShapMaps(data,
                                    self.shap_type,
                                    self.path_root,
                                    self.run_name,
                                    self.figures,
                                    self.title,
                                    self.day_hour)
        plotter.initialize_shap()
        plotter.plot_figures(save_fig=False)


    def summarize(self):
        self.folder_path = os.path.join(self.path_root,'5_ml',self.run_name)
        
        if self.summarise_run:
            if self.file_name: 
                if self.several_folders:
                    self.summarise_several_folders()
                else:
                    self.summarize_ml_file()
            else: self.summarize_ml_folder()
        
            self.df = self.clean_df(self.df)
            self.save_ml_summary()

        if self.figures is not None:
            self.create_figures()
    

def main():
    
    request = get_input(PROJECT_SRC_PATH,'postprocessing/sum_ml.yml')

    sr = SummarizeMl(**request)
    sr.summarize()


if __name__ == "__main__":
    main() 
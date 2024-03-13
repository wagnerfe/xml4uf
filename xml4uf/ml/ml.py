from comet_ml import Experiment
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import shap

from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

CAUSAL_SHAP_PATH = os.path.realpath(os.path.join(PROJECT_SRC_PATH,'causal_shap'))
sys.path.append(CAUSAL_SHAP_PATH)

import utils.utils as utils
import utils.utils_ml as utils_ml
import utils.utils_causal as utils_causal
import utils.plotting as plotting
import causal_shap.explainer as causal_shap


class Predictor():
    def __init__(self=None,
                 df = None,
                 model_name = None,
                 target = None,
                 features = None,
                 split = None,
                 hyper_params = None
                 ):
        
        self.df = df
        self.target = target
        self.features = features
        self.model_name = model_name
        self.split = split
        self.hyper_params = hyper_params

        self.data = {}
        self.model = None
        self.selected_params = None
        self.city = None    


    def splitting(self, city, num_cities, verbose=True):
        if num_cities > 1: fold_name = city
        else: fold_name = None #TODO: citywise SHAP requires foldname to be integer

        self.data = utils_ml.train_test_split(self.df,
                                            self.split,
                                            self.target,
                                            kfold=fold_name,
                                            noise=False,
                                            verbose=verbose)


    def select_model(self):
        if self.model_name == 'LinearRegression':
            self.model = LinearRegression()
        if self.model_name == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor()
        if self.model_name == 'RandomForestRegressor':
            self.model = RandomForestRegressor()
        if self.model_name == 'XGBRegressor':
            self.model = XGBRegressor()


    def tune_hyperparameter(self):
        if utils.list_in_dict(self.hyper_params):
            self.selected_params = utils_ml.optimize_hype(self.model_name,
                                                        self.model,
                                                        self.data['X_train'],
                                                        self.data['y_train'], 
                                                        self.hyper_params)
        else: 
            self.selected_params = self.hyper_params

        self.model.set_params(**self.selected_params)

    
    def fit(self):
        print("Fitting {}".format(self.model_name))
        self.model.fit(self.data['X_train'], self.data['y_train'])
        self.data['y_predict'] = self.model.predict(self.data['X_test'])

    
    def cv_fold(self, folds, eval_shap, causal_order):
        cv = model_selection.KFold(n_splits=folds, shuffle=True, random_state=1)
        
        df_train_all = pd.DataFrame()
        df_test_all = pd.DataFrame()
        y_predict_all = pd.DataFrame()
        y_test_all = pd.DataFrame()
        y_train_all = pd.DataFrame()
        X_train_all = pd.DataFrame()
        X_test_all = pd.DataFrame()
        
        shap_values = pd.DataFrame()
        causal_shap_values = pd.DataFrame()
        
        r2_model_all = []
        r2_pred_all = []
        mae_pred_all = []
        rmse_pred_all = []
        mean_pred_all = []
        std_pred_all = []

        learning_rate_all = []
        n_estimators_all = []
        max_depth_all = []
        colsample_bytree_all = []
        colsample_bylevel_all = []

        self.data['X'] = self.df[[col for col in self.df.columns if 'ft' in col]]
        y = self.df[[self.target]]

        i = 0
        for train_idx, test_idx in cv.split(self.data['X']):
            self.data['X_train'], self.data['X_test'] = self.data['X'].iloc[train_idx], self.data['X'].iloc[test_idx]
            self.data['y_train'], self.data['y_test'] = y.iloc[train_idx], y.iloc[test_idx]
            df_test = self.df[['tractid',self.target]].iloc[test_idx]
            df_train = self.df[['tractid',self.target]].iloc[train_idx]
            df_test['fold'] = i
            df_train['fold'] = i

            # tune & train & predict
            self.tune_hyperparameter()
            self.model.fit(self.data['X_train'], self.data['y_train'], verbose=False, eval_set=[(self.data['X_train'], self.data['y_train']), (self.data['X_test'], self.data['y_test'])])
            self.data['y_predict'] = pd.Series(self.model.predict(self.data['X_test']), index=self.data['X_test'].index)

            # print
            r2_model, r2_pred, mae_pred, rmse_pred, mean_pred, std_pred = utils_ml.print_r2(self.model, self.data, i, get_metrics=True)
            
            # shaps
            self.get_shap(eval_shap, causal_order)
            causal_shap_fold, shap_fold = self.postprocess_shap(eval_shap)

            # concate
            df_test_all = pd.concat([df_test_all, df_test], axis=0)
            y_predict_all = pd.concat([y_predict_all, self.data['y_predict']], axis=0)
            y_test_all = pd.concat([y_test_all, self.data['y_test']], axis=0)
            y_train_all = pd.concat([y_train_all, self.data['y_train']], axis=0)
            X_train_all = pd.concat([X_train_all, self.data['X_train']], axis=0)
            X_test_all = pd.concat([X_test_all, self.data['X_test']], axis=0)
            
            shap_values = pd.concat([shap_values, shap_fold], axis=0)
            causal_shap_values = pd.concat([causal_shap_values, causal_shap_fold], axis=0)

            r2_model_all.append(r2_model)
            r2_pred_all.append(r2_pred)
            mae_pred_all.append(mae_pred)
            rmse_pred_all.append(rmse_pred)
            mean_pred_all.append(mean_pred)
            std_pred_all.append(std_pred)
            
            learning_rate_all.append(self.selected_params['learning_rate'])
            n_estimators_all.append(self.selected_params['n_estimators'])
            max_depth_all.append(self.selected_params['max_depth'])
            colsample_bytree_all.append(self.selected_params['colsample_bytree'])
            colsample_bylevel_all.append(self.selected_params['colsample_bylevel'])
            
            i += 1

        # squeeze and overwrite
        self.data['df_test'] = df_test_all.squeeze(axis=1) 
        self.data['df_train'] = df_train_all.squeeze(axis=1) 
        self.data['X_train'] = X_train_all.squeeze(axis=1) # we keep X_train only for consistency with city_kfold()
        self.data['X_test'] = X_test_all.squeeze(axis=1)
        self.data['y_train'] = y_train_all.squeeze(axis=1) # we keep y_train only for consistency with city_kfold()
        self.data['y_test'] = y_test_all.squeeze(axis=1)
        self.data['y_predict'] = y_predict_all.squeeze(axis=1)
        
        self.data['df_shap'] = shap_values
        self.data['df_causal_shap'] = causal_shap_values
        
        self.data['r2_model_folds'] = r2_model_all
        self.data['r2_pred_folds'] = r2_pred_all
        self.data['mae_pred_folds'] = mae_pred_all
        self.data['rmse_pred_folds'] = rmse_pred_all
        self.data['mean_sample_folds'] = mean_pred_all
        self.data['std_sample_folds'] = std_pred_all

        self.selected_params['learning_rate'] = np.mean(learning_rate_all)
        self.selected_params['n_estimators']= np.mean(n_estimators_all)
        self.selected_params['max_depth'] = np.mean(max_depth_all)
        self.selected_params['colsample_bytree'] = np.mean(colsample_bytree_all)
        self.selected_params['colsample_bylevel'] = np.mean(colsample_bylevel_all)

        self.postprocess_df(eval_shap)
        

    def results(self, city_scaler={}, city=None):
        if city_scaler:
            df_rescaled, _ = utils_ml.rescale_data(self.data, city_scaler[city])
            y_test_abs = df_rescaled['y_test']
            y_predict_abs = df_rescaled['y_predict']
        else:
            y_test_abs, y_predict_abs = self.data['y_test'], self.data['y_predict']

        self.data['r2_model'] = self.model.score(self.data['X_train'],self.data['y_train'])
        self.data['r2_pred'] = metrics.r2_score(y_test_abs,y_predict_abs)
        self.data['mae_pred'] = metrics.mean_absolute_error(y_test_abs,y_predict_abs)
        self.data['rmse_pred'] = np.sqrt(metrics.mean_squared_error(y_test_abs,y_predict_abs))
        self.data['mean_sample'] = y_test_abs.mean()
        self.data['std_sample'] = y_test_abs.std()

        self.data['learning_rate'] = self.selected_params['learning_rate'] 
        self.data['n_estimators'] = self.selected_params['n_estimators'] 
        self.data['max_depth'] = self.selected_params['max_depth'] 
        self.data['colsample_bytree'] = self.selected_params['colsample_bytree'] 
        self.data['colsample_bylevel'] = self.selected_params['colsample_bylevel'] 

        self.data['model'] = self.model
        if city_scaler: self.data['scaler'] = city_scaler[city]
    
        print('--------------------')
        print('Metrics of Model:')
        print("R2: ", round(self.data['r2_model'],3))        
        print(f'Metrics of predicting {self.target}:')
        print('R2: {}'.format(round(self.data['r2_pred'],3)))
        print('MAE: {} m'.format(round(self.data['mae_pred'],1)))
        print('RMSE: {} m'.format(round(self.data['rmse_pred'],1)))
        print('Mean sample: {}'.format(round(self.data['mean_sample'],1)))
        print('Std sample: {}'.format(round(self.data['std_sample'],1)))


    def get_shap(self, eval_shap, causal_order):
        if eval_shap is not None:
            if isinstance(eval_shap, str): eval_shap = [eval_shap]
            if 'tree_shap' in eval_shap: self.tree_shap()
            if 'causal_shap' in eval_shap: self.causal_shap(causal_order)


    def tree_shap(self):
        print(f'Determining shap values for {self.features}...')
        if self.model_name == 'LinearRegression': explainer = shap.explainers.Linear(self.model)
        else: explainer = shap.explainers.Tree(self.model)
        self.data['shap_test'] = explainer(self.data['X_test'])


    def causal_shap(self, causal_order):
        print(f'Determining causal shap values for {self.features}...')
        confounding = [False]*len(causal_order) # TODO update
        explainer_symmetric = causal_shap.Explainer(self.data['X_train'], self.model)
        p = self.data['y_train'].mean()
        self.data['causal_shap_test'] = explainer_symmetric.explain_causal(self.data['X_test'],
                                                                            p,
                                                                            ordering=causal_order,
                                                                            confounding=confounding,
                                                                            asymmetric = False,
                                                                            seed=2)


    def postprocess_shap(self, eval_shap):
        causal_shap_fold = None
        shap_fold = None
        
        if eval_shap is not None:
            if 'causal_shap_test' in self.data.keys():          
                causal_shap_fold = pd.DataFrame(self.data['causal_shap_test'].values, 
                                                index=self.data['X_test'].index, 
                                                columns=self.data['X_test'].columns+'_shap')
            
            if 'shap_test' in self.data.keys():
                shap_fold = pd.DataFrame(self.data['shap_test'].values, 
                                        index=self.data['X_test'].index, 
                                        columns=self.data['X'].columns+'_shap')
        
        return causal_shap_fold, shap_fold
 
    def postprocess_df(self, eval_shap):
        df_y = pd.concat([self.data['y_test'],self.data['y_predict']],axis=1)
        df_y = df_y.rename(columns={self.target:'y_test',0:'y_predict'})
        df_out = pd.merge(self.data['df_test'], df_y, left_index=True,right_index=True) 
        df_out = pd.merge(df_out,self.data['X_test'],left_index=True,right_index=True)
        
        if eval_shap is not None:
            if 'df_causal_shap' in self.data.keys():
                self.data['df_causal_shap'] = pd.merge(df_out, self.data['df_causal_shap'],left_index=True,right_index=True)
                self.data.pop('causal_shap_test',None) # remove leftover from kfold 
            if 'df_shap' in self.data.keys():
                self.data['df_shap'] = pd.merge(df_out, self.data['df_shap'],left_index=True,right_index=True)
                self.data.pop('shap_test',None) # remove leftover from kfold 
        else:
            self.data['df_out'] = df_out

        for key in ['X','df_test']:
            self.data.pop(key,None)
        


class MlXval():
    '''
    Run ML based on calculated features. 

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
    - prediction results:
        ../5_ml/<RUN_NAME>/<FILE_NAME>.pkl
        
    '''
    def __init__(self = None, 
                name = None, 
                run_name = None, 
                path_root = None, 
                day_hour = None,
                resolution = None,
                feature_sample = None, 
                bound = None,
                features = None,
                causal_order = None,
                eval_features = None,
                eval_shap = None,
                track_comet = None,
                target = None,
                sample_size = None,
                figures = None,
                normalize_center = None,
                clean_kwargs = None,
                model = None,
                split = None, # adjust!
                folds = None,
                hyper_params = None,
                ):
        
        # file params
        if type(name) == list: self.city_name = name
        else: self.city_name = [name]
        self.run_name = run_name
        self.path_root = path_root 
        self.day_hour = day_hour
        self.resolution = resolution
        self.feature_sample = None # WARNING Depricated
        self.bound = bound 
        # run params 
        self.features = features
        self.causal_order = causal_order
        self.eval_features = eval_features
        self.eval_shap = eval_shap
        self.track_comet = track_comet
        self.target = target
        self.sample_size = sample_size
        self.figures = figures
        self.norm_cent = normalize_center
        self.clean_kwargs = clean_kwargs
        # ml_params
        self.model_name = model
        self.split = split # as float in perc; f.e 0.8 is 80% test-train split
        self.folds = folds
        self.hyper_params = hyper_params
        # vars
        self.path_out = None
        self.estimator=None
        self.model = None
        self.file_name = None
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_predict = None
        self.df_test = None
        self.results=None
        self.causal_data = None
        self.city_scaler = None


    def load_data(self): 
        print('Initialising ml run for {}'.format(self.city_name))

        if self.causal_order is not None:
            flat_causal_order = [item for sublist in self.causal_order for item in sublist]
            if all([ft in self.features for ft in flat_causal_order]):
                self.features = [ft for ft in self.features if ft in flat_causal_order]
                print(f'For causal order: {self.causal_order}, choosing features:')
                print(self.features)
            else: raise ValueError(f'Error! Found node in causal order that is not in feature list!')

        self.df, self.city_scaler = utils_ml.load_cities_sample(city_names = self.city_name, 
                                                                target = self.target, 
                                                                path_root = self.path_root, 
                                                                day_hour = self.day_hour, 
                                                                bound = self.bound, 
                                                                features = self.features, 
                                                                norm_cent = self.norm_cent,
                                                                show = True,
                                                                clean_kwargs = self.clean_kwargs)
        self.df, self.sample_size = utils_ml.sample_data(self.df, self.sample_size) # sampling might conflict rescaling
                   

    def city_kfold(self):
        self.data_sum = {}
        for city in self.city_name:
            print('-----------')
            print(f'predicting in {city}')
            city_tree = Predictor(self.df,
                             self.model_name,
                             self.target,
                             self.features, 
                             self.split, 
                             self.hyper_params)
            city_tree.splitting(city, len(self.city_name))
            city_tree.select_model()
            city_tree.tune_hyperparameter()
            city_tree.fit()
            city_tree.results(self.city_scaler,city)
            city_tree.get_shap(self.eval_shap, self.causal_order)
            
            self.data_sum[city] = city_tree.data


    def postprocess_kfold(self):
        for city in self.city_name:
            if self.eval_shap is not None:
                if 'tree_shap' in self.eval_shap:
                    df = self.data_sum['all_folds']['df_shap']
                    self.data_sum[city+'_shap_test'] = df.loc[df['tractid'].str.contains(city)].reset_index(drop=True)
                
                if 'causal_shap' in self.eval_shap:
                    df = self.data_sum['all_folds']['df_causal_shap']
                    self.data_sum[city+'_causal_shap_test'] = df.loc[df['tractid'].str.contains(city)].reset_index(drop=True)

            if 'df_out' in self.data_sum['all_folds'].keys():
                df = self.data_sum['all_folds']['df_out']
                self.data_sum[city+'_out'] = df.loc[df['tractid'].str.contains(city)].reset_index(drop=True)

            if self.city_scaler: 
                self.data_sum[city+'_scaler'] = self.city_scaler[city]


    def kfold(self): 
        self.data_sum = {}
        print('-----------')
        print(f'Starting {self.folds} fold CV...')
        tree = Predictor(self.df,
                        self.model_name,
                        self.target,
                        self.features, 
                        self.split, 
                        self.hyper_params)
        tree.select_model()
        tree.cv_fold(self.folds, self.eval_shap, self.causal_order)
        tree.results()            
        self.data_sum['all_folds'] = tree.data
        self.postprocess_kfold()
        

    def _prepare_file_name(self):
        res_name = self.resolution[0:3]
        
        if len(self.city_name) > 1: 
            cities_abbrev = ''
            for c in sorted(self.city_name): cities_abbrev += c[0:2]
        else: cities_abbrev = self.city_name[0]

        self.file_name = ('ml_'+cities_abbrev+
                        '_'+res_name+
                        '_'+self.model_name+
                        '_'+self.target[0:3]+
                        '_sp'+str(self.split))
        if self.sample_size: self.file_name = self.file_name +'_ns'+str(self.sample_size)
        elif self.feature_sample: self.file_name = self.file_name +'_nf'+str(self.feature_sample)


    def _prepare_path_out(self):
        self.path_out = os.path.join(self.path_root,'5_ml',self.run_name)
        Path(self.path_out).mkdir(parents=True, exist_ok=True)
        self._prepare_file_name()


    def _prepare_csv(self):
        main_metrics = ['r2_model',
                        'r2_pred',
                        'mae_pred',
                        'rmse_pred',
                        'mean_sample',
                        'std_sample']
        
        if self.folds == 'city':
            lst=[]
            for i,city in enumerate(self.data_sum.keys()):    
                results_dict = self.data_sum[city]
                dict_metrics = {key: np.round(results_dict[key],2) for key in main_metrics}
                dict_metrics['city'] = city
                lst.append(dict_metrics)
            return pd.DataFrame(lst)
        
        else:
            main_metrics = [m+'_folds' for m in main_metrics]
            fold_results = {key: np.round(self.data_sum['all_folds'][key],2) for key in main_metrics}
            df = pd.DataFrame(fold_results)
            df['fold'] = ['f_'+str(i) for i in range(self.folds)]
            
            summary_stats = df[main_metrics].mean().to_dict()
            summary_stats = {k: round(v, 2) for k, v in summary_stats.items()}
            summary_stats['fold'] = 'f_mean'
            
            return pd.concat([df,pd.DataFrame(summary_stats,index=[self.folds])])


    def plot_shap(self):
        shap_types = utils_ml.get_shap_types(self.eval_shap,self.folds)
        if shap_types:
            for shap_type in shap_types:
                print(f'Plotting shap for {shap_type}')
                plotter = plotting.ShapMaps(self.data_sum,
                                            shap_type,
                                            self.path_root,
                                            self.run_name,
                                            self.figures)
                plotter.initialize_shap()    
                plotter.plot_figures(save_fig=True)
        else: print('No figures created as no shap vals could be found...')


    def save_results(self):
        print('Saving results...')
        self._prepare_path_out() 
        
        with open(os.path.join(self.path_out,self.file_name+'.pkl'), 'wb') as f: 
            pickle.dump(self.data_sum, f)

        df = self._prepare_csv()
        df.to_csv(os.path.join(self.path_out,self.file_name+'.csv'),index=False)


    def ml_training(self):
        self.load_data()

        if self.folds=='city': self.city_kfold()
        else: self.kfold() 
        
        self.save_results()
        
        if self.figures is not None: 
            self.plot_shap()
        
        


def main():
    
    request = utils.get_input(PROJECT_SRC_PATH,'ml/ml.yml')

    ml = MlXval(**request)
    ml.ml_training()
    utils.save_dict_to_txt(request, os.path.join(ml.path_out,ml.file_name+'_params.txt'))


if __name__ == "__main__":
    main() 
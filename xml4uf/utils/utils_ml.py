import os
import geopandas as gpd
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import shap

import utils.utils as utils
import utils.utils_causal as utils_causal


seed = 1
np.random.seed(seed)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")


def train_test_split(df,
                    split,
                    target='distance_m',
                    kfold=None,
                    noise=False,
                    verbose=True):
    
    if kfold:
        df_train = df.loc[df.city_name!=kfold]
        df_test = df.loc[df.city_name==kfold]
        if split is not None:
            if split == 'balance_city_samples':
                df_train = balance_train_samples(df_train)
            else:   
                df_train, df_test = split_fold(df_train, df_test, split)            
    else:
        df_train = df.sample(frac=split,random_state=0)
        df_test = df.drop(df_train.index) 

    X_train = df_train[[col for col in df_train.columns if "ft_" in col]] 
    X_test = df_test[[col for col in df_test.columns if "ft_" in col]] 
    y_train = df_train[target]
    y_test = df_test[target]
        
    if noise:
        X_train["feature_noise"] = np.random.normal(size=len(df_train))
        X_test["feature_noise"] = np.random.normal(size=len(df_test))
    
    if verbose:
        print('X_train: {}'.format(X_train.shape))
        print('y_train: {}'.format(y_train.shape))
        print('X_test: {}'.format(X_test.shape))
        print('y_test: {}'.format(y_test.shape))
        print('Train-Split: {} %'.format(round((len(df_train)/len(df)), 2)))
    return {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test,'df_train':df_train,'df_test':df_test}


def balance_train_samples(df_train,seed=None):
    if seed is None: seed=0

    df_train = df_train.dropna()
    samples = df_train.city_name.value_counts().min()

    df_out = pd.DataFrame()
    for city in set(df_train['city_name']):
        df_city = df_train.loc[df_train.city_name==city].reset_index(drop=True)
        df_city = df_city.sample(n=samples,random_state=seed)
        df_out = pd.concat([df_out,df_city])
    return df_out.reset_index(drop=True)  


def split_fold(df_train, df_test, split):
    # split_fold ensure split ratio in every fold (by adjust train and test size)        
    test_size = (len(df_train)/(1-split))*split
    if len(df_test)>=test_size:
        df_test = df_test.sample(n=int(test_size),random_state=0)
    else:
        train_size = (len(df_test)/split)*(1-split)
        df_train = df_train.sample(n=int(train_size),random_state=0)
    return df_train, df_test


# def assign_mean_distance(df, path_root, city_name, crs_local, day_hour, od_col):
#     gdf_points = utils.read_od(path_root, city_name, crs_local, day_hour, od_col,'points')
#     gdf_points['id_origin'] = gdf_points['id_origin'].astype(str)
#     gdf_points_mean = gdf_points.groupby('id_'+od_col)['distance_m'].mean().to_frame().reset_index()

#     df = df.drop_duplicates(subset='id_'+od_col).reset_index(drop=True)
#     return pd.merge(df, gdf_points_mean, on='id_'+od_col, how='left')


# def assign_num_trips(df, path_root, city_name, crs_local, day_hour, od_col):
#     gdf_points = utils.read_od(path_root, city_name, crs_local, day_hour, od_col,'points')
#     gdf_points['id_origin'] = gdf_points['id_origin'].astype(str)
#     gdf_points_num = gdf_points.groupby('id_'+od_col).size().reset_index(name='num_trips')

#     df = df.drop_duplicates(subset='id_'+od_col).reset_index(drop=True)
#     return pd.merge(df, gdf_points_num, on='id_'+od_col, how='left')


def optimize_hype(model_name, estimator,X_train,y_train,hype_params):
    print("Optimizing Hyperparameters..")
    if model_name == 'RandomForestRegressor':
        if 'learning_rate' in hype_params.keys(): # random forrests don't accept learning rates
            hype_params.pop('learning_rate',None)

    tuning = GridSearchCV(estimator=estimator, param_grid=hype_params, scoring="r2")
    tuning.fit(X_train, y_train)
    print("Best Parameters found: ", tuning.best_params_)
    return tuning.best_params_


def get_sklearn_train_val(X, y): 
    """
    Standard script to obtain train and validation performances of data X in
    predicting y. For performance and off-the-shelf performance, we use
    gradient boosting regression
    
    Args:
        X(np.ndarray or pd.DataFrame): features
        y(np.ndarray or pd.DataFrame): target 
    
    Returns:
        float: train r2 score
        float: train mean abs error
        float: val r2 score
        float: val mean abs error
    """ 

    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    if not isinstance(y, np.ndarray):
        y = y.to_numpy()

    if len(X.shape) == 1:
        X = np.atleast_2d(X).T

    if len(y.shape) == 1:
        y = np.atleast_2d(y).T
        
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    scaled_X = X_scaler.fit_transform(X)
    scaled_y = y_scaler.fit_transform(y)
    
    n_folds = 5

    results = pd.DataFrame(np.zeros((n_folds, 4)), 
        columns=[
            "train_r2",
            "train_mae",
            "val_r2",
            "val_mae",
        ])
    
    kfold = KFold(n_splits=5, shuffle=True)

    for i, (train_index, val_index) in enumerate(kfold.split(scaled_X, scaled_y)):

        train_X, train_y = scaled_X[train_index], scaled_y[train_index]
        val_X, val_y = scaled_X[val_index], scaled_y[val_index]

        model = GradientBoostingRegressor()
        model = model.fit(train_X, train_y.flatten())

        pred_train = np.atleast_2d(model.predict(train_X)).T
        pred_val = np.atleast_2d(model.predict(val_X)).T

        results.at[i, "train_r2"] = r2_score(train_y, pred_train)
        results.at[i, "val_r2"] = r2_score(val_y, pred_val)
        results.at[i, "train_mae"] = mae(
            y_scaler.inverse_transform(train_y).flatten(),
            y_scaler.inverse_transform(pred_train).flatten())
        results.at[i, "val_mae"] = mae(
            y_scaler.inverse_transform(val_y).flatten(),
            y_scaler.inverse_transform(pred_val).flatten())
    
    return results.mean(axis=0)


def eval_feature_importances(X, y):
    """
    Runs the following procedure to how features contribute
    to the overall prediction of y:
    
    1. Runs regression on individual features
    2. In a greedy-forward scheme add features to an feauture-free dataset
    3. Uses a student-t test to evaluate if prediction has improved
    
    The procedure only contains methods that are computationally slim

    This is inspired by Table I from 
    https://ojs.aaai.org/index.php/AAAI/article/view/7806
    
    Args:
        X(pd.DataFrame): predictive data
        y(pd.Series): target
    """ 

    features = X.columns
    metrices = ["train_r2", "train_mae", "val_r2", "val_mae"]
    individual_performances = pd.DataFrame(index=features, columns=metrices)
    individual_performances["abs_pearson_corr"] = X.corrwith(y).abs()
   
    print("Testing individual feature importances.") 
    for feature in features:
        individual_performances.loc[feature, metrices] = get_sklearn_train_val(
            X[[feature]],
            y
        )
    
    individual_performances.sort_values(by="val_mae", 
                                        ascending=True, 
                                        inplace=True)
    print(individual_performances)

    sorted_index = individual_performances.index
    accum_performances = pd.DataFrame(index=sorted_index, columns=metrices)
    
    print("Testing accumulated features starting with most predictive.") 
    for i, feature in enumerate(individual_performances.index):

        features = individual_performances.index[:i+1]
        curr_perf = get_sklearn_train_val(X[features], y)
        accum_performances.loc[feature] = curr_perf
    
    print(accum_performances)

    return {'indiv_perform':individual_performances,'accum_perform':accum_performances}


def shap_importance(shap_values,X):
    vals = np.abs(shap_values.values).mean(0)
    feature_names = X.columns

    fi_shap = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    fi_shap.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)

    shap_rel_vals = fi_shap.feature_importance_vals/sum(fi_shap.feature_importance_vals)
    for feat, imp in zip(fi_shap.col_name, fi_shap.rel_vals):
       print("{}: {}".format(
            feat,
            round(imp, 3)
            )) 
    return shap_rel_vals


def normalize_cols_fold(df,cols):
    for i,col in enumerate(df[cols]):
        if col not in ['ft_city_area_km2', 'ft_network_length_km']:
            df[col] = df[col]/df[col].max()
    return df


def nomralize_cols_sum(df): # centering of cols across all cities
    for col in ['ft_city_area_km2', 'ft_network_length_km']:
        if col in df.columns:
            df[col] = df[col]/df[col].max()
    return df


def center_cols_fold(df,cols):
    for i,col in enumerate(df[cols]):
        if col not in ['ft_city_area_km2', 'ft_network_length_km']:
            df[col] = df[col]-df[col].mean()
    return df


def center_cols_sum(df):
    for col in ['ft_city_area_km2', 'ft_network_length_km']:
        if col in df.columns:
            df[col] = df[col]-df[col].mean()
    return df


def get_iou_from_clean_kwargs(**clean_kwargs):
    if 'iou' in clean_kwargs.keys():
        return clean_kwargs['iou']
    else: 
        return None


def load_cities_sample(city_names = utils.DEFAULT_CITIES,
                        target ='distance_m',
                        path_root=utils.get_path_root(),
                        day_hour=[6,7,8,9],
                        bound='fua',
                        features=utils.DEFAULT_FEATURES,
                        norm_cent=False,
                        show = True,
                        add_geoms=False,
                        clean_kwargs = None):
    city_scaler={}
    df=pd.DataFrame()
    
    if clean_kwargs is not None:
        iou = get_iou_from_clean_kwargs(**clean_kwargs)
    else:
        iou=None

    if add_geoms: city_dict = {}
    
    for city in city_names:
        print(f'Loading data for {city}')
        df_city = utils.load_features(city,
                                    features=features,
                                    target=target,
                                    target_time=day_hour,
                                    bound=bound,
                                    add_geoms=add_geoms, # if add_geoms, return dict (as we cannot have multiple geoms with different crs in same df)
                                    iou = iou, # intersection of union share at bound geoms
                                    path_root = path_root,
                                    testing = False)
        
        if clean_kwargs is not None:
            df_city = apply_cleaning(df_city,
                                    city,
                                    path_root,
                                    day_hour,
                                    bound,
                                    **clean_kwargs)

        if norm_cent:
            scalable_cols = [target]+[col for col in df_city.columns if 'ft' in col]
            city_scaler[city] = StandardScaler()
            df_city[[col for col in df_city.columns if col in scalable_cols]] = city_scaler[city].fit_transform(df_city[[col for col in df_city.columns if col in scalable_cols]])
        
        df_city['city_name'] = city
        if show: print(df_city.head(2))
        
        if add_geoms:
            city_dict[city] = df_city
        else:
            df = pd.concat([df,df_city])
    
    if add_geoms:
        return city_dict, city_scaler
    else:
        df = df.dropna()
        return df.reset_index(drop=True), city_scaler
  

def learning_model(model_name, model, data):
    print("Fitting {}".format(model_name))
    model.fit(data['X_train'], data['y_train'])
    data['y_predict'] = model.predict(data['X_test'])
    return model, data


def sample_data(df, sample_size):
    if sample_size is not None:
        try: 
            df = df.sample(n=sample_size).reset_index(drop=True)
        except: print('Warning! Required sample size larger than available data. No sample created.')
    print('Using a {} sample for run...'.format(len(df)))
    return df,len(df)


def rescale(data, scaler, features):
    df_test = data['df_test']    

    df_test['y_test'] = data['y_test']
    df_test['y_predict'] = data['y_predict']
    df_y_test = df_test[['y_test']+features]
    df_y_predict = df_test[['y_predict']+features]

    y_test_rescaled = pd.DataFrame(scaler.inverse_transform(df_y_test))[0]
    y_predict_rescaled = pd.DataFrame(scaler.inverse_transform(df_y_predict))[0]
    return y_test_rescaled, y_predict_rescaled


def print_r2(model, data, i=None, get_metrics=False):
    r2_train = model.score(data['X_train'], data['y_train'])
    r2_test = r2_score(data['y_test'], data['y_predict'])

    y_test_abs, y_predict_abs = data['y_test'], data['y_predict']
    mae_test = metrics.mean_absolute_error(y_test_abs,y_predict_abs)
    rmse_test = np.sqrt(metrics.mean_squared_error(y_test_abs,y_predict_abs))
    mean_test = np.mean(y_test_abs)
    std_test = np.mean(y_test_abs)

    if i is not None:
        print(f'Fold {i}: R2_train: {r2_train:.3f}, R2_test: {r2_test:.3f}')
    else: 
        print(f'R2_train: {r2_train:.3f}, R2_test: {r2_test:.3f}')
        
    if get_metrics: return r2_train, r2_test, mae_test, rmse_test, mean_test, std_test
    

def get_shap_types(eval_shap, fold_type):
    shap_types = []
    if eval_shap is not None:
        if 'causal_shap' in eval_shap: # TODO adjust to also plot normal Shap?
            print('Plotting causal shap figures...')
            if fold_type=='city': shap_types.append('causal_shap_test')
            else: shap_types.append('causal_shap_values')
        if 'tree_shap' in eval_shap: 
            if fold_type=='city': shap_types.append('shap_test')
            else: shap_types.append('shap_values')
    return shap_types


def load_old_cities_sample(city_names = ['ber','bos','lax','sfo','rio','bog'],
                            features = ['ft_dist_cbd_orig',
                                        'ft_employment_access_1_orig',
                                        'ft_pop_dense_meta_orig',
                                        'ft_income_orig',
                                        'ft_beta_orig'],
                            norm_cent=False,
                            target ='distance_m',
                            path_root= '/Users/felix/cluster_remote/p/projects/eubucco/other_projects/urbanformvmt_global/data',
                            day_hour=9,
                            resolution='polygons',
                            bound='fua',
                            feature_sample=None,
                            ):
    
    if 'hex' in resolution: id_col ='hex_id'
    else: id_col = 'id' 

    city_scaler={}
    data={}
    for city in city_names:
        print(f'Loading data for {city}')
        df_city = create_old_feature_array(path_root, city, day_hour, resolution, bound, features, id_col, target, feature_sample)
        if norm_cent:
            scalable_cols = [target]+[col for col in df_city.columns if 'ft' in col]
            city_scaler[city] = StandardScaler()
            df_city[[col for col in df_city.columns if col in scalable_cols]] = city_scaler[city].fit_transform(df_city[[col for col in df_city.columns if col in scalable_cols]])
            df_city = df_city.dropna()
        data[city] = df_city
    
    return data, id_col, city_scaler


def create_old_feature_array(path_root, city, day_hour, resolution, bound, features, id_col, target, feature_sample):
    df_tmp = utils.read_old_od(path_root, city, utils.get_crs_local(city), day_hour, 'origin',resolution,bound)        
    df_tmp = load_old_features(df_tmp,features,city,path_root, day_hour,id_col,target,feature_sample,resolution)
    return df_tmp


def load_old_features(gdf, 
                features, 
                city_name,
                path_root,
                day_hour,
                id_col, 
                target = 'distance_m',
                feature_sample=None, 
                resolution='points'):


    gdf = gdf[[id_col,target,'geometry']]
    dir_name = os.path.join(path_root,'3_features',city_name,'t'+str(day_hour),resolution)        

    for feature in features:
        file_name = feature+'_'+resolution+'.csv'
        df_tmp = pd.read_csv(os.path.join(dir_name,file_name))
        if 'Unnamed: 0' in df_tmp.columns: df_tmp = df_tmp.drop(columns='Unnamed: 0')
        gdf = pd.merge(gdf,df_tmp, on=id_col)
    return gdf


def rescale_data(city_data, scaler):
    features = [col for col in city_data['X_test'] if 'ft' in col]
    # create df with y predict and y test
    df_test = city_data['df_test'].reset_index(drop=True)
    df_test['y_test'] = city_data['y_test'].reset_index(drop=True)
    df_test['y_predict'] = city_data['y_predict'] # no reset_index as np array
    df_y_test = df_test[['y_test']+features]
    df_y_predict = df_test[['y_predict']+features]

    # rescale
    df_y_test_rescaled = pd.DataFrame(scaler.inverse_transform(df_y_test),columns=df_y_test.columns)
    df_y_predict_rescaled = pd.DataFrame(scaler.inverse_transform(df_y_predict), columns=df_y_predict.columns)
    df_rescaled = pd.merge(df_y_test_rescaled[['y_test']],df_y_predict_rescaled, left_index=True, right_index=True )
    return df_rescaled, df_test['y_predict']


def get_shap_abbreviation(shap_type):
    if shap_type =='causal_shap_test': return 'cshap'
    else: return 'mshap'


def append_shap_vals(city_data, shap_type):
    df = city_data['df_test'].reset_index(drop=True)
    shap_vals = {}
    shap_vals['ft'] = [col+'_'+ get_shap_abbreviation(shap_type) for col in city_data['X_test'] if 'ft' in col]
    shap_vals['values'] = city_data[shap_type].values # getting values out of explainer object
    df_shap = pd.DataFrame(shap_vals['values'],columns = shap_vals['ft'], index=df.index)
    return df_shap


def rescale_shap(city_data, y_predict,df_rescaled, shap_type, min_baseline=False):
    df_shap = append_shap_vals(city_data, shap_type)    
    # calculate %-parts of each shap feature value
    df_shap_parts = df_shap.div(y_predict, axis=0)
    # remove mean of y_predict to get shap val
    # IMPORTANT: we decide to compare shap against city average and NOT mean of training sample (how its usually done with SHAP)
    # Additionally, we allow to compare agansit city min value to spot features with highest shap impact
    if min_baseline:
        y_predict_shap = df_rescaled['y_predict'] - df_rescaled['y_test'].min()
    else:
        y_predict_shap = df_rescaled['y_predict'] - df_rescaled['y_test'].mean()

    # multiply parts with y_predict shap to get same percentages
    df_shap_rescaled = df_shap_parts.mul(y_predict_shap,axis=0)
    return df_shap_rescaled


def update_shap_obj(explainer_in, df_rescaled, features, shap_type):    
    explainer = explainer_in[shap_type]
    df_shap_rescaled = df_rescaled[[col for col in df_rescaled if get_shap_abbreviation(shap_type) in col]]
    # now we assign it to the shap explainer object
    shap_obj = explainer
    shap_obj.values = df_shap_rescaled.values
    shap_obj.base_values = np.repeat(df_rescaled['y_test'].mean(), len(df_rescaled['y_test']))
    shap_obj.data = df_rescaled[features].values
    return shap_obj


def get_feature_names(data):
    first_key = list(data.keys())[0]
    return list(data[first_key]['X_train'].columns)


def rescale(city_data, shap_type=None, min_baseline=False):
    # rescales per individual city; runs without any shap values are not supported
    df_rescaled, y_predict = rescale_data(city_data,city_data['scaler'])
    
    if shap_type is None: shap_types = ['shap_test','causal_shap_test']
    else: shap_types = [shap_type]
    
    for shap_x in shap_types:
        df_shap_rescaled = rescale_shap(city_data, y_predict, df_rescaled, shap_x, min_baseline)
        df_rescaled = pd.merge(df_rescaled, df_shap_rescaled,left_index=True, right_index=True)
    
    return df_rescaled    


def get_rescaled_explainer(data, shap_type = None, min_baseline=False):
    # rescales for all cities, adds df_rescaled and updated explainer object into dict
    # naming is quite bad as its both rescaling the explainer & adding df_rescaled
    features = get_feature_names(data)

    if shap_type is None: shap_types = ['shap_test','causal_shap_test']
    else: shap_types = [shap_type]
    
    for city in data.keys():
        print(f'Rescaling shap values for {city}')
        data[city]['df_rescaled'] = rescale(data[city], shap_type, min_baseline)
        
        for shap_x in shap_types:
            data[city][shap_x] = update_shap_obj(data[city], data[city]['df_rescaled'], features, shap_x)   
        
    return data


def apply_cleaning(df, city, path_root, day_hour, bound, **clean_kwargs):
# clean_kwargs = 
# {
#   'set_lower_bound':
#       ft1: <cut_off value>,
#       ft2: <cut_off value>,
#   'set_upper_bound':
#       ft1: <cut_off value>,
#       ft2: <cut_off value>,
#   'clean_airports':True    
# }

    if 'set_lower_bound' in clean_kwargs.keys():
        if 'num_trips' in clean_kwargs['set_lower_bound'].keys():
            df_trips = utils.load_features(city,
                                features=None,
                                target='num_trips',
                                target_time=day_hour,
                                bound=bound,
                                add_geoms=False,
                                iou=None,
                                path_root = path_root,
                                testing = False)
            df = pd.merge(df,df_trips[['tractid','num_trips']],on='tractid')

        for ft in df.columns:
            if ft in clean_kwargs['set_lower_bound'].keys():
                df = cut_lower_ft_bound(df,ft,cut_off=clean_kwargs['set_lower_bound'][ft])

    if 'set_upper_bound' in clean_kwargs.keys():
        for ft in df.columns:
            if ft in clean_kwargs['set_upper_bound'].keys():
                df = cut_upper_ft_bound(df,ft,cut_off=clean_kwargs['set_upper_bound'][ft])
        
    if 'clean_airports' in clean_kwargs.keys():
        if clean_kwargs['clean_airports']:
            df = clean_airports(df, city, path_root)
    
    if 'num_trips' in df.columns: df=df.drop(columns='num_trips')
    return df


def cut_lower_ft_bound(df,col,cut_off):
    mask=df[col]
    idx=mask.loc[mask>cut_off].index
    print(f'Removing {len(df)-len(idx)} TAZ for cleaning {col} at lower cut_off {cut_off}...')
    return df.loc[idx].reset_index(drop=True)


def cut_upper_ft_bound(df,col,cut_off):
    mask=df[col]
    idx=mask.loc[mask<cut_off].index
    print(f'Removing {len(df)-len(idx)} TAZ for cleaning {col} at upper cut_off {cut_off}...')
    return df.loc[idx].reset_index(drop=True)


def clean_airports(df, city, path_root): 
    crs_local = utils.get_crs_local(city)
    gdf_airports = utils.read_feature_preproc('airports',path_root,city,crs_local) 
    gdf = utils.init_geoms(path_root,city,'fua')
    tractid_airport = gpd.sjoin(gdf, gdf_airports)['tractid']
    print(f'Removing {len(tractid_airport)} TAZ as they are on airports')
    return df.loc[~df['tractid'].isin(tractid_airport)].reset_index(drop=True)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

import utils.utils as utils
import postprocessing.plotting as plotting


def init_link_assumptions(N):
    return {j:{(i, 0):'o?o' for i in range(N) if (i, 0) != (j, 0)} for j in range(N)}


def set_node_links_soft(node,link_assumptions, into_node=True):
    if into_node: a,b = '-?>','<?-'
    else: a,b = '<?-','-?>'
    
    for i in range(len(link_assumptions)):
        if i != node:
            link_assumptions[node][(i,0)] = a
            link_assumptions[i][(node,0)] = b
    return link_assumptions


def set_node_links_hard(node,link_assumptions, into_node=True):
    if into_node: a,b = '-->','<--'
    else: a,b = '<--','-->'
    
    for i in range(len(link_assumptions)):
        if i != node:
            link_assumptions[node][(i,0)] = a
            link_assumptions[i][(node,0)] = b
    return link_assumptions


def get_fold_rows(df ,max_folds, fold):
    # for each city in df take nth-row defined by fold value
    df_out = pd.DataFrame()
    for city in set(df['city_name']):
        df_city = df.loc[df.city_name==city].reset_index(drop=True)
        df_city = df_city[df_city.index % max_folds == fold]
        df_out = pd.concat([df_out,df_city])
        
    return df_out


def split_folds(df ,max_folds, fold, random_fold=False):
    if isinstance(fold,int): # a fold number is provided 
        if random_fold:
            print(f'Preparing fold for seed {fold}')
            df = df.sample(n=int(len(df)*1/max_folds), random_state=fold)    
        else:
            print(f'Preparing sample for fold {fold}')
            df = get_fold_rows(df, max_folds, fold)
        
    else: # a city name is provided as a fold
        print(f'Preparing sample incl. all cities apart from {fold}')
        df = df.loc[df.city_name!=fold]
    
    return df.reset_index(drop=True)


def balance_city_samples(df, seed = None):
    if seed is None: seed=1

    df = df.dropna()
    samples = df.city_name.value_counts().min()

    df_out = pd.DataFrame()
    for city in set(df['city_name']):
        df_city = df.loc[df.city_name==city].reset_index(drop=True)
        df_city = df_city.sample(n=samples,random_state=seed)
        df_out = pd.concat([df_out,df_city])
    return df_out.reset_index(drop=True)  


def select_causal_features(feature_index, graph, columns):
    target_array = graph[feature_index]
    causal_features = []
    causal_index = []
    for i,el in enumerate(target_array):
        if target_array[i] == ['<--']: # if <-- on target 
            causal_features.append(columns[i])
            causal_index.append(i)
    return causal_index,causal_features


def select_relevant_cols(df, target='distance_m'):
    return df[[target]+[feat for feat in df.columns if 'ft' in feat]].columns


def select_feature_index(col, relevant_cols):
    return list(relevant_cols).index(col)


def get_edge_vals(feature_index, data, causal_ft, causal_indices,city_name):
    plot_data = {'feature':causal_ft}
    data_col = []
    for i in causal_indices:
        data_col.append(data['val_matrix'][feature_index][i][0])

    plot_data[city_name] = data_col
    return pd.DataFrame(plot_data)


def load_causal_features(target, causal_feature_path, city = None):
    # in comparison to select_causal_features, this function first loads
    # the dag file from causal_feature_path, selects the relevant city and
    # then selects the relevant causal features
    
    list_paths = glob.glob(causal_feature_path+'/*.pkl')        
    list_files = [path.rsplit('/',1)[1] for path in list_paths] 
    # for individual cities, every city has its one .pkl file, which we choose based on city name
    if len(list_files)>1: file_name = [file for file in list_files if city in file]
    # with a suammary dag for all cities, only one .pkl file is present
    else: file_name = list_files
    
    if len(file_name)>1:
        raise ValueError(f'Error: Found more than one file')
    
    data = utils.load_pickle(os.path.join(causal_feature_path, file_name[0]))
    relevant_cols = select_relevant_cols(data['df'], target)
    feature_index = select_feature_index(target,relevant_cols)
    _, causal_features =  select_causal_features(feature_index, data['graph'],relevant_cols)
    
    if city is not None:
        print(f'For {city} selected the following causal features:\n{causal_features}')
    else:
        print(f'Selected the following causal features:\n{causal_features}')
    
    return causal_features, data


def find_parents(feature_list, data, relevant_cols):
    all_parents = []
    for ft in feature_list:
        feature_index = select_feature_index(ft,relevant_cols)
        _, parents = select_causal_features(feature_index, data['graph'],relevant_cols)
        if parents:
            all_parents += parents
    return all_parents


def get_dag_chain(causal_features,data,target):
    parent_dict = {}
    dag_chain = []
    parent_dict_cleaned={}

    parent_dict[0] = causal_features
    for i in range(len(causal_features)): 
        # find all parents of all causal ft
        parent_dict[i+1] = find_parents(parent_dict[i], data, [target]+causal_features)
        # remove parents in child list
        parent_dict_cleaned[i] = [p for p in parent_dict[i] if p not in parent_dict[i+1]]
        # remove non causal ft
        parent_dict_cleaned[i] = [p for p in parent_dict_cleaned[i] if p in causal_features]
        
    for i in list(reversed(sorted(parent_dict_cleaned.keys()))): 
        # sort for chain and remove empty 
        if parent_dict_cleaned[i]:
            dag_chain += [parent_dict_cleaned[i]]

    print(f'Assuming the following DAG chain: {dag_chain}')
    return dag_chain


def convert_names(df_merged):
    for old, new in plotting.FEATURE_NAMES.items():
        df_merged['feature'] = df_merged['feature'].str.replace(old, new, regex=False)
        df_merged = df_merged.rename(columns=plotting.CITY_NAMES)
    return df_merged.sort_values('feature', ascending=True)


def translate_var_names(list_var_names):
    return [plotting.FEATURE_NAMES[var] for var in list_var_names]




    
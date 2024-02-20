import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import matplotlib.patches as mpatches
import shap
from pathlib import Path
import numpy as np
import seaborn as sns

import utils.utils_causal as utils_causal
import utils.utils as utils
import utils.utils_ml as utils_ml

FONTSIZE = 35

FEATURE_NAMES = {'distance_m':'VKT',
                'ft_dist_cbd':'Distance to Center',
                'ft_dist_cbd4':'Distance to Center',
                'ft_employment_access_2':'Distance to 20% Employment',
                'ft_employment_access_1':'Distance to 10% Employment',
                'ft_employment_access_01':'Distance to Employment',
                'ft_employment_access_05':'Distance to 5% Employment',
                'ft_income':'Income',
                'ft_income_groups':'Income',
                'ft_income_groups3':'Income',
                'ft_beta':'Street Connectivity',
                'ft_dist_airport':'Airport Access',
                'ft_pop_dense_meta':'Population Density',
                'ft_pop_dense':'Population Density',
                'ft_job_dense': 'Job Density',
                'ft_trips_capita_pop_dense_meta':'Trips per Capita',
                'score_spatiotemporal': 'Transit Access',
                'ft_lu_entropy_normed': 'Land Use Diversity',
                'ft_lu_entropy_classic': 'Land Use Diversity (classic)'
                }           
        
UNIT_NAMES = {'distance_m':'[km]',
            'ft_dist_cbd':'[km]',
            'ft_dist_cbd4':'[km]',
            'ft_employment_access_2':'[km]',
            'ft_employment_access_1':'[km]',
            'ft_employment_access_01':'[km]',
            'ft_employment_access_05':'[km]',
            'ft_income':' ',
            'ft_income_groups':' ',
            'ft_income_groups3':' ',
            'ft_beta':r"[n/km$^2$]",
            'ft_dist_airport':'[km]',
            'ft_pop_dense_meta':r"[n/km$^2$]",
            'ft_pop_dense':r"[n/km$^2$]",
            'ft_job_dense': ' ',
            'ft_trips_capita_pop_dense_meta_orig':' ',
            'score_spatiotemporal':' ',
            'ft_lu_entropy_normed':' ',
            'ft_lu_entropy_classic':' '
            }

CITY_NAMES = {'ber':'Berlin',
              'bos':'Boston',
              'lax':'Los Angeles',
              'sfo':'Bay Area',
              'rio':'Rio de Janeiro',
              'bog':'Bogota'}

COLORS = {'Berlin':'black',
            'Boston':'red',
            'Los Angeles':'brown',
            'Bay Area':'orange',
            'Rio de Janeiro':'blue',
            'Bogota':'green'}

RUN_MAP = {'shap_test': 'shap_ckf', # tree shap,  city kfold
        'causal_shap_test':'cshap_ckf', # causal shap, city kfold
        'shap_values': 'shap_ikf', # shap, indiv fold,
        'causal_shap_values': 'cshap_ikf'} # causal shap, indiv fold



class ShapMaps():

    def __init__(self=None,
                 data = None,
                 shap_type=None,
                 path_root = None,
                 run_name = None,
                 figures = None,
                 title = None,
                 day_hour = None,
                 ):
        self.data = data
        self.shap_type = shap_type
        self.path_root = path_root
        self.run_name = run_name
        self.figures = figures
        self.title = title
        self.day_hour = day_hour

        self.path_out = None
        self.city_folds = None
        self.id = 'tractid'


    def validate_sample_size(self,df_pre,df_post):
        if len(df_pre)!=len(df_post):
            print(f'Lost samples when merging! Pre: {len(df_pre)}, Post: {len(df_post)}')


    def shap_percentages(self):
        self.rel_shap = {}
        for city in self.data.keys():
            abs_shap = []
            for ft in self.data[city][self.shap_type].feature_names:
                abs_shap.append(np.round(np.mean(np.abs(self.data[city][self.shap_type][:,ft].values)),2))

            sum_shap = np.sum(abs_shap)
            
            val_dict = {}
            for i, ft in enumerate(self.data[city][self.shap_type].feature_names):
                val_dict[ft] = int(np.round(100*(abs_shap[i]/sum_shap),2))
            
            self.rel_shap[city] = val_dict



    def shap_geoms_per_city(self, city, od_col='origin', resolution='polygons', bound='fua'):    
        df = self.data[city]['df_test']

        if self.day_hour is not None:
            gdf = utils.read_od(self.path_root,
                                city,
                                utils.get_crs_local(city),
                                self.day_hour,
                                od_col,
                                resolution=resolution,
                                bound=bound)
        else: 
            gdf = utils.init_geoms(self.path_root, city, bound)
            
        shap_vals = {}
        shap_vals['ft'] = [col+'_shap' for col in self.data[city]['X_test'] if 'ft' in col]
        
        # based on 5 city folds
        if self.shap_type in ['shap_test','causal_shap_test']: 
            shap_vals['values'] = self.data[city][self.shap_type].values # getting values out of explainer object
            df_shap = pd.DataFrame(shap_vals['values'],columns = shap_vals['ft'], index=df.index)
        # based on individual city 5 fold
        else: 
            shap_vals['values'] = self.data[city][self.shap_type].values # getting values out of df
            df_shap = self.data[city][self.shap_type].add_suffix('_shap')
        
        df_out = pd.merge(df.reset_index(drop=True),df_shap.reset_index(drop=True), left_index=True, right_index=True)
        gdf_out = pd.merge(gdf, df_out, on = self.id)
        
        self.validate_sample_size(df,gdf_out)
        return gdf_out


    def initialize_shap(self):
        # set marker for city kfold or individual city fold
        self.run_id=RUN_MAP[self.shap_type]
        self.shap_geoms = {}
        
        # prep constants and adjust units
        self.names = FEATURE_NAMES
        self.units = UNIT_NAMES
        self.data = adjust_units(self.data,self.shap_type)
            
        if self.day_hour is not None: # assign old ft naming
            print('Warning! Recognized old run version!')
            self.names = {str(key)+'_orig': val for key, val in self.names.items()}
            self.units = {str(key)+'_orig': val for key, val in self.units.items()}
            self.id='id'

        if any("map" in fig for fig in self.figures):
            for city in self.data.keys():
                print(f'Preparing {city} geoms...')
                self.shap_geoms[city] = self.shap_geoms_per_city(city)

        self.path_out = os.path.join(self.path_root,'5_ml',self.run_name,'plots')
        Path(self.path_out).mkdir(parents=True, exist_ok=True)


    def plot_figures(self, save_fig = False):
        if self.shap_type in ['shap_test','causal_shap_test']: # based on 5 city folds
            if 'map' in self.figures: self.map(save_fig=save_fig)
            if 'map_w_scatter' in self.figures: self.map_w_scatter(save_fig=save_fig)
            if 'map_w_ft_map' in self.figures: self.map_w_ft_map(save_fig=save_fig)
            if 'city_scatter' in self.figures: self.city_scatter(save_fig=save_fig)
            if 'city_individual_scatter' in self.figures: self.city_individual_scatter(save_fig=save_fig)
            if 'beeswarm' in self.figures: self.beeswarm(save_fig=save_fig)            
            if 'shap_comparison' in self.figures: self.shap_comparison(save_fig = save_fig)
            if 'bars' in self.figures: self.shap_bars(save_fig=save_fig)
        else: 
            if 'map' in self.figures: self.map(save_fig=save_fig)
            if 'map_w_ft_map' in self.figures: self.map_w_ft_map(save_fig=save_fig)
            if 'beeswarm' in self.figures: self.beeswarm_legacy(save_fig=save_fig)
            if 'bars' in self.figures: self.shap_bars(save_fig=save_fig)
            

    def map(self, save_fig=False, cmap=shap.plots.colors.red_blue): 
        figname= self.run_id+'_map'

        for city in self.data.keys():        
            gdf = self.shap_geoms[city]
            for col in gdf.columns:
                if 'shap' in col:
                    _,ax = plt.subplots(1,1,figsize=(20,10))
                    
                    #gdf.plot(ax=ax, column=col, cmap=cmap, legend =True, legend_kwds={'shrink': 0.3})
                    gdf.plot(ax=ax, column=col, cmap=cmap, legend =False)
                    ax.axis('off')
                    if self.title: ax.set_title(f"{CITY_NAMES[city]}",fontsize=FONTSIZE) # TODO apply potentialy to all
                    else: ax.set_title(f"{CITY_NAMES[city]}: Shapley Values {self.names[col[:-5]]}, {figname}",fontsize=FONTSIZE)
                    
                    if save_fig: self.save_fig(figname,city,col)
                    else: plt.show()


    def map_w_ft_map(self, save_fig=False, cmap=shap.plots.colors.red_blue):
        figname= self.run_id+'_map_w_ft'

        for city in self.data.keys():
            gdf = self.shap_geoms[city]
            for col in gdf.columns:
                if 'shap' in col:
                    _,ax = plt.subplots(1,2,figsize=(20,10))
                    
                    divnorm_ft=colors.TwoSlopeNorm(vmin=min(gdf[col[:-5]]), vcenter=gdf[col[:-5]].mean(), vmax=max(gdf[col[:-5]]))
                    gdf.plot(ax=ax[0], column=col, cmap=cmap, legend =True, legend_kwds={'shrink': 0.3})
                    gdf.plot(ax=ax[1], column=col[:-5], cmap='coolwarm',norm=divnorm_ft, legend =True, legend_kwds={'shrink': 0.3})
                    
                    ax[0].axis('off')
                    ax[0].set_title(f"{CITY_NAMES[city]}: Shapley Values {self.names[col[:-5]]}, {figname}",fontsize=FONTSIZE)
                    ax[1].axis('off')
                    ax[1].set_title(f'{self.names[col[:-5]]}',fontsize=FONTSIZE)
                    
                    if save_fig: self.save_fig(figname,city,col)
                    else: plt.show()


    def map_w_scatter(self, save_fig=False, cmap=shap.plots.colors.red_blue):
        figname= self.run_id+'_map_w_scatter'
        
        for city in self.data.keys():
            gdf = self.shap_geoms[city]
            explanation_causal = self.data[city][self.shap_type]
            for col in gdf.columns:
                if 'shap' in col:
                    _,ax = plt.subplots(1,2,figsize=(20,10))
                    #divnorm=colors.TwoSlopeNorm(vmin=min(gdf[col]), vcenter=0., vmax=max(gdf[col]))
                    #divnorm=colors.TwoSlopeNorm(vmin=min(gdf[col[:-5]]), vcenter=0., vmax=max(gdf[col[:-5]]))
                    #gdf.plot(ax=ax[0], column=col, cmap='PiYG',norm=divnorm, legend =True)
                    gdf.plot(ax=ax[0], column=col, cmap=cmap, legend =True, legend_kwds={'shrink': 0.3})
                    #shap.plots.scatter(explanation_causal[:,col[0:-5]],color=explanation_causal[:,col[0:-5]],ax = ax[1], show=False, cmap="PiYG")
                    shap.plots.scatter(explanation_causal[:,col[0:-5]],ax = ax[1], show=False)

                    ax[0].axis('off')
                    ax[0].set_title(f"{CITY_NAMES[city]}: Shapley Values {self.names[col[:-5]]}, {figname}",fontsize=FONTSIZE)
                    #ax[1].axis('off')
                    ax[1].set_title(f"{self.feature_names[col[:-5]]} Scatter plot",fontsize=FONTSIZE)
                    
                    if save_fig: self.save_fig(figname,city,col)
                    else: plt.show() 
        

    def city_scatter(self, save_fig=False, fit_line=True):   
        figname= self.run_id+'_city_scatter'
        
        cc, labels, handles = get_plot_args(self.data.keys(), palett='muted')
        #panelname = ['A','B','C','D']
        i=0

        first_key = list(self.data.keys())[0]
        ft_names = self.data[first_key]['X_train'].columns
        
        fig,axs = plt.subplots(ncols = 2, nrows=2, figsize=(18,12))
        for col, ax in zip(ft_names, axs.ravel()):
                
                scatter_kws = {'alpha':1.0, 's':10}
                line_kws = {'alpha':1.0}
                for city in self.data.keys():
                    
                    explanation_causal = self.data[city][self.shap_type].sample(200)
                    vals = [ft[i] for ft in explanation_causal.values]
                    dat = [d[i] for d in explanation_causal.data]
                    
                    if fit_line:
                        scatter_kws['alpha'] = 0.2    
                        sns.regplot(x=dat, y=vals,lowess=True, ax=ax, scatter_kws = scatter_kws, line_kws = line_kws, color = cc[CITY_NAMES[city]])
                    else:
                        sns.regplot(x=dat, y=vals, fit_reg=False, ax=ax, scatter_kws = scatter_kws, color = cc[CITY_NAMES[city]], label=f"{CITY_NAMES[city]}")
                        scatter_kws['alpha'] -= 0.1
                    

                # titles, labels, legends
                if self.title: ax.set_title(f"{self.names[col]}",fontsize=FONTSIZE)
                else: plt.title(f'Scatter Plot: {self.names[col]},{figname}',fontsize=FONTSIZE)
                plt.legend(fontsize=25)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                
                ax.set_xlabel(f"{self.names[col]} [{self.units[col]}]", fontsize=15)
                ax.set_ylabel(f"Causal Shapley Value [km]", fontsize=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # if col == 'ft_pop_dense_meta':
                #     ax.set_xlim(-1000,25*1e3)

                i+=1
                plt.tight_layout() 
        
        fig.legend(handles = handles, labels = labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5,-0.05))               
        
        if save_fig: self.save_fig(figname,'all_cities',col)
        else: plt.show()

    
    def city_individual_scatter(self, save_fig=False, fit_line=False):   
        figname= self.run_id+'_scatter'

        first_key = list(self.data.keys())[0]
        
        for city in self.data.keys():

            fig,ax = plt.subplots(2,2,figsize=(15,10))
            j,k = 0,0
            for i,col in enumerate(self.data[first_key]['X_train'].columns):
                scatter_kws = {'alpha':1}
                color = COLORS[CITY_NAMES[city]]
                
                explanation_causal = self.data[city][self.shap_type].sample(200)
                vals = [ft[i] for ft in explanation_causal.values]
                dat = [d[i] for d in explanation_causal.data]
                sns.regplot(x=dat, y=vals, fit_reg=False, ax=ax[j,k], scatter_kws = scatter_kws, color = color, label=f"{CITY_NAMES[city]}")
                
                # titles, labels, legends
                ax[j,k].set_xlabel(f"{self.names[col]} [{self.units[col]}]")
                ax[j,k].set_ylabel(f"Shapley Value [km]")

                k += 1
                if k>1: 
                    k-=2
                    j=1
            fig.tight_layout()

            # save
            if save_fig: self.save_fig(figname,city,col)
            else: plt.show()


    def shap_bars(self, save_fig=False):
        figname = self.run_id+'_bars'
        
        for city in self.data.keys():
            explanation_causal = self.data[city][self.shap_type]
            explanation_causal = self.rename_fts_in_shap(explanation_causal)
            
            _,ax  = plt.subplots()
            shap.plots.bar(explanation_causal,show=False)
        
            ax.set_title(f"{CITY_NAMES[city]}",fontsize=FONTSIZE)
            plt.gcf().set_size_inches(6,3)
            plt.tight_layout() 
                
            if save_fig: self.save_fig(figname,city)
            else: plt.show()


    def shap_comparison(self, save_fig):
        figname = self.run_id+'_shap_comparison'
        data_tmp = utils.load_pickle('/Users/felix/cluster_remote/p/projects/eubucco/other_projects/urbanformvmt_global/data/5_ml/t25_1st_submission/individual_cities_normed/all_feature/ml_bebobolarisf_pol_XGBRegressor_dis_spNone_ns5600.pkl')
        for city in self.data.keys():
            if 'shap_test' in self.data[city].keys():            
                explanation_causal = self.data[city]['causal_shap_test']
                explanation_causal = self.rename_fts_in_shap(explanation_causal)

                explanation_marginal = data_tmp[city]['shap_test']
                explanation_marginal = self.rename_fts_in_shap(explanation_marginal)
                
                fig = plt.figure()
                ax0 = fig.add_subplot(131)
                ax0.set_title('causal Shap')
                shap.plots.bar(explanation_causal,show=False)
                ax1 = fig.add_subplot(132)
                ax1.set_title('tree Shap')
                shap.plots.bar(explanation_marginal,show=False)

                plt.gcf().set_size_inches(20,3)
                plt.tight_layout() 
                
                if save_fig: self.save_fig(figname,city)
                else: plt.show()


    def rename_fts_in_shap(self, explainer):
        explainer.feature_names = [self.names[ft] for ft in explainer.feature_names] 
        return explainer


    def beeswarm(self, save_fig=False):         
        figname= self.run_id+'_beeswarm'
        
        fig,axs = plt.subplots(ncols = 2, nrows=3,figsize=(13,10))
        for city, ax in zip(self.data.keys(), axs.ravel()):
            explanation_causal = self.data[city][self.shap_type]
            explanation_causal_tmp = self.rename_fts_in_shap(explanation_causal)

            plt.sca(ax)
            shap.summary_plot(explanation_causal_tmp, show =False, plot_size=None,plot_type="bar")

            if self.title: plt.title(f"{CITY_NAMES[city]}")
            else: ax.set_title(f"{CITY_NAMES[city]}, {figname}")
        
        if save_fig: self.save_fig(figname,city)
        else: plt.show()


    def beeswarm_legacy(self, save_fig=False):
        figname= self.run_id+'_beeswarm'

        for city in self.data.keys():
            shap_values = self.data[city][self.shap_type]
            ft_names = [self.names[ft] for ft in self.data[city]['X'].columns]

            _,ax = plt.subplots()
            shap.summary_plot(shap_values.sort_index().to_numpy(),
                             self.data[city]['X'].sort_index(),
                             show =False,
                             feature_names = ft_names)
            plt.gcf().set_size_inches(7,4)
            ax.set_xlim(-1.0, 3.0)
            plt.tight_layout() 
            if self.title: ax.set_title(f"{CITY_NAMES[city]}",fontsize=FONTSIZE)
            else: ax.set_title(f"{CITY_NAMES[city]}, {figname}",fontsize=FONTSIZE)
            
            if save_fig: self.save_fig(figname,city)
            else: plt.show()


    def max_shap_effect(self):     
        # meta
        colors = ['#66c2a5',
            '#fc8d62',
            '#8da0cb',
            '#e78ac3',]
        cmap = matplotlib.colors.ListedColormap(colors)

        # legend
        FONTSIZE = 17
        labels = [FEATURE_NAMES[col] for col in gdf.columns if ('ft_' in col[0:3]) & ('shap' not in col[-4:])]
        handles = []
        for i,ft in enumerate(labels):
                patch = mpatches.Patch(color=colors[i], label=ft)
                handles.append(patch)

        handles.append(mpatches.Patch(facecolor='none',edgecolor='lightgray', hatch='//', label='Excluded'))
        labels.append('Excluded')

        # figure
        fig,axs = plt.subplots(ncols = 2, nrows= 3,figsize=(20,20),gridspec_kw = {'wspace':-0.2, 'hspace':0.1})
        for city, ax in zip(city_dict.keys(), axs.ravel()):
            # prep data per city
            gdf = city_dict[city]
            gdf = gdf.replace({'max_shap_ft_index':FEATURE_NAMES,'min_shap_ft_index':FEATURE_NAMES })
            indx = remove_outliers(gdf,1.96)
            gdf_ = gdf.loc[indx]
            gdf_outliers = gdf.loc[~gdf.index.isin(indx)]
            gdf_pos = gdf_.loc[gdf.shap_pos==True]
            
            # plot
            gdf.exterior.plot(ax=ax,color='black',alpha=0.2, linewidth=0.1)
            gdf_outliers.plot(ax=ax, facecolor = 'none',hatch="//", edgecolor="lightgray", linewidth=0.1)
            gdf_pos.plot(ax=ax,column='max_shap_ft_index', cmap=cmap)

            # individual chart formatting
            ax.set_axis_off()
            ax.set_title(f'{CITY_NAMES[city]}',fontsize=FONTSIZE)

        # figure formating
        fig.legend(handles = handles, labels = labels, loc='lower center', ncol=2, fontsize=FONTSIZE)
        plt.show()



    def save_fig(self,figname, city=None, col=None):
        print(f'Saving {figname} for {city}...')
        if city: figname = figname +'_'+city
        if col: figname = figname + '_'+col
        figname = figname + '.png'
        plt.savefig(os.path.join(self.path_out,figname),bbox_inches='tight')


def generate_df(col, data, city_name):
    relevant_cols = utils_causal.select_relevant_cols(data['df'])
    feature_index = utils_causal.select_feature_index(col,relevant_cols)
    causal_indices, causal_ft = utils_causal.select_causal_features(feature_index, data['graph'],relevant_cols)
    return utils_causal.get_edge_vals(feature_index, data, causal_ft, causal_indices, city_name)


def get_abs(df_merged):
    df_merged.loc[:, df_merged.columns != 'feature'] = df_merged.loc[:, df_merged.columns != 'feature'].abs()
    return df_merged


def create_bar_plot(data, path, col='distance_m'):
    df_merged = pd.DataFrame()
    for city in data.keys():
        df_tmp = utils_causal.generate_df(col, data[city], city) 
        if not df_merged.empty:
            df_merged = pd.merge(df_merged,df_tmp, on='feature',how='outer')
        else:
            df_merged = df_tmp.copy()

    df_merged = utils_causal.convert_names(df_merged)
    df_merged = get_abs(df_merged)

    ax = df_merged.plot.bar(x='feature',color=COLORS)
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    ax.set_ylabel(f"Causal effect (cross MCI) on {names[col]}")
    ax.get_figure().savefig(path,bbox_inches="tight")



def remove_outliers(gdf, num_std=2, **kwargs):
    # removes outliers by 'num_std' standard deviations
    # around error mean
    y_error = gdf['y_predict']-gdf['y_test']
    error_mean = y_error.mean()
    error_std = y_error.std()

    min_bound = error_mean-num_std*error_std
    max_bound = error_mean+num_std*error_std
    y_error_clean = y_error[(y_error>min_bound)&(y_error<max_bound)]

    if kwargs:
        if kwargs['ax'] is not None: ax = kwargs['ax']
        else: fig,ax = plt.subplots()
        binwidth=500
        y_error.plot.hist(ax=ax,bins=np.arange(min(y_error), max(y_error) + binwidth, binwidth),alpha=0.6)
        y_error_clean.plot.hist(ax=ax, bins=np.arange(min(y_error_clean), max(y_error_clean) + binwidth, binwidth), alpha=0.6)
        
        if 'city' in kwargs.keys(): ax.set_title(f"{kwargs['city']} mean: {error_mean:.1f} m, 2x std: {2*error_std:.1f} m")
        else: ax.set_title(f'mean: {error_mean:.1f} m, 2x std: {2*error_std:.1f} m')
    
    return y_error_clean.index


def get_plot_args(cities, palett=None):    
    labels = [CITY_NAMES[city] for city in cities]
    handles = []
    colors = sns.color_palette(palett,n_colors=len(cities))
    cc = {}
    for i,city in enumerate(labels):
        patch = mpatches.Circle((0,0),1,color=colors[i], label=city)
        handles.append(patch)
        cc[city] = colors[i]
    return  cc, labels, handles


def adjust_units(data, shap_type):
    for city in data.keys():
        exp = data[city][shap_type]
        # adjust unit shap vals
        exp.values = exp.values/1000 # to scale to km
        # adjust unit features
        for ft_index, ft in enumerate(exp.feature_names):
            if ('ft_dist_cbd' in ft) or ('ft_employment_access' in ft):
                for i, elem in enumerate(exp.data):
                    exp.data[i][ft_index] = elem[ft_index]/1000
            elif ft in ['ft_pop_dense_meta','ft_pop_dense']:
                for i, elem in enumerate(exp.data):
                    exp.data[i][ft_index] = elem[ft_index]*1e6
        # assign to city
        data[city][shap_type] = exp
    return data
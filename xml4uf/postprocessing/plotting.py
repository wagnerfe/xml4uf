import os,sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import matplotlib.patches as mpatches
import shap
from pathlib import Path
import numpy as np
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from localreg import localreg, rbf

# as jupyter notebook cannot find __file__, import module and submodule path via current_folder
PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, '..','..','..', 'xml4uf'))
sys.path.append(PROJECT_SRC_PATH)

import utils.utils as utils
import utils.utils_ml as utils_ml
from utils.utils import FEATURE_NAMES, CITY_NAMES, UNITS, CO2FACTORS, CITY_AREAS, CITY_UCI


COLORS = {'ber':'blue',
        'bos':'orange',
        'lax':'brown',
        'sfo':'red',
        'rio':'green',
        'bog':'purple'}

RUN_MAP = {'shap_test': 'shap_ckf', # tree shap,  city kfold
            'causal_shap_test':'cshap_ckf', # causal shap, city kfold
            'shap_values': 'shap_ikf', # shap, indiv fold,
            'causal_shap_values': 'cshap_ikf'} # causal shap, indiv fold



class ShapFigures():

    def __init__(self=None,
                 data = None,
                 geoms = None,
                 features = None,
                 cities = None,
                 title = None,
                 fontsize = None,
                 labelsize = None,
                 save_figs = None,
                 shap_cmap = None,
                 city_colors = None,
                 path_out = None
                 ):
        
        self.data = data
        self.geoms = geoms
        self.features = features
        self.cities = cities
        self.title = title
        self.fontsize = fontsize
        self.labelsize = labelsize
        self.save_figs = save_figs
        self.shap_cmap = shap_cmap
        self.city_colors = city_colors
        self.path_out = path_out
        
        self.run_id = None
        self.city_folds = None
        self.id = 'tractid'            


    def map(self, shap_type): 
        figname= self.run_id+'_map'

        for city in self.cities:        
            gdf = self.geoms[city+'_'+shap_type]
            
            vmax = max(gdf[[col+'_co2' for col in self.features]].max())
            vmin = min(gdf[[col+'_co2' for col in self.features]].min())
            divnorm=colors.TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax)
            
            fig,axs = plt.subplots(ncols = 2, nrows=int(len(self.features)/2), figsize=(10,6))
            for col, ax in zip(self.features, axs.ravel()):
                gdf.exterior.plot(ax=ax,color='black',alpha=0.8, linewidth=0.1)
                gdf.plot(ax=ax,
                        column=col+'_co2',
                        legend=True,
                        vmin=vmin,
                        vmax=vmax,
                        cmap='coolwarm',
                        norm = divnorm,
                        legend_kwds={'shrink':0.3,
                                     'label':r'kgCO$_2$ / Trip'})
                
                ax.axis('off')
                ax.set_title(f"{FEATURE_NAMES[col]}",fontsize=self.labelsize)
            plt.suptitle(f"{CITY_NAMES[city]}: {figname}",fontsize=self.fontsize)
            self.save_fig(fig, figname,city)
        

    def _get_smooth_xy(self,
        x,
        y,
        n_points=500,
        method='polynomial',
        lengthscale=1.
        ):

        assert method in ['polynomial', 'gp', 'binning', 'localreg']

        smooth_x = np.linspace(x.min(), sorted(x)[-3], n_points)
        
        if method == 'polynomial':
            # poly = PolynomialFeatures(degree=3).fit_transform(np.atleast_2d(x.values).T)
            poly = PolynomialFeatures(degree=3).fit_transform(np.atleast_2d(x).T)
            model = LinearRegression().fit(poly, y)

            smooth_y = (
                model
                .predict(
                    PolynomialFeatures(degree=3)
                    .fit_transform(
                        smooth_x.reshape(-1, 1)
                    )
                )
            )

        elif method == 'gp':
            
            kernel = (
                ConstantKernel(1.0, constant_value_bounds="fixed") *
                RBF(length_scale=lengthscale, length_scale_bounds="fixed")
            )

            model = GaussianProcessRegressor(kernel=kernel).fit(np.atleast_2d(x).T, y)
            smooth_y = model.predict(np.atleast_2d(smooth_x).T, return_std=False)

        elif method == "binning":

            n_per_bin = 60
            width = 0.08 # percent of distance between max and min
            n_exclude = 5

            window_size = (x.max() - x.min()) * width

            smooth_x = np.linspace(x.min(), sorted(x)[-n_exclude], n_points)
            smooth_y = np.zeros_like(smooth_x)

            for i, x_val in enumerate(smooth_x):
                mask = (x > x_val - window_size/2) & (x < x_val + window_size/2)
                smooth_y[i] = y[mask].mean()

            data = (
                pd.DataFrame({'y': smooth_y, 'x': smooth_x})
                .rolling(n_per_bin).mean().dropna()
            )

            smooth_x = data['x'].values
            smooth_y = data['y'].values

        elif method == "localreg":

            scaler = StandardScaler().fit(np.atleast_2d(x).T)

            smooth_x = np.linspace(x.min(), sorted(x)[-4], n_points)

            prep = lambda x: scaler.transform(np.atleast_2d(x).T).flatten()        
            smooth_y = localreg(prep(x), y, prep(smooth_x), degree=1, kernel=rbf.tricube, radius=2)

        else:
            raise NotImplementedError

        return smooth_x, smooth_y


    def _remove_x_perc(self,dat, cut_line):
        if cut_line:
            two_perc = len(dat)-int(0.99*len(dat))
            idx_bot = np.argsort(dat)[:two_perc]
            idx_top = np.argpartition(dat, -two_perc)[-two_perc:]
            idx_remove = np.append(idx_bot,idx_top)
            return ~np.isin(np.arange(len(dat)),idx_remove)
        else:
            return slice(None)


    def city_scatter(self,
                    shap_type,
                    scatter_kwargs,
                    plot_kwargs,
                    legend_kwargs,
                    cut_line=False,
                    ):
        
        figname = self.run_id+'_scatter'
        method = scatter_kwargs.pop('method')
        cut_line = scatter_kwargs.pop('cut_line')
        ymin = scatter_kwargs.pop('ymin')
        ymax = scatter_kwargs.pop('ymax')
        
        i=0
        fig,axs = plt.subplots(ncols = 2, nrows=2, figsize=(10,6))
        for col, ax in zip(self.features, axs.ravel()):
            max_x = 0
            feature_index = self.features.index(col)
            for city in self.cities:
                vals = self.data[city][shap_type].values[:,feature_index]*CO2FACTORS[city]/1000
                dat = self.data[city][shap_type].data[:,feature_index]

                mask_cut = self._remove_x_perc(dat,cut_line)

                smooth_x, smooth_y = self._get_smooth_xy(dat[mask_cut],
                                                vals[mask_cut],
                                                method=method,
                                                lengthscale=max(dat[mask_cut])*3)

                #idx = np.random.choice(np.arange(len(dat)), 100, replace=False)
                idx = slice(None)
                ax.scatter(dat[idx], vals[idx], color=legend_kwargs['colors'][CITY_NAMES[city]], **scatter_kwargs)
                ax.plot(smooth_x, smooth_y, color=legend_kwargs['colors'][CITY_NAMES[city]], label=CITY_NAMES[city], **plot_kwargs)
                
                if max(dat)>max_x: max_x = max(dat)

            # ax visuals
            ax.hlines(y=0, xmin=0, xmax=max_x, color='gray', linestyle='dashed', linewidth=1) 
            ax.tick_params(axis='both', which='major', labelsize=self.labelsize, colors='gray')
            ax.set_xlabel(fr"{FEATURE_NAMES[col]} {UNITS[col]}",fontsize=self.labelsize)
            ax.set_ylabel(fr"Feature effect [kgCO$_2$/Trip]", fontsize=self.labelsize)
            ax.set_facecolor('white')
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_visible(True)
            ax.spines[['left','bottom']].set_color('black')
            ax.tick_params(axis='x', colors='black')
            ax.tick_params(axis='y', colors='black')
            
            # ax lims
            if col == 'ft_pop_dense': ax.set_xlim(-1000,30*1e3)    
            if col == 'ft_beta': ax.set_xlim(-10,300)    
            ax.set_ylim(ymin,ymax)
            
            i+=1
            plt.tight_layout() 
            
            # ax texts
            # axs[0,0].text(100, 1100, "(A)", fontsize=20)
            # axs[0,1].text(35, 1100, "(B)", fontsize=20)
            # axs[1,0].text(26000, 1100, "(C)", fontsize=20)
            # axs[1,1].text(530, 1100, "(D)", fontsize=20)

            for ax in axs.flatten():
                ax.grid(linestyle='dashed')

        fig.legend(handles = legend_kwargs['handles'], labels = legend_kwargs['labels'], loc='lower center', ncol=6, bbox_to_anchor=(0.5,-0.11),facecolor='white')
        if self.title is not None: suptitle = self.title+':'+figname
        else: suptitle = figname
        fig.suptitle(suptitle,y=1.02,fontsize = self.fontsize)
        self.save_fig(fig, figname)

    
    def individual_scatter(self, shap_type, scatter_kwargs):   
        figname = self.run_id+'_scatter'
        
        for city in self.cities:
            fig,axs = plt.subplots(ncols = 2, nrows=int(len(self.features)/2), figsize=(10,6))
            for col, ax in zip(self.features, axs.ravel()):                
                feature_index = self.features.index(col)
                explanation = self.data[city][shap_type].sample(int(len(self.data[city][shap_type].values)*0.5))
                vals = explanation.values[:,feature_index]*CO2FACTORS[city]/1000
                dat = explanation.data[:,feature_index]
                
                sns.regplot(x=dat,
                            y=vals,
                            fit_reg=False,
                            ax=ax,
                            scatter_kws = {'alpha':1,
                                            'marker':scatter_kwargs['marker'], 
                                            's':scatter_kwargs['s']},
                            color = self.city_colors[city],
                            label=f"{CITY_NAMES[city]}")
                
                # ax
                ax.spines[['top','right']].set_visible(False)
                ax.set_xlabel(f"{FEATURE_NAMES[col]}", fontsize = self.labelsize)
                ax.set_ylabel(f"Feature effect [kgCO$_2$/Trip]",fontsize = self.labelsize)
                ax.set_ylim(scatter_kwargs['ymin'],scatter_kwargs['ymax'])

            plt.suptitle(f'{CITY_NAMES[city]},{figname}',fontsize=self.fontsize)
            self.save_fig(fig, figname,city)


    def city_bars(self, shap_type, legend_kwargs):
        figname = self.run_id+'_city_bars'

        def prep_bar_plot(data,city,shap_type):
            explanation_causal = data[city][shap_type]
            df_shap_vals = pd.DataFrame(explanation_causal.values)
            df_shap_vals.columns = [ft for ft in explanation_causal.feature_names]
            df_shap = pd.DataFrame(abs(df_shap_vals).mean(),columns=['Average feature effect [km]'])
            df_shap = df_shap.reset_index().rename(columns={'index':'Feature'})
            return df_shap
        
        df_out = pd.DataFrame()
        for city in self.cities:
            df_tmp = prep_bar_plot(self.data, city, shap_type)
            df_tmp['Relative Feature Effect [%]'] = df_tmp['Average feature effect [km]']/df_tmp['Average feature effect [km]'].sum()
            df_tmp['city'] = CITY_NAMES[city]
            df_out = pd.concat([df_out, df_tmp])

        fig,ax = plt.subplots(1,figsize=(8,5))
        sns.barplot(ax=ax,
                    data=df_out,
                    y='Relative Feature Effect [%]',
                    x='Feature',
                    hue='city',
                    palette=sns.color_palette([self.city_colors[c] for c in self.cities]))
        
        #ax.get_legend().set_visible(False)
        ax.set_title(f'Relative feature importance, {figname}')
        ax.spines[['top','right']].set_visible(False)
        #fig.legend(handles = legend_kwargs['handles'], labels = legend_kwargs['labels'], loc='center', ncol=6, bbox_to_anchor=(0.6,-0.05))
        
        self.save_fig(fig, figname)


    def _2ft_scatter(self, shap_type, city, fts, axi, scatter_kwargs,plot_kwargs,method, cut_line):
        
        max_x = 0
        # val_min = 0
        c = {fts[0]:'#66c2a5',
            fts[1]: '#fc8d62',}

        for col in fts:
            explanation = self.data[city][shap_type]
            vals = explanation.values[:,self.features.index(col)]*CO2FACTORS[city]/1000    
            dat = explanation.data[:,self.features.index(fts[0])] # we always plot against 1st ft of fts
            
            mask_cut = self._remove_x_perc(dat, cut_line)
            smooth_x, smooth_y = self._get_smooth_xy(dat[mask_cut],
                                            vals[mask_cut],
                                            method=method,
                                            lengthscale=max(dat[mask_cut])*3)

            axi.scatter(dat, vals, **scatter_kwargs, color=c[col])
            axi.plot(smooth_x, smooth_y, label=CITY_NAMES[city], **plot_kwargs, color=c[col])

            if max(dat)>max_x: # for hline
                max_x = max(dat)

            # if vals.min()<val_min:
            #     val_min = vals.min()
        axi.hlines(y=0, xmin=0, xmax=max_x, color='gray', linestyle='dashed', linewidth=1) 
        axi.spines[['top','right']].set_visible(False)
        axi.set_xlabel(fr"{FEATURE_NAMES[fts[0]]} [km]", fontsize=self.labelsize)
        axi.set_ylabel(fr"[kgCO$_2$/Trip]", fontsize=self.labelsize)


    def _get_min_max_shap_cbd_density(self, shap_type, city, fts):
        print(f'finding min max shap for {city}')
        gdf = self.geoms[city+'_'+shap_type] 
        fts_tmp = [ft+'_shap' for ft in fts]

        gdf['min_shap_ft_index'] = gdf[fts_tmp].idxmin(axis=1)
        gdf['max_shap_ft_index'] = gdf[fts_tmp].idxmax(axis=1)
        gdf['min_shap_ft'] = gdf[fts_tmp].min(axis=1)
        gdf['max_shap_ft'] = gdf[fts_tmp].max(axis=1)
        gdf['shap_pos'] = (gdf[fts_tmp]>=0).any(axis=1)
        gdf['shap_neg'] = (gdf[fts_tmp]<0).any(axis=1)

        # clean column names
        gdf['max_shap_ft_index'] = gdf['max_shap_ft_index'].str.replace('_shap', '')
        gdf['min_shap_ft_index'] = gdf['min_shap_ft_index'].str.replace('_shap', '')
        return gdf.replace({'max_shap_ft_index':FEATURE_NAMES,'min_shap_ft_index':FEATURE_NAMES }) 

        


    def _map(self,shap_type, city, fts, axj, mean_dist):
        cmap = matplotlib.colors.ListedColormap(['#66c2a5','#fc8d62'])
        gdf = self._get_min_max_shap_cbd_density(shap_type,city,fts)
        gdf_in = gdf.loc[gdf.y_test>=mean_dist]
        #gdf_buff2 = gdf_buff.copy(deep=True)

        # plot
        self.geoms[city+'_'+shap_type].plot(ax=axj, color='white', alpha=0.6)
        gdf.plot(ax=axj, facecolor = 'none',hatch="//", edgecolor='lightgray', linewidth=0.01)
        gdf_in.plot(ax=axj,column='max_shap_ft_index', cmap=cmap,legend=False)
        #gdf_buff.plot(ax=axj,marker="*",markersize=90,color='gray')


        # formatting
        axj.set_axis_off()
        axj.set_title(CITY_NAMES[city])


    def map_scatter_corridors(self, 
                            shap_type,
                            scatter_kwargs,
                            plot_kwargs,
                            ft0='ft_dist_cbd',
                            ft1='ft_pop_dense',
                            ):
        
        figname = self.run_id + '_corridors'
        method = scatter_kwargs.pop('method')
        cut_line = scatter_kwargs.pop('cut_line')
        _ = scatter_kwargs.pop('ymin')
        _ = scatter_kwargs.pop('ymax')

        fig,ax = plt.subplots(ncols = 2, nrows=len(self.cities), figsize=(10,10),gridspec_kw = {'wspace':0, 'hspace':0.5})
        for i,city in enumerate(self.cities):    
            self._map(shap_type, 
                    city,
                    [ft0,ft1],
                    ax[i,0],
                    self.data[city][shap_type].base_values[0]
                    )
            
            self._2ft_scatter(shap_type,
                        city,
                        [ft0,ft1],
                        ax[i,1],
                        scatter_kwargs,
                        plot_kwargs,
                        method,
                        cut_line
                        )
            
        if self.title is not None: suptitle = self.title+':'+figname
        else: suptitle = figname
        fig.suptitle(suptitle,fontsize=self.fontsize,y=0.995)
        
        self.save_fig(fig, figname)


    def bars(self, shap_type):
        figname = self.run_id+'_bars'

        fig,axs = plt.subplots(ncols = 2, nrows=3,figsize=(13,10))
        for city, ax in zip(self.cities, axs.ravel()):
            explanation = self.data[city][shap_type]
            plt.sca(ax)
            shap.summary_plot(explanation,
                            show =False,
                            plot_size=None,
                            color = self.city_colors[city],
                            plot_type="bar")
            plt.locator_params(nbins=4)
            ax.set_title(f"{CITY_NAMES[city]}")
        
        if self.title is not None: suptitle = self.title+':'+figname
        else: suptitle = figname
        fig.suptitle(suptitle,fontsize=self.fontsize,y=0.995)
        self.save_fig(fig, figname, city)


    def beeswarm(self, shap_type):         
        figname= self.run_id+'_beeswarm'
        
        fig,axs = plt.subplots(ncols = 2, nrows=3,figsize=(13,10))
        for city, ax in zip(self.cities, axs.ravel()):
            explanation = self.data[city][shap_type]
            plt.sca(ax)
            shap.summary_plot(explanation, show =False, plot_size=None,plot_type="dot", cmap=self.shap_cmap)

            ax.set_title(f"{CITY_NAMES[city]}")
        if self.title is not None: suptitle = self.title+':'+figname
        else: suptitle = figname
        fig.suptitle(suptitle,fontsize=self.fontsize,y=0.995)
        self.save_fig(fig, figname)

    # def shap_comparison(self): # TODO
    #     data_tmp = utils.load_pickle('/Users/felix/cluster_remote/p/projects/eubucco/other_projects/urbanformvmt_global/data/5_ml/t25_1st_submission/individual_cities_normed/all_feature/ml_bebobolarisf_pol_XGBRegressor_dis_spNone_ns5600.pkl')
    #     for city in self.cities:
    #         if 'shap_test' in self.data[city].keys():            
    #             explanation_causal = self.data[city]['causal_shap_test']
    #             explanation_causal = self.rename_fts_in_shap(explanation_causal)

    #             explanation_marginal = data_tmp[city]['shap_test']
    #             explanation_marginal = self.rename_fts_in_shap(explanation_marginal)
                
    #             fig = plt.figure()
    #             ax0 = fig.add_subplot(131)
    #             ax0.set_title('causal Shap')
    #             shap.plots.bar(explanation_causal,show=False)
    #             ax1 = fig.add_subplot(132)
    #             ax1.set_title('tree Shap')
    #             shap.plots.bar(explanation_marginal,show=False)

    #             plt.gcf().set_size_inches(20,3)
    #             plt.tight_layout() 
    #             if not self.save_fig: plt.show()


    # def map_w_ft_map(self, shap_type):
    #     figname= self.run_id+'_map_w_ft'

    #     for city in self.cities:
    #         gdf = self.geoms[city+'_'+shap_type]
    #         for col in gdf.columns:
    #             if 'shap' in col:
    #                 _,ax = plt.subplots(1,2,figsize=(20,10))
                    
    #                 divnorm_ft=colors.TwoSlopeNorm(vmin=min(gdf[col[:-5]]), vcenter=gdf[col[:-5]].mean(), vmax=max(gdf[col[:-5]]))
    #                 gdf.plot(ax=ax[0], column=col, cmap=self.shap_cmap, legend =True, legend_kwds={'shrink': 0.3})
    #                 gdf.plot(ax=ax[1], column=col[:-5], cmap='coolwarm',norm=divnorm_ft, legend =True, legend_kwds={'shrink': 0.3})
                    
    #                 ax[0].axis('off')
    #                 ax[1].axis('off')
    #                 plt.suptitle(f"{CITY_NAMES[city]}: {col[:-5]}, {figname}",fontsize=self.fontsize)
    #                 self.save_fig(fig, figname,city,col)   


    def save_fig(self,fig, name, city=None, col=None):
        if self.save_figs:
            print(f'Saving {name}')
            if city: name = name +'_'+city
            if col: name = name + '_'+col # TODO?
            if self.title is not None: name = name + '_' + self.title
            name += '.png'
            fig.savefig(os.path.join(self.path_out,name),bbox_inches='tight',dpi=300)
        else:
            plt.tight_layout
            plt.show()  


class PlotManager():
    
    def __init__(self=None,
                run_name = None,
                path_root = None,
                path_out = None,
                data = None,
                cities = None,
                figures = None,
                shap_type=None,
                shap_cmap = None,
                title = None,
                save_figs = None,
                ):
        
        # vars
        self.data = data    
        if type(shap_type) == list:self.shap_type = shap_type
        else: self.shap_type = [shap_type]
        self.cities = cities
        self.figures = figures
        self.title = title
        self.save_figs = save_figs
        self.geoms = None

        # paths
        self.path_root = path_root
        self.path_out = None
        if self.save_figs:
            if path_out is not None:
                self.path_out = path_out
            else:
                self.path_out = os.path.join(path_root,'5_ml',run_name,'plots')
                Path(self.path_out).mkdir(parents=True, exist_ok=True)    
        
        # naming 
        self.run_name = run_name
        self.figname = ['A','B','C','D']
        
        # fontsizes
        self.fontsize = 15
        self.labelsize = self.fontsize-4

        # coloring
        self.palette = ['#66c2a5', # TODO only works for 4 features - adjust
                    '#fc8d62',
                    '#8da0cb',
                    '#e78ac3',]
        self.feature_colors = sns.color_palette(self.palette, n_colors=4)
        self.city_colors = COLORS
        if shap_cmap is not None: self.shap_cmap = shap_cmap
        else: self.shap_cmap = shap.plots.colors.red_blue

        # scatter kwargs
        self.scatter_kwargs = {'method': 'localreg', # 'gp', 'polynomial','binning'
                                'cut_line':True,
                                'alpha': 0.6,
                                's': 2,
                                'marker': '.',
                                'ymin':-0.5,
                                'ymax':2.3,
                                }
        self.plot_kwargs = {'linewidth': 1}


    def adjust_units(self,data):
        for shap_type in self.shap_type:
            for city in data.keys():
                exp = data[city][shap_type]
                exp.values = exp.values/1000 # to scale to km
                for ft_index, ft in enumerate(exp.feature_names):
                    if ('ft_dist_cbd' in ft) or ('ft_employment_access' in ft):
                        for i, elem in enumerate(exp.data):
                            exp.data[i][ft_index] = elem[ft_index]/1000
                    elif ft in ['ft_pop_dense_meta','ft_pop_dense']:
                        for i, elem in enumerate(exp.data):
                            exp.data[i][ft_index] = elem[ft_index]
                # assign to city
                data[city][shap_type] = exp
        return data
    

    def shap_percentages(self):
        self.rel_shap = {}
        for city in self.cities:
            abs_shap = []
            for ft in self.data[city][self.shap_type].feature_names:
                abs_shap.append(np.round(np.mean(np.abs(self.data[city][self.shap_type][:,ft].values)),2))

            sum_shap = np.sum(abs_shap)
            
            val_dict = {}
            for i, ft in enumerate(self.data[city][self.shap_type].feature_names):
                val_dict[ft] = int(np.round(100*(abs_shap[i]/sum_shap),2))
            
            self.rel_shap[city] = val_dict


    def rename_fts_in_shap(self):
        # renames features in explainer objects for visualization
        for shap_type in self.shap_type:
            for city in self.cities:
                explainer = self.data[city][shap_type]
                explainer.feature_names = [FEATURE_NAMES[ft] for ft in explainer.feature_names] 
                self.data[city][shap_type] = explainer


    def validate_sample_size(self,df_pre,df_post):
        if len(df_pre)!=len(df_post):
            print(f'Lost samples when merging! Pre: {len(df_pre)}, Post: {len(df_post)}')


    def assign_geoms_to_shap(self):    
        geoms = {}
        for shap_type in self.shap_type:
            for city in self.cities:
                print(f'Preparing {city} geoms...')
                df = self.data[city]['df_test'].reset_index(drop=True)
                df_merge = pd.merge(df[['tractid']],self.data[city]['df_rescaled'][['y_test']],left_index=True,right_index=True)

                gdf = utils.init_geoms(self.path_root, city, bound='fua')
                    
                shap_vals = {}
                shap_vals['ft'] = [col+'_shap' for col in self.data[city]['X_test'] if 'ft' in col]
                
                # based on 5 city folds
                shap_vals['values'] = self.data[city][self.shap_type[0]].values 
                df_shap = pd.DataFrame(shap_vals['values'],columns = shap_vals['ft'], index=df_merge.index)
            
                df_out = pd.merge(df_merge,df_shap, left_index=True, right_index=True)
                gdf_out = pd.merge(gdf, df_out, on = 'tractid')

                # also add co2 values to gdf_out
                for ft in self.features:
                    gdf_out[ft+'_co2'] = gdf_out[ft+'_shap']*CO2FACTORS[city]/1000 #kgCO2eq/km
                
                self.validate_sample_size(df,gdf_out)
                geoms[city+'_'+shap_type] = gdf_out
        
        return geoms


    def get_city_stats(self):
        self.stats = {}
        for city in self.cities:
            city_stats = {}
            city_stats['centrality'] = CITY_UCI[city]
            city_stats['area'] = CITY_AREAS[city]
            city_stats['co2'] = np.mean(self.data[city]['df_rescaled'].y_test/1000*CO2FACTORS[city])/1000
            city_stats['r2'] = np.round(self.data[city]['r2_pred'],2)
            self.stats[city] = city_stats


    def get_args(self):  
        
        def get_label(city, city_stats):
            city_name = CITY_NAMES[city].replace(" ", "\ ")
            return ("$\mathbf{" + city_name + f"}}\ $"+"\n"+r"$\overline{CO}$$_2$"+f": {city_stats[city]['co2']:.2f}\nUCI: {city_stats[city]['centrality']:.2f}")

        self.get_city_stats()
        labels = [get_label(city,self.stats) for city in self.cities]
        handles = []
        #colors = sns.color_palette(self.city_colors.values,n_colors=len(self.cities))
        cc = {}
        for city in self.cities:
            patch = mpatches.Circle((0,0),1,color=self.city_colors[city], label=CITY_NAMES[city])
            handles.append(patch)
            cc[CITY_NAMES[city]] = self.city_colors[city]
        return  {'colors':cc, 'labels':labels,'handles':handles}


    def get_feature_names(self):
        first_key = list(self.cities)[0]
        return self.data[first_key]['X_train'].columns.tolist()


    def get_cities_ordered(self):
        cities = self.data.keys()
        # get uci vals for cities as subcet (to handle also less than all cities)
        city_uci_tmp = {x: CITY_UCI[x] for x in CITY_UCI.keys() if x in cities}
        # sorts based on uci ascending order
        return sorted(city_uci_tmp, key=city_uci_tmp.get,reverse=True)
        

    def initialize(self, mounted):
        # check if run locally
        if (self.data is None) & (mounted):
            path_files = os.path.join(self.path_root,'5_ml',self.run_name)
            run_pkl = glob.glob(path_files+'/*.pkl')
            if len(run_pkl)==1:
                self.data = utils.load_pickle(run_pkl[0])
            else: raise ValueError(f'No or several datasets founds: {run_pkl}')

        # rescale explainern
        self.data = utils_ml.get_rescaled_explainer(self.data, 'causal_shap_test') # TODO ony causal shap supported

        # units
        self.data = self.adjust_units(self.data)

        # get city and feature names
        if self.cities is None:
            self.cities = self.get_cities_ordered()
        self.features = self.get_feature_names()
        self.rename_fts_in_shap()

        # get geoms for map plots
        if any("map" in fig for fig in self.figures):
            self.geoms = self.assign_geoms_to_shap()

        # get colors and handles
        self.legend_kwargs = self.get_args()


    def plotting(self,sf,fig_type, shap_type):
        if fig_type=='map': sf.map(shap_type)
        if fig_type=='city_scatter': sf.city_scatter(shap_type,
                                            self.scatter_kwargs,
                                            self.plot_kwargs,
                                            self.legend_kwargs)
        if fig_type=='individual_scatter': sf.individual_scatter(shap_type,self.scatter_kwargs)
        if fig_type=='beeswarm': sf.beeswarm(shap_type)            
        if fig_type == 'city_bars': sf.city_bars(shap_type,self.legend_kwargs)
        if fig_type == 'map_scatter_corridors': sf.map_scatter_corridors(shap_type,                                                            
                                                                    self.scatter_kwargs,
                                                                    self.plot_kwargs,
                                                                    ft0='ft_dist_cbd',
                                                                    ft1='ft_pop_dense',)
        if fig_type=='bars': sf.bars(shap_type)
        #if fig_type=='shap_comparison': sf.shap_comparison(shap_type)


    # def init_fig(self,fig):
    #     if fig in ['beeswarm', 'bars']:
    #         return plt.subplots(ncols = 2, nrows=3,figsize=(13,10))     


    def create(self,mounted):
        
        self.initialize(mounted)
        sf = ShapFigures(data=self.data,
                         geoms = self.geoms,
                         features=self.features,
                         cities = self.cities,
                         title = self.title,
                         fontsize = self.fontsize,
                         labelsize = self.labelsize,
                         save_figs=self.save_figs,
                         shap_cmap = self.shap_cmap,
                         city_colors= self.city_colors,
                         path_out=self.path_out,
                         )

        for shap_type in self.shap_type:
            # run_id contains abbreviated shap and fold type for file storage
            sf.run_id = RUN_MAP[shap_type]
            for fig_type in self.figures:
                self.plotting(sf, fig_type, shap_type)



def main():
    
    # allows to run plotting locally w.o. executing ml.py before:
    mounted, _ = utils.check_if_mounted()

    if mounted:
        request = utils.get_input(PROJECT_SRC_PATH,'postprocessing/plotting.yml')
        pm = PlotManager(**request)
        pm.create(mounted)    

if __name__ == "__main__":
    main() 
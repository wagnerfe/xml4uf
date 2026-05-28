
# XML4UF
This is the code repository for the publication **Refining urban typologies: Causal insights into urban form, car commuting, and related CO2 emissions** by Wagner et al (2026) ([link to paper](https://iopscience.iop.org/article/10.1088/1748-9326/ae6881)). The paper utilises causal/e**X**plainable **ML** for **U**rban **F**orm analysis (XML4UF). It combines urban form and mobility data from 6 cities including Berlin, Los Angeles, Bay Area, Boston, Rio de Janeiro and Bogota and applies causal graph discovery, supervised machine learning and causal shapley value analysis. This repository summarises the code used for the publication and aims at creating an overview as well as an introduction to causal graph discovery and ML. 

## Abstract
Urban transport is a major source of greenhouse gas emissions, making effective urban planning crucial for climate mitigation. Global typologies of cities can help to scale planning strategies, yet they hardly capture how interventions translate into local contexts. Big urban data, combined with artificial intelligence, holds great potential to facilitate scalable yet location-specific planning to reduce urban travel and related emissions. However, current research falls short in recognizing underlying variable dependencies, understanding neighborhood-specific differences, and comparing relationships across world regions. Here, we present a systematic quantitative analysis of how urban form influences car-based work trip distance and related emissions in six cities on three continents. We integrate causal discovery and explainable machine learning and apply it to a sample of 10 million mobility data points derived from GPS and call detail records. We find significant direct dependencies between urban form and car-based work trip distance, neglected in previous research. Across cities, geographic access to city center and employment matters more than density or connectivity, yet the magnitude of such effects and their spatial distribution vary depending on a city’s size and centrality. Compact central development appears most effective, while our approach identifies suburban corridors up to 40 km from the center where densification offers additional mitigation potential. In more polycentric cities, subcenter development provides further leverage. Our results contribute high-resolution comparative evidence to refine urban typologies for effective emission reduction in the context of car-based work travel.

## System requirements and installation
The code is run and tested on the PIK HLRS2015 computing cluster, using all software dependencies as described in `environment.yml`. The computing cluster uses a slurm workload manager, which is why we use a [python wrapper](https://github.com/ai4up/slurm-pipeline)and split the code in .yml files and .py files. When using the slurm workload manager, you need to create an additional `slurm_config.yml` file as described in the respective git. However, running the code locally is also possible, via specifying the appropriate path in `bin/env_config.yml`. 

In addition, you need to create the following folder structure in order to run the code with your own data successfully:
- `data/`
	- `0_raw_data` here, you place all downloaded raw data
	- `1_preprocessed_data`
	- `2_cleaned_data`
	- `3_features`
	- `4_causal_inference`
	- `5_ml`

## Overview
The following describes each folder of the repository in more detial:	
 - `bin` all .yml files defining run parameters
 - `submodules`: all submodules used in the pipeline
 - `xml4uf`: python code for the pipeline
 - `notebooks`: notebooks containing examples of individual steps:
     - `feature_engineering.ipynb` provides an overview of how features are calculated
	 - `dag_discovery.ipynb` provides an overview of the causal graph discovery
	 - `ml.ipynb` provides an overview of the machine learning pipeline 
	 - `sample_data.pkl` is an anonymised example data set 








 

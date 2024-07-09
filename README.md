
# XML4UF
This is the code repository for the research on using causal/e**X**plainable **ML** for **U**rban **F**orm analysis (XML4UF). It combines urban form and mobility data from 6 cities including Berlin, Los Angeles, Bay Area, Boston, Rio de Janeiro and Bogota and applies causal graph discovery, supervised machine learning and causal shapley value analysis. This repository summarises the code used for the publication and aims at creating an overview as well as an introduction to causal graph discovery and ML. A preprint of the publication titled [Causal relationships between urban form and travel CO2 emissions across three continents.](https://arxiv.org/abs/2308.16599) can be found on arxiv.

## Abstract
To reduce urban transport emissions globally, effective planning of urban form is primordial. Big urban data combined with artificial intelligence-based methods bear the potential for scalable analyses that nonetheless reflect the varying characteristics of different cities, facilitating the transfer of locally applicable planning strategies. However, current research falls short in utilizing this potential at three levels: (1) Causality – Can causality be established beyond theoretical and correlation-based analyses? (2) Context specificity – How do the effects of urban form on travel vary across neighborhoods in a city? (3) Generalizability – Do relationships hold across different cities and world regions? Here, we address all three gaps via causal graph discovery and explainable machine learning, using 10 million mobility data points from six cities across three continents, to assess the impact of urban form on intra-city car travel. We find significant cause-effect relationships of urban form on trip emissions and inter-feature relationships, which had been neglected in previous work. Across cities we find that high access matters more than high density or street connectivity, while effect magnitudes and locations vary depending on a city’s centrality and size. Besides, we identify city-specific suburban neighborhoods within 20-40 km distance from the center that benefit more from densification than higher access, highlighting the role of subcenter development. Our work provides timely methodological and practical perspectives on a long-standing academic debate of high relevance for low-carbon urban futures.

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








 

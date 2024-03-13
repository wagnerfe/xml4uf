
# XML4UF
This is the code repository for the research on using causal/e**X**plainable **ML** for **U**rban **F**orm analysis (XML4UF). It combines urban form and mobility data from 6 cities including Berlin, Los Angeles, Bay Area, Boston, Rio de Janeiro and Bogota and applies causal graph discovery, supervised machine learning and causal shapley value analysis. This repository summarises the code used for the publication and aims at creating an overview as well as an introduction to causal graph discovery and ML. A preprint of the publication titled [Using machine learning to understand causal relationships between urban form and travel CO2 emissions across continents](https://arxiv.org/abs/2308.16599) can be found on arxiv.

## Abstract
Climate change mitigation in urban mobility requires policies reconfiguring urban form to increase accessibility and facilitate low-carbon modes of transport. However, current policy research has insufficiently assessed urban form effects on car travel at three levels: (1) Causality -- Can causality be established beyond theoretical and correlation-based analyses? (2) Generalizability -- Do relationships hold across different cities and world regions? (3) Context specificity -- How do relationships vary across neighborhoods of a city? Here, we address all three gaps via causal graph discovery and explainable machine learning to detect urban form effects on intra-city car travel, based on mobility data of six cities across three continents. We find significant causal effects of urban form on trip emissions and inter-feature effects, which had been neglected in previous work. Our results demonstrate that destination accessibility matters most overall, while low density and low connectivity also sharply increase CO2 emissions. These general trends are similar across cities but we find idiosyncratic effects that can lead to substantially different recommendations. In more monocentric cities, we identify spatial corridors -- about 10--50 km from the city center -- where subcenter-oriented development is more relevant than increased access to the main center. Our work demonstrates a novel application of machine learning that enables new research addressing the needs of causality, generalizability, and contextual specificity for scaling evidence-based urban climate solutions.

## Overview
The code is run on a computing cluster using slurm workload manager and a [python wrapper](https://github.com/ai4up/slurm-pipeline), which is why we split the code in .yml files and .py files. When using the slurm workload manager, you need to create an additional `slurm_config.yml` file as described in the respective git. However, running the code locally is also possible, via specifying the appropriate path in `bin/env_config.yml`. 

 - `bin` all .yml files defining run parameters
 - `submodules`: all submodules used in the pipeline
 - `xml4uf`: python code for the pipeline
  - `notebooks`: notebooks containing examples of individual steps:
		 - `feature_engineering.ipynb` provides an overview of how features are calculated
		 - `dag_discovery.ipynb` provides an overview of the causal graph discovery
		 - `ml.ipynb` (TBD) provides an overview of the machine learning pipeline 
		 - `sample_data.pkl` is an anonymised example data set 






 

A DEDUPLICATION PIPELINE TO CLEAN AND REMOVE DUPLICATES IN FSQ PLACES DATA

Pipeline Structure:
├── data
│   ├── blacklists                                  <- Contain blacklists for specific FSQ datasets
│   ├── fsq_data                                    <- FSQ Datasets for specific cities and states
    ├── data_setup.ipynb                            <-               
│
├── d02_notebooks                                 <- Jupyter notebooks that produce plots used in the submission
│   ├── 1_preprocessing_311_complaints.ipynb      <- Basic EDA of 311 flood reports
│   ├── 2_preprocessing_demographic_data.ipynb    <- Shows processing of Census and ACS data into demographic features
│   ├── 3_model_example.ipynb                     <- Read inference on real world summaries and produce visualizations
│   └── 4_producing_results.ipynb                 <- Read semi-synthetic simulation summaries and produce visualizations
│
├── d03_src                                       <- Source code for use in this project, which can be imported as modules into the notebooks and scripts
│   ├── model_*.py                                <- Model functions     
│   ├── evaluate_*.py                             <- Misc. functions for reading model results and visualization
│   ├── process_*.py                              <- Misc. functions for data pre-processing
│   └── vars.py                                   <- Main variables used in other scripts such as paths and flood event dates
│
├── d04_scripts                                   <- Full code routines
│   ├── calibrate.py                              <- Semi-synthetic data experiments to test model calibration
│   ├── generate.py                               <- Generates data to test priors
│   ├── inference.py                              <- Empirical routines
│   └── read_inferences.py                        <- Read job outputs into posterior and summaries            
│
├── d05_joboutputs                                <- Outputs should be directed and read from this directory. Not included due to size limits
│
├── d06_processed-outputs                         <- Outputs from the read_inference.py script such as posterior summaries and inferred quantities
│
└── d07_plots                                     <- Plots used on paper


USAGE

You have two sets of options: 
1. Provide Your Own FSQ Data or Generate FSQ Data
2. Provide a Custom Blacklist for your data, or Generate a Blacklist within the pipeline

If choosing 1., 
Deduplicate on zip 10005: 
python run_pipeline.py 
--input data/fsq_data/fsq_pois_10005
--blacklist_path /share/… 
--name_threshold 90 
--max_distance 100 
--precision 1 
--labeled_df_file_path results/fsq_dedup_10005.parquet 
--metric_file_path results/fsq_10005_test


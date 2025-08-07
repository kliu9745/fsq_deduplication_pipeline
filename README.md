# A DEDUPLICATION PIPELINE TO CLEAN AND REMOVE DUPLICATES FROM FSQ PLACES DATA


### Pipeline Structure
```text
├── data
│   ├── blacklists                  <- Contain blacklists for specific FSQ datasets
│   ├── data_setup.ipynb            <- Notebook for FSQ data extraction
│   ├── data_setup.py               <- Script for data_setup.ipynb used in other files
│   ├── fsq_data                    <- Stores FSQ Datasets for specific cities and states
│   ├── generate_blacklist.ipynb    <- Generates a Custom Blacklist (common areas/words to filter out during name matching) for your FSQ Dataset
│   ├── generate_blacklist.py       <- Script for generate_blacklist.ipynb used in other files
├── deduplication
│   ├── geohash_utils.py           <- Contains geohash-specific helper functions
│   ├── helpers.ipynb              <- Stores the majority of helper/util functions used in deduplicated
│   ├── name_utils.py              <- Contains util functions for name matching similarity, choosing the representative poi of a duplicate group, and default NYC blacklist
│   ├── other_utils.py             <- Other helpers 
│   ├── processor.ipynb            <- Contains main functions for geohash partitioning and grouping POIs based on spatial proximity and name similarity
│   └── processor.py               <- Script for processor.ipynb used in main pipeline
├── logs                           <- Produces both a console log and error log file here for each job ran in the slurm script
├── results                        <- Stores the final deduplicated and labeled datasets produced by the pipeline
├── analysis                        
│   └── validation.ipynb           <- Contains Visualizations and Plots for FSQ Data, Validations, and Demographic Analysis
├── fsq_deduplication_script.ipynb <- Original file used to build pipeline -- use as a reference only
├── dedup_job.sbatch               <- Use to run pipeline in batches
└── run_pipeline.py                <- Main script to run pipeline

### SETUP and USAGE
First, cd to this folder called 'fsq_dedup_pipeline' in your local directory.
On G2, the path is: '/share/garg/accessgaps2024/fsq_dedup_pipeline'. 
Then, activate the environment: 'conda activate /share/garg/conda_virtualenvs/wildfires'.

### Running the Pipeline:

You have two sets of options: 
1. Provide Your Own FSQ Data or Generate FSQ Data
2. Provide You Own Blacklist for your data, or Generate a Custom Blacklist within the pipeline

1: Provide Own FSQ Data (CSV and Parquet only):
    add parameter --input <path_to_fsq_df_file>
   Generate FSQ Data
    add parameters:
        --fsq_release_date <str date in the form: 'YYYY-MM-DD'> (Default is '2025-06-10', must be a valid FSQ update release date) 
    + one of three parameters: 
        --regions <state/country abbrv str> <state/country abbrv str> ..., 
        --boundingbox <min_lon> <max_lon> <min_lat> <max_lat>, 
        or --county <geoid> <geoid> ...

2: Provide Your Own Blacklist
    add parameter --blacklist_path <path_to_blacklist>
   Generate a Custom Blacklist
    add parameters: --generate_blacklist and --generated_blacklist_path <path_to_save_generated_blacklist_to>

Other parameters:
 --max_distance <num greater than 0> (Default is 50)
 --name_threshold <between 0-100> (Default is 80)
 --precision <between 1-12> (Default is 7)
 --labeled_df_file_path <path_to_save_labeled_df_to> (REQUIRED)
 --metric_file_path <path_to_save_dedup_metrics_to>


### EXAMPLE: Deduplicate on zip 10005

1. Provide both data and blacklist yourself: 
python run_pipeline.py 
--input data/fsq_data/fsq_pois_10005
--blacklist_path /share/… 
--name_threshold 90 
--max_distance 100 
--precision 1 
--labeled_df_file_path results/fsq_lbled_1005.parquet 
--metric_file_path results/fsq_10005_metrics

2. Provide data but generate custom blacklist
python run_pipeline.py 
--input data/fsq_data/fsq_pois_10005 
--generate_blacklist 
--generated_blacklist_path data/blacklists/blacklist_10005
--max_distance 100 
--name_threshold 90 
--labeled_df_file_path results/fsq_lbled_1005.parquet 
--metric_file_path results/fsq_10005_metrics

3. Generate data (by bounding box) and custom blacklist
python run_pipeline.py 
---boundingbox -74 40 -73 41
----fsq_release_date '2025-06-10'
--generate_blacklist 
--generated_blacklist_path data/blacklists/blacklist_10005
--max_distance 100 
--name_threshold 90 
--labeled_df_file_path results/fsq_lbled_1005.parquet 
--metric_file_path results/fsq_10005_metrics

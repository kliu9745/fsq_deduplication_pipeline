# A DEDUPLICATION PIPELINE TO CLEAN AND REMOVE DUPLICATES IN FSQ PLACES DATA


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

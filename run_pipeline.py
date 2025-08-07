import argparse
import os
import pandas as pd
import geopandas as gpd
import asyncio
from pathlib import Path
import json

from data.data_setup import generate_df_from_fsq_by_region, generate_df_from_fsq_by_us_county
from deduplication.processor import deduplicate
from deduplication.name_utils import NYC_BLACKLIST
from data.generate_blacklist import generate_blacklist
# from data.generate_blacklist import

def parse_args():
    parser = argparse.ArgumentParser(description="Run FSQ deduplication pipeline.")

    # INPUT OPTIONS
    parser.add_argument("--input", type=str, help="Path to input FSQ POI GeoDataFrame (optional)", required=False)
    parser.add_argument("--regions", nargs='+', help="Regions list to generate FSQ Dataset if no input GeoDataFrame is given", required = False)
    parser.add_argument("--boundingbox", nargs='+', help="List [min_lon, min_lat, max_lon, max_lat] to generate FSQ Dataset if no input GeoDataFrame is given", required = False)
    parser.add_argument("--county", nargs='+', help="County list to generate FSQ Dataset if no input GeoDataFrame is given", required = False)
    parser.add_argument("--fsq_release_date", type=str, default="2025-06-10", help="FSQ release date for data pull", required=False)

    # BLACKLIST OPTIONS
    parser.add_argument("--blacklist_path", type=str, help="Optional path to precomputed blacklist file")
    parser.add_argument('--generate_blacklist', action='store_true', help="Generate blacklist from FSQ data")
    parser.add_argument('--generated_blacklist_save_path', type =str, help="Store the custom blacklist made for FSQ data in a txt file")

    # DEDUP ARGUMENTS
    parser.add_argument("--max_distance", type=int, default=100, help="Max distance (meters) for spatial grouping")
    parser.add_argument("--name_threshold", type=int, default=90, help="Fuzzy name similarity threshold")
    parser.add_argument("--precision", type=int, default=7, help="Geohash precision for input gdf")
    parser.add_argument("--labeled_df_file_path", type=str, help="Output name for labeled df file", required=True)
    parser.add_argument("--metric_file_path", type=str, help="Output name for metrics file", required=False)
    parser.add_argument("--generated_fsq_data_path", type=str, help="Output name for generated FSQ Data file", required=False)
    
    return parser.parse_args()

def load_blacklist(path):
    if path:
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return NYC_BLACKLIST

def main():
    args = parse_args()
    blacklist = load_blacklist(args.blacklist_path)

    if args.input:
        print(f"Loading FSQ data from {args.input}")
        try:
            gdf = gpd.read_parquet(args.input)
        except:
            gdf = pd.read_csv(args.input)
    else:
        # If requested, generate input GeoDataFrame
        print(f"Generating FSQ data for regions: {args.regions}")
        gdf = generate_df_from_fsq_by_region(args.regions, args.fsq_release_date, args.generated_fsq_data_path)

    if args.blacklist_path:
        print(f"Loading blacklist file from {args.blacklist_path}")
        with open(args.blacklist_path, 'r') as f:
            blacklist = json.load(f)
    elif args.generate_blacklist:
        blacklist = generate_blacklist(gdf, geohash_precision = args.precision, file_path = args.generated_blacklist_save_path)
    else:
        blacklist = NYC_BLACKLIST

    print("Running Deduplication")
    _, orig_labeled, metrics_df = asyncio.run(
        deduplicate(
            gdf,
            max_distance=args.max_distance,
            name_similarity_threshold=args.name_threshold,
            precision = args.precision,
            blacklist=blacklist,
            file_name = args.metric_file_path
        )
    )
    print(f"Saving labeled dataset to {args.labeled_df_file_path}")
    Path(args.labeled_df_file_path).parent.mkdir(parents=True, exist_ok=True)
    orig_labeled.to_parquet(args.labeled_df_file_path)
    metrics_df.to_csv(args.metric_file_path, index=False)
    print("The pipeline has ran successfully!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()





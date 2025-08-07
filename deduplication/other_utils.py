# %%
# from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import geopandas as gpd
import re
import unicodedata
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
# !pip install python-geohash
import geohash

from geopy.distance import geodesic

# !pip install rapidfuzz
from rapidfuzz import process,fuzz
from rapidfuzz.fuzz import partial_ratio
from shapely.geometry import Point
import duckdb
import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
import re
from shapely import wkt

def convert_fsq_csv_to_gdf(fsq_file, geometry_col ='geometry'):
    """ 
    Loads a FSQ Places dataset (CSV or Parquet) and converts it into a GeoDataFrame.

    Parameters:
        fsq_file (str): path to FSQ Dataset file (CSV or Parquet)
        geometry_col (str): column name used to store POI geometries
    """
    try:
        df = pd.read_csv(fsq_file)
    except:
        df = pd.read_parquet(fsq_file)
    df[geometry_col] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry=geometry_col, crs="EPSG:4326")
    gdf['date_created'] = pd.to_datetime(gdf['date_created'], errors='coerce')
    gdf['date_closed'] = pd.to_datetime(gdf['date_closed'], errors='coerce')
    return gdf

def extract_top_category(cat_array):
    """
    Returns the top-level category of a single 'fsq_category_label' entry within a POI row.
    If cat_array is not an np.ndarray or is an empty np.ndarray, then returns ''

    Parameters:
        cat_array (np.ndarray): array of string category labels, where each label is broken into hierarchical subcategories using '<'/
    Returns: a string representing the top-level category of a label
    """
    if isinstance(cat_array, np.ndarray) and len(cat_array) > 0:
        return cat_array[0].split(' > ')[0].strip()
    else:
        return ''
    
def select_most_recent_row(df):
    """
    Take the most recent row from a group of sPOIs based on 'date_refreshed' or 'date_created'.
    If 'date_refreshed' is NaN for all rows, use 'date_created' instead.
    Parameters:
        df (pd.DataFrame): DataFrame containing a group of POIs with 'date_refreshed' and 'date_created' columns.
    """
    if df['date_refreshed'].notna().any():
        return df.sort_values(by = 'date_refreshed', ascending=False).iloc[0]
    else:
        return df.sort_values(by = 'date_created', ascending=False).iloc[0]
    
def normalize_group(df, cols):
    for col in cols:
        scaler = MinMaxScaler()
        df[col + "_norm"] = scaler.fit_transform(df[col])
        # transform(lambda x: MinMaxScaler().fit_transform(x.values.reshape(-1, 1)))
    return df

def save_results_to_gdf(df, resolved_map):
    """
    Adds two new columns to [df]: 'isdup' and 'resolved_fsq_id'.
    For each row in [df], 'isdup' = True when this row is a duplicate and isdup = 0 otherwise
    For each duplicate row r in [df], 'resolved_fsq_id' maps r to its corr. row in the filtered df, that it was merged into.
    Parameters:
    df:
    dup_lst:
    """
    df['isdup'] = df['fsq_place_id'].isin(resolved_map.keys()).astype(bool)
    df['resolved_fsq_id'] = df['fsq_place_id'].map(resolved_map)
    # update for kept rows in filtered df
    kept_ids = set(resolved_map.values())
    kept = df['fsq_place_id'].isin(kept_ids)
    df.loc[kept, 'isdup'] = True
    df.loc[kept, 'resolved_fsq_id'] = df.loc[kept, 'fsq_place_id']
    return df

def filter_by_zcta(gdf, zcta, zcta_codes):
    """
    Filters the GeoDataFrame [gdf] to include only POIs within the specified ZCTA code.
    
    Parameters:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing POIs with 'geometry' column.
        zcta (gpd.GeoDataFrame): GeoDataFrame containing ZCTA polygons.
        zcta_code (str): The ZCTA code to filter by.
    
    Returns:
        gpd.GeoDataFrame: Filtered GeoDataFrame containing only POIs within the specified ZCTA.
    """

    # Make sure both CRS's are EPSG:4326 (meters)
    gdf = gdf.to_crs(epsg=4326)
    zcta = zcta.to_crs(epsg=4326)
    zcta_filtered = zcta[zcta['ZCTA5CE20'].isin(zcta_codes)]
    return gpd.sjoin(gdf, zcta_filtered, predicate = 'within')


def calculate_metrics(rows_in_group, group, category, centroid):
    # metrics
    metrics = {}

    # Number of distinct names and addresses
    metrics['n_names'] = rows_in_group['name'].nunique()
    metrics['n_addresses'] = rows_in_group['address'].nunique()


    # Dominant name and address ratio (safe version)
    name_counts = rows_in_group['name'].value_counts(normalize=True)
    metrics['dominant_name_ratio'] = name_counts.iloc[0] if not name_counts.empty else None

    addr_counts = rows_in_group['address'].value_counts(normalize=True)
    metrics['dominant_address_ratio'] = addr_counts.iloc[0] if not addr_counts.empty else None

    # Mean name similarity (pairwise)
    name_list = rows_in_group['name'].tolist()
    if len(name_list) > 1:
        name_similarities = [
            partial_ratio(a, b) for i, a in enumerate(name_list) for b in name_list[i+1:]
        ]
        metrics['mean_name_similarity'] = float(np.mean(name_similarities))
    else:
        metrics['mean_name_similarity'] = 100

    # Spatial spread
    coords = rows_in_group[['latitude', 'longitude']].values
    centroid_coords = (centroid.y, centroid.x)
    if len(coords) > 1:
        dists = [
            geodesic(coords[i], coords[j]).meters
            for i in range(len(coords)) for j in range(i+1, len(coords))
        ]
        cent_dist = [ geodesic(coords[i], centroid_coords).meters for i in range(len(coords))]
        metrics['mean_distance_m'] = float(np.mean(dists))
        metrics['max_distance_m'] = float(np.max(dists))
        metrics['mean_cent_dist'] = float(np.mean(cent_dist))
        metrics['max_cent_distance'] = float(np.max(cent_dist))
    
    else:
        metrics['mean_distance_m'] = 0
        metrics['max_distance_m'] = 0
        metrics['mean_cent_dist'] = 0
        metrics['max_cent_distance'] = 0

    # Date variance
    for date_col in ['date_opened', 'date_refreshed', 'date_closed']:
        if date_col in rows_in_group.columns:
            try:
                dates = pd.to_datetime(rows_in_group[date_col].dropna())
                if len(dates) > 1:
                    metrics[f'{date_col}_std_days'] = float((dates - dates.min()).dt.days.std())
                    metrics[f'{date_col}_range_days'] = int((dates.max() - dates.min()).days)
                    # if metrics[f'{date_col}_std_days'] > 100:
                        # print(f"{date_col}std_days: " + str(metrics[f'{date_col}_std_days']))
                    # if metrics[f'{date_col}_range_days'] > 100:
                        # print(f"{date_col}range_days: " + str(metrics[f'{date_col}_range_days']))

                else:
                    metrics[f'{date_col}_std_days'] = 0
                    metrics[f'{date_col}_range_days'] = 0
            except Exception:
                metrics[f'{date_col}_std_days'] = None
                metrics[f'{date_col}_range_days'] = None

    group_id = "_".join(sorted(group))
    metrics['group_id'] = group_id
    metrics['category'] = category
    metrics['group_size'] = len(group)
    print(f"Group metrics: {metrics}")
    return metrics

def visualize_metrics(file_path):
    metrics = pd.read_csv(file_path)
    # print("Metrics: ")
    # print(metrics.head())
    summary = metrics.describe(include='all')
    print("Summary: ")
    print(summary[['n_names', 'n_addresses', 'dominant_name_ratio', 'mean_name_similarity', 'mean_distance_m', 'max_distance_m', 'mean_cent_dist', 'max_cent_distance']])
    
    # Plot distribution of name similarity
    plt.figure(figsize=(8, 5))
    sns.histplot(metrics['mean_name_similarity'], bins=30, kde=True)
    plt.title("Distribution of Mean Name Similarity")
    plt.xlabel("Mean Name Similarity")
    plt.ylabel("Group Count")
    plt.show()

    # Dominant name ratio
    sns.histplot(metrics['dominant_name_ratio'], bins=30, kde=True)
    plt.title("Distribution of Dominant Name Ratio")
    plt.xlabel("Dominant Name Ratio")
    plt.show()

    # Mean pairwise distance (m) within each group
    sns.histplot(metrics['mean_distance_m'], bins=30, kde=True)
    plt.title("Distribution of Mean Distances Between POIs in a Group")
    plt.xlabel("Mean Distance (m)")
    plt.show()

    # Max pairwise distance (m) within each group
    sns.histplot(metrics['max_distance_m'], bins=30, kde=True)
    plt.title("Distribution of Max Distances Between POIs in a Group")
    plt.xlabel("Max Distance (m)")
    plt.show()

    # Group Size
    sns.histplot(metrics['group_size'], bins=30, kde=True)
    plt.title("Distribution of Group Size")
    plt.xlabel("Group Size")
    plt.show()

    # STD of date_closed
    sns.histplot(metrics['date_closed_std_days'], bins=30, kde=True)
    plt.title("Distribution of STD of Dates Closed within a Group")
    plt.xlabel("STD of Dates Closed within a Group")
    plt.show()


    # Range of date_closed
    sns.histplot(metrics['date_closed_range_days'], bins=30, kde=True)
    plt.title("Distribution of Range of Dates Closed within a Group")
    plt.xlabel("Range of Dates Closed within a Group")
    plt.show()

    # Dominant Address Ratio
    sns.histplot(metrics['dominant_address_ratio'], bins=30, kde=True)
    plt.title("Distribution of Dominant Address Ratio")
    plt.xlabel("Dominant Address Ratio")
    plt.show()

    # Num of Names
    sns.histplot(metrics['n_names'], bins=30, kde=True)
    plt.title("Distribution of Number of Distinct Names In a Group")
    plt.xlabel("Number of Distinct Names In a Group")
    plt.show()

    # Dominant Address Ratio
    sns.histplot(metrics['n_addresses'], bins=30, kde=True)
    plt.title("Distribution of Number of Unique Addresses in a Group")
    plt.xlabel("Number of Unique Addresses in a Group")
    plt.show()




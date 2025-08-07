# %%
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import geopandas as gpd
import re
import unicodedata
from collections import Counter

# import seaborn as sns
# import matplotlib.pyplot as plt
# !pip install networkx
import networkx as nx
# !pip install python-geohash
# import geohash

from geopy.distance import geodesic

# !pip install rapidfuzz
from rapidfuzz import process,fuzz
from rapidfuzz.fuzz import partial_ratio
from shapely.geometry import Point
# import duckdb
# import aiohttp
# import asyncio
# import nest_asyncio
# nest_asyncio.apply()
# import re
# from shapely import wkt

import sys
import os
sys.path.append(os.path.abspath('./deduplication'))
from geohash_utils import assign_geohashes, get_neighboring_geohashes
from name_utils import NYC_BLACKLIST, remove_common_words, clean_name, choose_common_name_from_group
from other_utils import select_most_recent_row, extract_top_category, save_results_to_gdf, calculate_metrics


### Step 2: Filtering based on Fuzzy Matching + Spatial Promixity
def group_similar_names_spatial_graph(df, tree, coords, earth_radius, max_distance, similarity_threshold=85, blacklist = NYC_BLACKLIST):
    """ 
    Parameters:
        df (DataFrame): FSQ DataFrame filtered for a specific category of POIs in a single geohash grid
    
    """
    G = nx.Graph()

    # adding all POIs as nodes
    for _, row in df.iterrows():
        G.add_node(row['fsq_place_id'], name = row['name'], lon = row['longitude'], lat = row['latitude'])

    # build edges based on spatial proximity and name similarity
    for i in range(len(df)):
        # if i not in visited:
        fsq_id_i = df.iloc[i]['fsq_place_id']
        name_i = df.iloc[i]['name']
        indices = tree.query_radius([coords[i]], r=max_distance / earth_radius)[0]

        for j in indices:
            if j != i:
                fsq_id_j = df.iloc[j]['fsq_place_id']
                name_j = df.iloc[j]['name']
            # Calculate similarity using token sort ratio
                sim_i_j = fuzz.token_set_ratio(remove_common_words(clean_name(name_i), blacklist, False), remove_common_words(clean_name(name_j), blacklist, False))
                if sim_i_j >= similarity_threshold:
                    G.add_edge(fsq_id_i, fsq_id_j)

    # Find connected components
    groups = [g for g in list(nx.connected_components(G)) if len(g) > 1]
    return groups     

### Step 3: Process a Geohash Group
# process geohash groups
async def process_groups(gdf, hash, category_list, name_similarity_threshold, max_distance, precision=7, blacklist = NYC_BLACKLIST, resolved_map = {}, file_path = None):
    # Ensure gdf is in WGS84 for geohashing
    gdf = gdf.to_crs(epsg=4326)  
    # collect pois in the geohash and its neighbors
    neighboring_geohashes = get_neighboring_geohashes(hash)
    local_gdf = gdf[gdf['geohash'].isin(neighboring_geohashes)].copy()
    earth_radius = 6371000  # Earth's radius (m)

    all_group_metrics = []
    # iterate over each category and update df to remove duplicates
    for category in category_list:
        # print("category: "+category)
        category_gdf = local_gdf[local_gdf['top_category'] == category].copy()
        if category_gdf.empty:
            continue

        coords = np.radians(np.array(list(zip(category_gdf.geometry.y, category_gdf.geometry.x))))
        tree = BallTree(coords, metric='haversine')
  
        # create groups of indices of POIs with similar names within a certain distance in the geohash
        sim_name_close_groups = group_similar_names_spatial_graph(category_gdf, tree, coords, earth_radius, max_distance, name_similarity_threshold, blacklist)
        # print("sim_name_close_groups: " + str(sim_name_close_groups)) if sim_name_close_groups else print("No groups")
        # input parent ids for each group in sim_name_close_groups
        # parent_ids, group_parent_dict, df_w_parent = await assign_parent_ids(category_gdf, sim_name_close_groups)

        if sim_name_close_groups:        
            # for each group of POIs with similar names,
            # if the group contains 2+ POIs with the same parent id, combine them by centroid and keep the most frequent name
            # if the group does not have a parent id, keep the most recent POI based on 'date_refreshed' or 'date_created'
            for group in sim_name_close_groups:
                if len(group) > 1:
                    print("\nGroup:")
                    print(category_gdf[category_gdf['fsq_place_id'].isin(group)][['name']])
                    # print("addresses: " + category_gdf[category_gdf['fsq_place_id'].isin(group)][['address']])
                    
                    rows_in_group = category_gdf[category_gdf['fsq_place_id'].isin(group)].copy()
                    latitude = rows_in_group['latitude'].mean()
                    longitude = rows_in_group['longitude'].mean()
                    centroid = Point(longitude, latitude)

                    # metrics
                    all_group_metrics.append(calculate_metrics(rows_in_group, group, category, centroid))
                    # for i, row in rows_in_group.iterrows():
                        # print("name: " + str(row['name']) + ", website: " + str(row['website']) + ", email: " + str(row['email']) + ", date_closed: " + str(row['date_closed']))
                    names = [n for n in rows_in_group['name'].tolist()]
                    
                    # common_name = longest_common_substring(names) 
                    common_name = choose_common_name_from_group(names, blacklist)
                    if common_name is None:
                        print("No common name found for group, skipping...")
                        continue
                        
                    most_recent_row = select_most_recent_row(rows_in_group)
                    # print("most recent name: "+ most_recent_row['name'])

                    # id to keep
                    most_recent_row_id = most_recent_row['fsq_place_id'] if isinstance(most_recent_row, pd.Series) else most_recent_row.iloc[0]['fsq_place_id']

                    # update resolved_map ({discarded_id: kept_id}) for this group  
                    for row in group:
                        resolved_map[row] = most_recent_row_id

                    # index to keep in original df
                    # gdf.index.get_loc(gdf[gdf['fsq_place_id'] == most_recent_row_id].index[0])
                    most_recent_label_index = gdf[gdf['fsq_place_id'] == most_recent_row_id].index[0]
                    # print("most_recent_label_index: " + str(most_recent_label_index))
                    # print("row name at that index: " + gdf.at[most_recent_label_index, 'name'])

                    # update row to keep in original gdf
                    gdf.at[most_recent_label_index, 'name'] = common_name
                    print("common_name: "+ common_name)
                    if not all(not d for d in rows_in_group['date_closed']):
                        gdf.at[most_recent_label_index, 'date_closed'] = rows_in_group['date_closed'].sort_values(ascending = False).iloc[0]

                    # update gdf and df with the common name and centroid
                    gdf.at[most_recent_label_index, 'latitude'] = latitude
                    gdf.at[most_recent_label_index, 'longitude'] = longitude
                    gdf.at[most_recent_label_index, 'geometry'] = centroid
                    # print("crs: " + str(gdf.crs))
                    # print("most_recent_row_id: " + most_recent_row_id)
                    drop_ids = set(group) - {most_recent_row_id}
                    gdf = gdf[~gdf['fsq_place_id'].isin(drop_ids)]
                    # print("df size: " + str(len(gdf)))
        # print("Processed a category")
    if file_path:
        metrics_df = pd.DataFrame(all_group_metrics)
    else:
        metrics_df = None
    return gdf, resolved_map, metrics_df

async def deduplicate(gdf, max_distance = 50, name_similarity_threshold = 80, precision = 7, blacklist = NYC_BLACKLIST, file_name = None): 
    """
    Deduplicates POIs using a grid-search technique, filtering out duplicates based on spatial proximity and fuzzy name similarity.
    
    Parameters:
        gdf (gpd.GeoDataFrame): FSQ GeoDataFrame with 'fsq_place_id', 'name', 'fsq_category_labels', 'latitude', 'longitude', date and 'geometry' columns, and more.
        max_distance (float): Maximum distance in meters for considering POIs as duplicates.
        name_similarity_threshold (int): Minimum fuzzy match ratio to consider names as duplicates.
    Returns:
        gpd.GeoDataFrame: Deduplicated POIs.
        original_saves: a copy of gdf with two additional columns: 'isdup' and 'resolved_fsq_id'. For each POI p, isdup is True if it is a duplicate 
        and resolved_fsq_id is the fsq_place_id of the kept POI in p's duplicate group. If p is not a duplicate, isdup is False and reoslved_fsq_id is None.
    """
    if gdf.empty:
        print("At least one input DataFrame is empty. Returning original df and gdf DataFrames.")
        return gdf

    gdf = gdf.to_crs("EPSG:3857")

    gdf.loc[:, 'date_created'] = pd.to_datetime(gdf['date_created'], errors='coerce')
    gdf.loc[:, 'date_closed'] = pd.to_datetime(gdf['date_closed'], errors='coerce')
    gdf.loc[:, 'date_refreshed'] = pd.to_datetime(gdf['date_refreshed'], errors='coerce')

    gdf = assign_geohashes(gdf, precision)
    gdf['parent_id'] = ''
    original = gdf.copy()
    
    gdf_list = []
    resolved_map = {}
    all_metrics_list = []
    print("num hashes: " + str(len(gdf['geohash'].unique())))
    for hash in gdf['geohash'].unique():
        # Filter the DataFrame for the current geohash
        gdf_geohash = gdf[gdf['geohash'] == hash]
        if gdf_geohash.empty:
            continue
        
        # Get unique categories in the current geohash
        gdf_geohash['top_category'] = gdf_geohash['fsq_category_labels'].apply(extract_top_category)

        unique_top_categories = gdf_geohash['top_category'].unique().tolist()
        # Process groups for the current geohash and its categories
        processed_gdf, resolved_map, metrics_df = await process_groups(gdf_geohash, hash, unique_top_categories, name_similarity_threshold, max_distance, precision, blacklist, resolved_map, file_name)
        gdf_list.append(processed_gdf)
        if file_name:
            all_metrics_list.append(metrics_df)
            
    gdf_concat = pd.concat(gdf_list, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf_concat, geometry='geometry', crs="EPSG:3857")
    original_saved = save_results_to_gdf(original, resolved_map)
    if file_name:
        metrics_concat = pd.concat(all_metrics_list, ignore_index = True)
    else:
        metrics_concat = None
    return gdf, original_saved, metrics_concat


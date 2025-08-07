from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import re
import unicodedata
from collections import defaultdict, Counter
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union
import pandas as pd
import geohash
import nltk
from nltk.corpus import stopwords
# Download stopwords once
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize(text):
    """Lowercase, remove punctuation, and split into tokens."""
    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    return words

def token_distribution_by_geohash(df, geohash_precision=5):
    """ 
    Helper function for [generate_blacklist].
    Produces a DataFrame storing the tokens of all names in [df], 
    along with the global count for each token and the number of regions it appears in.

    Returns the total number of regions the df is partitioned into and the token dataframe
    as a tuple, n_regions, token_df

    Parameters:
        df (FSQ POI DataFrame/GeoDataFrame): must have 'name', 'latitude' and 'longitude' columns.
        geohash_precision (int, optional): must be between 1 (inclusive) and 12 (inclusive). 
        Larger precisions mean smaller partitioning of grids in [df]. The default param is 5.
    """
    df = df.copy()
    
    # Vectorized geohash encoding using zip and list comprehension
    df['geohash'] = [geohash.encode(lat, lon, precision=geohash_precision) 
                     for lat, lon in zip(df['latitude'], df['longitude'])]
    
    # Tokenize names
    df['tokens'] = df['name'].apply(tokenize)  # assumes list of strings

    print(f"Total POIs: {len(df)}")
    
    # Flatten all tokens for global count
    global_token_counts = Counter(token for tokens in df['tokens'] for token in tokens)
    
    # Group by geohash
    geo_token_counts = defaultdict(Counter)
    geo_token_sets = defaultdict(set)

    for geo, tokens in zip(df['geohash'], df['tokens']):
        geo_token_counts[geo].update(tokens)
        geo_token_sets[geo].update(tokens)

    # Now compile final stats
    rows = []
    for token, global_count in global_token_counts.items():
        n_regions = sum(1 for geo in geo_token_counts if token in geo_token_counts[geo])
        rows.append({
            'token': token,
            'global_count': global_count,
            'n_regions': n_regions
        })

    token_df = pd.DataFrame(rows)
    n_unique_regions = df['geohash'].nunique()
    
    return n_unique_regions, token_df.sort_values(by='global_count', ascending=False)

def generate_blacklist(gdf, geohash_precision=5, file_path = None):
    """ 
    Creates a list of globally common and widespread words found from names in [gdf].

    Parameters:
        gdf (GeoDataFrame): contains tokens and their global counts and number of regions appeared in 
        (look at token_distribution_by_geohash). Must have 'token', 'global_count', and 'region_coverage' columns.
        geohas_precision (int, optional): must be between 1 (inclusive) and 12 (inclusive). 
        Larger precisions mean smaller partitioning of grids in [df]. The default param is 5.
        file_path (string, optional): path to a txt file storing the blacklist, where each line is a single token.
    """
    n_regions, token_df = token_distribution_by_geohash(gdf, geohash_precision)
    print(f"Total unique geohashes: {n_regions}")
    token_df['global_pct'] = token_df['global_count'] / token_df['global_count'].max()
    token_df['region_coverage'] = token_df['n_regions'] / n_regions
    blacklist = token_df[(token_df['global_pct'] > 0.01) & (token_df['region_coverage'] > 0.005)]['token'].to_list()
    if file_path:
        # Open the file in write mode ('w')
        with open(file_path, 'w') as f:  
            for item in blacklist:
                # Write each item followed by a newline
                f.write(f"{item}\n") 
    return blacklist 




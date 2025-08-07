# %% [markdown]
# # Data Setup

# %%
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import geopandas as gpd
import re
import unicodedata
from collections import Counter

import duckdb
import aiohttp
import asyncio
import nest_asyncio
nest_asyncio.apply()
import re
from shapely import wkt
from shapely.geometry import Polygon
from shapely.ops import unary_union


# %% [markdown]
# ### Generating FSQ Places Dataset
def generate_df_from_bb(min_lon, max_lon, min_lat, max_lat, fsq_release_date = "2025-06-10", file_name = 'gdf_fsq_pois'):
    """
    Produces a parquet file of the FSQ POI dataset within a given bounding box using the version published on [fsq_release_date].
    The file is saved at '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_bb.parquet'.
    Parameters:
        min_lon (float): southmost longitude
        max_lon (float): northmost longitude
        min_lat (float): westmost latitude
        max_lat (float): eastmost latitude
        fsq_release_date (string): FSQ release date, in the form "YYYY-MM-DD".
        file_name (string): name of file that the resulting fsq dataset will be saved as.
    ______________________________
    Ex Usage: 
    To generate (one time only):
        US_MIN_LAT = 24.396308   # Southern tip of Florida
        US_MAX_LAT = 49.384358   # Northern border (Minnesota)
        US_MIN_LON = -124.848974 # Western edge (California coast)
        US_MAX_LON = -66.93457   # Eastern edge (Maine)
        us_df = generate_df_from_bb(US_MIN_LON, US_MAX_LON, US_MIN_LAT, US_MAX_LAT, fsq_release_date = "2025-06-10", file_name = 'us_by_bb')
    To retrieve parquet on future runs:
        us_df = pd.read_parquet('/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_bb.parquet')
    """
    s3_path = f"s3://fsq-os-places-us-east-1/release/dt={fsq_release_date}/places/parquet/*.parquet"
    output_path = f"/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_bb.parquet"

    # Connect to DuckDB
    conn = duckdb.connect()

    # Read from S3 using DuckDB and filter with bounding box
    query = f"""
        SELECT *
        FROM read_parquet('{s3_path}')
        WHERE 
            longitude BETWEEN {min_lon} AND {max_lon}
            AND latitude BETWEEN {min_lat} AND {max_lat}
    """
    print("Running query...")
    df = conn.execute(query).fetch_df()
    conn.close()


    if df.empty:
        print("Warning: No POIs found in the given bounding box.")
    else:
        # Save to Parquet
        print(f"Saving {len(df)} rows to {output_path}")
        df.to_parquet(output_path)

    return df

def generate_df_from_fsq_by_region(selected_regions_lst, fsq_release_date = "2025-06-10", file_name = 'gdf_fsq_pois'):
    """Produces a parquet file of the FSQ POI dataset within the regions of [selected_regions_lst], 
    using the version published on [fsq_release_date].
    The file is saved at '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_region.parquet'.
    Parameters:
        selected_regions_lst (list): list of states, provinces, territories, etc. 
                                    Abbreviations are used in the countries: US, CA, AU, BR.
                                    Please check the FourSquare Places Dataset documentation for more specific details.
        fsq_release_date (string): FSQ release date, in the form "YYYY-MM-DD".
        file_name (string): name of file that the resulting fsq dataset will be saved as.

    Ex Usage:
    To generate (one time only):
    ny_df = generate_df_from_fsq_by_region(['NY'], fsq_release_date = "2025-06-10", file_name = 'ny_fsq_pois')
    To retrieve parquet on future runs:
    ny_df = pd.read_parquet('/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_region.parquet')
    """
    assert isinstance(selected_regions_lst, list) and selected_regions_lst != []

    s3_path = f"s3://fsq-os-places-us-east-1/release/dt={fsq_release_date}/places/parquet/*.parquet"

    # Connect to DuckDB
    conn = duckdb.connect()

    # Query POIs within Manhattan bounds
    copy_query = f"""
        COPY (
            SELECT *
            FROM read_parquet('{s3_path}')
            WHERE 
                region in {selected_regions_lst}
        ) TO '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_region.parquet' (FORMAT PARQUET)
    """

    conn.execute(copy_query)

    # Load the result into a dataframe
    q = duckdb.connect()
    q.execute(f"SELECT * FROM '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_region.parquet'")
    res = q.fetch_df()
    conn.close()
    q.close()
    return res

#### Per County
county = gpd.read_file('/share/garg/accessgaps2024/fsq_dedup_pipeline/data/tl_2024_us_county/tl_2024_us_county.shp')

def generate_df_from_fsq_by_us_county(selected_county_geoids, fsq_release_date = "2025-06-10", file_name = 'gdf_fsq_pois'):
    """Produces a parquet file of the FSQ POI dataset within the US counties of [selected_county_geoids], 
    using the version published on [fsq_release_date].
    The file is saved at '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_county.parquet'.
    Parameters:
        selected_us_county_geoids (list): list of US county geoids (5-digit strings). Please check Census.Gov documentation for more specific details.
        fsq_release_date (string): FSQ release date, in the form "YYYY-MM-DD".
        file_name (string): name of file that the resulting fsq dataset will be saved as.

    Ex Usage:
    To generate (one time only):
    nyc_df = generate_df_from_fsq_by_us_county(["36005", "36047", "36061", "36081", "36085"], fsq_release_date = "2025-06-10", file_name = 'nyc_fsq_pois')
    To retrieve parquet on future runs:
    nyc_df = pd.read_parquet('/share/garg/accessgaps2024/fsq_dedup_pipeline/data/fsq_data/{file_name}_by_county.parquet')
    """
    assert isinstance(selected_county_geoids, list) and selected_county_geoids != []
    # Filter to the specified counties by GEOID
    gdf = county[county["GEOID"].isin(selected_county_geoids)].to_crs("EPSG:4326")
    gdf_geom = unary_union(gdf['geometry'])

    # Get WKT (Well-Known Text) for DuckDB
    gdf_wkt = gdf_geom.wkt

    s3_path = f"s3://fsq-os-places-us-east-1/release/dt={fsq_release_date}/places/parquet/*.parquet"

    # Connect to DuckDB
    conn = duckdb.connect()
    conn.execute("INSTALL spatial;")
    conn.execute("LOAD spatial;")

    # Register the WKT as a geometry in DuckDB
    conn.execute("CREATE TEMP TABLE gdf_bounds AS SELECT ST_GeomFromText(?) AS geom", [gdf_wkt])

    # Now do the spatial filter using the geometry column
    query = f"""
    COPY (
        SELECT *
        FROM read_parquet('{s3_path}'),
            gdf_bounds
        WHERE ST_Contains(gdf_bounds.geom, ST_Point(longitude, latitude))
    ) TO '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/{file_name}_by_county' (FORMAT PARQUET)
    """
    conn.execute(query)

    # Load the result into a dataframe
    q = duckdb.connect()
    q.execute(f"SELECT * FROM '/share/garg/accessgaps2024/fsq_dedup_pipeline/data/{file_name}_by_county.parquet'")
    res =  q.fetch_df()
    conn.close()
    q.close()
    return res




# !pip install python-geohash
import geohash

def assign_geohashes(df, precision=7):
    """
    Creates a new column, 'geohash', in [df] with the geohash code for each POI row.
    
    The length of the geohash code is [precision]. Longer geohashes are more precise and thus are smaller regions.
    Precision 1 is a ~(5000km x 5000km) area (ex: large country) while Precision 7 is a ~(153m x 153m) area (ex: size of a Manhattan zip).

    df (FSQ DataFrame/GeoDataFrame): must have 'latitude' and 'longitude' columns
    precision: length of a geohash -> the size of each grid in the partition. 1 <= precision <= 12.
    """
    assert 'latitude' in df.columns and 'longitude' in df.columns
    assert precision >= 1 and precision <= 12

    df['geohash'] = df.apply(lambda row: geohash.encode(row['latitude'], row['longitude'], precision=precision), axis=1)
    return df

def get_neighboring_geohashes(hash):
    """ 
    Returns a list of [hash] and the geohash codes of [hash]'s eight immediate neighbors.
    Can be used to query surrounding areas for a POI or nearby POIs.

    Parameters:
    hash (string): geohash between length 1-12
    """
    assert len(hash) >= 1 and len(hash) <= 12
    neighbors = geohash.neighbors(hash)
    # include the original geohash
    return [hash] + neighbors




from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import geopandas as gpd
import re
import unicodedata
from collections import Counter
import geohash


### NYC BLACKLIST
GLOBAL_FOOD = {"pizza", "pizzeria", "cafe", "caffe", "coffee", "grill", "restaurant", "bar", "deli", "bakery", "express",
                  "market", "shop", "mart", "store", "grocery", "supermarket", "food", "gourmet", "cart", "fresh", "kitchen", 
                  "diner", "pub", "bistro", "tavern", "farm", "chicken", "burger", "sandwich", "taco", "tacos", "sushi", "noodle", "noodles"
                  "salad", "sub", "ice cream", "dessert", "breakfast", "brunch", "lunch", "dinner", "takeout", "delivery", 
                  "snack", "chocolate", "tea", "juice", "smoothie", "wine", "beer", "cocktail", "brewery", "distillery", 
                  "winery", "patisserie", "pastry", "bagel", "donut", "pancake", "waffle", "crepe", "treat", "cuisine", "truck",
                  "italian", "mexican", "chinese", "japanese", "korean", "indian", "thai", "vietnamese", "greek", "spanish",
                  "french", "american", "cuban", "cajun", "creole", "soul food", "bbq", "steakhouse", "seafood",
                  "vegetarian", "vegan", "gluten-free", "organic", "local", "artisan", "handmade", "craft", "homemade",
                  "family-owned", "authentic", "traditional", "fusion", "gastro", "gastropub", "halal", "kosher"}
GLOBAL_TRANS_ADDR = {"east", "west", "north", "south", "st", "street", "ave", "avenue", 
                 "blvd", "road", "rd", "rd.", "drive", "cab", "car", "truck", "van", "taxi", "metro", "sub", "subway", "mta", "station", "apt", "apartment", "station", "corner", "bus", "express", "line", "line", "plaza", "plz", "square", "sq", "lane", "ln", "way", "wy", "court", "ct",
                 "park", "pl", "pkwy", "parkway", "circle", "cir", "highway", "hwy", "route", "rte", "exit", "exit", "bridge", "bridges", "crossing", "crossings",
                 "crossings", "intersection", "intersections", "boulevard", "boulevards", "roadway", "roadways", "driveway", "driveways","avenue", "avenues", "streetway"}
GLOBAL_LOC = {"city", "village", "town", "museum", "group", "house", "center", "ctr", "art", "shop", "show", "theatre", "theater", "office", "service", "services", "bank", "jewelry", "club", 
              "community", "garden", "park", "field", "beach", "ocean", "river", "playground", "school", "college", "university", "library", "gallery", "studio", "hall", "auditorium", "venue", "church"
              "cleaner", "cleaners", "laundry", "laundromt", "pharmacy", "church", "clinic", "gym", "hospital", "fitness", "nail", "nails", "salon", "spa", "barber", "project", "projects"}
GLOBAL_WORDS = {"the", "a", "an", "and", "&", "of", "in", "for", "to", "at", "@", "on", "out", "with", "by", "from", "as", "that", "this", "it", "is", "was", "be", "are", "day", "care", "co."}
GLOBAL_SCHOOL_WRDS = {"high", "middle", "elementary", "school", "academy", "charter", "magnet", "daycare", "day", "care"}
GLOBAL_FIRE_STATION = {"fdny", "engine", "ems", "rescue", "group"}
GLOBAL_POLICE_STATION = {"nypd"}
LOCAL_PHRASES_NYC = {"new", "york", "nyc", "manhattan", "brooklyn", "queens", "bronx", "staten", "island", "ny", "city", "upper", "lower", "side", "marks"}
COMMON_PHRASES = GLOBAL_FOOD.union(GLOBAL_TRANS_ADDR).union(GLOBAL_LOC).union(GLOBAL_WORDS).union(LOCAL_PHRASES_NYC).union(GLOBAL_SCHOOL_WRDS).union(GLOBAL_FIRE_STATION).union(GLOBAL_POLICE_STATION)

# Neighborhood and local area terms to ignore during name normalization
LOCAL_AREAS_NYC = [

    # Lower Manhattan / East Side
    "alphabet", "lower", "west", "north", "east", "south", "street", "side", "les",
    "two", "bridges", "chinatown", "nolita", "soho", "noho", "little", "italy",
    "bowery", "seaport", "civic", "center", "marks", "wall", "financial", "tribeca", "fidi", "delancey",
    "clinton", "canal",
    
    # Midtown / Gramercy / Chelsea
    "gramercy", "flatiron", "murray", "midtown",
    "koreatown", "garment", "district", "nomad", "chelsea", "hell", "hells",
    "hudson", "yards", "theater", "times", "square", "rockefeller", "kips", "turtle", "bay", "herald", "penn", "station", "empire",
    
    # Upper Manhattan
    "upper", "harlem", "spanish", "heights",
    "morningside", "hamilton", "inwood", "washington","manhattenville", "sugar", "hill", "dyckman", "fort", "george", "columbia",
    
    # Downtown / River Areas / Parks
    "tompkins", "river", "park",
    "stuytown", "stuyvesant", "oval", "union",
    "madison", "bryant", "central", "battery", "riverside", "fdr", "drive",
    
    # Outer Boroughs or Bordering Areas
    "brooklyn", "queens", "bushwick", "greenpoint", "williamsburg", "bed-stuy",
    "dumbo", "long island city", "astoria", "ridgewood",
]

NYC_BLACKLIST = COMMON_PHRASES.union(LOCAL_AREAS_NYC)


### NAME MATCHING HELPERS
def capitalize_str(name):
    parts = name.split()
    upper = [p.capitalize() for p in parts]
    return " ".join(upper)

def clean_name(name, lower = False):
    if not isinstance(name, str):
        return ""
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    # Remove accents/diacritics (e.g. é → e)
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    # Replace punctuation/symbols with space
    name = re.sub(r'[^a-zA-Z0-9\s]', ' ', name)
    return name.lower() if lower else name


# Simple number word mappings up to 100 (expand if needed)
NUM_WORDS = {
    "zero": 0, "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10,
    "eleven": 11, "eleventh": 11,
    "twelve": 12, "twelfth": 12,
    "thirteen": 13, "thirteenth": 13,
    "fourteen": 14, "fourteenth": 14,
    "fifteen": 15, "fifteenth": 15,
    "sixteen": 16, "sixteenth": 16,
    "seventeen": 17, "seventeenth": 17,
    "eighteen": 18, "eighteenth": 18,
    "nineteen": 19, "nineteenth": 19,
    "twenty": 20, "twentieth": 20,
    "thirty": 30, "thirtieth": 30,
    "forty": 40, "fortieth": 40,
    "fifty": 50, "fiftieth": 50,
    "sixty": 60, "sixtieth": 60,
    "seventy": 70, "seventieth": 70,
    "eighty": 80, "eightieth": 80,
    "ninety": 90, "ninetieth": 90
}

def words_to_number(tokens):
    """Convert tokens from their word form to digit-string form. 
    Ex: 
    words_to_number(['twenty', 'second']) = ['22']
    words_to_number(['seven'] = ['7'])
    """
    result = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in NUM_WORDS:
            val = NUM_WORDS[t]
            j = i + 1
            if j < len(tokens) and tokens[j] in NUM_WORDS:
                comb = val + NUM_WORDS[tokens[j]]
                result.append(str(comb))
                i += 2
                continue
            result.append(str(val))
        else:
            result.append(t)
        i += 1
    return result

def remove_common_words(name, blacklist = NYC_BLACKLIST, rem_common_phrases = True):
    name = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', name)
    parts = name.lower().split()
    parts = words_to_number(parts)
    parts = [p for p in parts if p not in blacklist]
    return " ".join(p for p in parts if not p.isdigit() and len(p) > 1)

def longest_common_substring(strs, min_len = 3, blacklist = NYC_BLACKLIST):
    """
    Find the longest common substring between a list of strings.
    Parameters:
        strs (list): List of 2+ strings to compare.
    """
    assert strs is not None and len(strs) >= 2, "At least two strings are required to find a common substring."

    strs = [remove_common_words(clean_name(s, True)) for s in strs]
    shortest_str = min(strs, key = len)
    longest_substring = ""

    n = len(shortest_str)
    for i in range(n):
        for j in range(i + 1, n + 1):
            substr = shortest_str[i:j]
            if all(substr in s for s in strs) and len(substr.strip()) >= min_len:
                if len(substr) > len(longest_substring):
                    longest_substring = substr.strip()
    return longest_substring

def choose_common_name_from_group(names, blacklist = NYC_BLACKLIST):
    assert names
    cleaned_names = [clean_name(n) for n in names]
    longest_substr = longest_common_substring(names)
    if len(longest_substr) < 3 or longest_substr.lower() in blacklist:
        return None
    
    # Compute address parts
    addr_parts = []
    for n in cleaned_names:
        parts = n.split()
        if len(parts) > 1 and parts[0].isdigit():
            num = parts[0]
            street = parts[1].lower()
            if street and num:
                addr_parts.append((num, street))
    street_set = {s for _, s in addr_parts}
    number_set = {n for n,_ in addr_parts}
    print('street_set: ' + str(street_set))
    print('number_set: ' + str(number_set))
    if len(street_set) == 1 and len(number_set) > 1:
        return None
    for og_name, cleaned_name in zip(names, cleaned_names):
        bklst_filtered_name = remove_common_words(cleaned_name, blacklist)
        if bklst_filtered_name == longest_substr.lower():
            return og_name
    return Counter(names).most_common(1)[0][0]


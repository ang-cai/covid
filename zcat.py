''' 
Create zcat matrix

Return n by n location zcat neighbor matrix where TRUE means that n location
borders or is connected by bridge/tunnel to the location at that index, and 
FALSE means otherwise.
'''
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

if __name__ == "__main__":
    nyc = gpd.read_file("nyc/nyc.shp")

    zcat_dict = {}
    # Make the matrix without considering bridges or tunnels
    for i in range(len(nyc.index)):
        zcat_col = nyc.touches(nyc.loc[i, "geometry"]).astype(int)
        zcat_dict[nyc.loc[i, "zcta"]] = zcat_col
    zcat_df = pd.DataFrame.from_dict(zcat_dict)
    zcat_df = zcat_df.set_index(nyc["zcta"])

    # Change values to true if there's a bridge or tunnel
    # connections = {}
    # for area in connections:
    #     zcat_df.loc[area, connections[area]] = 1
    #     zcat_df.loc[connections[area], area] = 1

    # Save to cvs
    zcat_df.to_csv("bordering_zcat.csv")
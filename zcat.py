''' 
Create zcat matrix

Return n by n location zcat neighbor matrix where 1 means that n location
borders or is connected by bridge/tunnel to the location at that index, and 
0 means otherwise.
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

    # Test print
    print(zcat_df.at["10305", "11209"])

    # Change values to true if there's a bridge or tunnel
    connections = {
        "10305": "11209", "10004": "11231", "10038": "11201", "11201": "10002", 
        "10002": "11211", "10017": "11101", "11101": "10044", "10044": "10021", 
        "11102": "10035", "11368": "11354", "11357": "10465", "10465": "11360", 
        "10475": "10464", "10037": "10451", "10454": "10035", "10451": "10039", 
        "10452": "10039", "10039": "10453", "10453": "10033", "10034": "10468",
        "10463": "10034", "11697": "11234"
    }
    for area in connections:
        zcat_df.loc[area, connections[area]] = 1
        zcat_df.loc[connections[area], area] = 1
        
    # Test print
    print(zcat_df.loc["10305", "11209"])

    # Save to cvs
    zcat_df.to_csv("connections_zcat.csv")
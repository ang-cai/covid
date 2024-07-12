''' 
Create zcat matrix

Return n by n location zcat neighbor matrix where 1 means that n location
borders or is connected by bridge/tunnel to the location at that index, and 
0 means otherwise.
'''
import geopandas as gpd
import pandas as pd

if __name__ == "__main__":
    nyc = gpd.read_file("nyc/nyc.shp")

    zcta_dict = {}
    # Make the matrix without considering bridges or tunnels
    for i in range(len(nyc.index)):
        zcta_col = nyc.touches(nyc.loc[i, "geometry"]).astype(int)
        zcta_dict[nyc.loc[i, "zcta"]] = zcta_col
    neighbors = pd.DataFrame.from_dict(zcta_dict)
    neighbors = neighbors.set_index(nyc["zcta"])

    # Change values to true if there's a bridge or tunnel
    connections = {
        "10305": "11209", # Verrazzano Narros Bridge
        # "10004": "11231", # Brooklyn Battery Tunnel
        # "10038": "11201", # Brooklyn Bridge
        # "11201": "10002", # Manhattan Bridge
        # "10002": "11211", # Williamsburg Bridge
        "10017": "11101", # Queens Midtown Tunnel
        "11101": "10044", # Queensboro Bridge
        "10044": "10021", # Queensboro Bridge
        "11101": "10044", # Roosevelt Island Bridge
        "11106": "10044", # Roosevelt Island Bridge
        "11102": "10035", # Robert F. Kennedy Bridge and Hell Gate Bridge
        "11357": "10465", # Bronx Whitestone Bridge
        "10465": "11360", # Throgs Neck Bridge
        "10037": "10451", # Third Avenue Bridge, Harlem River Lift Bridge, and Madison Avenue Bridge
        "10454": "10035", # Willis Avenue Bridge
        "10451": "10039", # 145th Street Bridge and Macombs Dam Bridge
        "10452": "10039", # Macombs Dam Bridge and High Bridge
        "10039": "10453", # Alexander Hamilton Bridge
        "10453": "10033", # Alexander Hamilton Bridge and Washington Bridge
        "10034": "10468", # University Heights Bridge
        "10463": "10034", # Broadway Bridge and Henry Hudson Bridge
        "11697": "11234", # Marine Parkway Bridge
        "11693": "11694"  # Cross Bay Bridge
    }
    for area in connections:
        neighbors.loc[area, connections[area]] = 1
        neighbors.loc[connections[area], area] = 1
        
    # Save to cvs
    neighbors.to_csv("exclude_manhattan_brooklyn_neighbor.csv")
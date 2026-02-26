# ======================================================
# AGRIGUARD AI DATA ENRICHMENT PIPELINE
# Climate + Soil + ET0 + Crop Season Inference
# ======================================================

import pandas as pd
import numpy as np
import requests
import io
import gzip
from datetime import datetime
import rasterio
import pynasapower as power
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================

INPUT_FILE = r"d:\New folder\India_Agri_Intelligence_Final\Multilingual_Expert_Advisory.csv"
OUTPUT_FILE = r"d:\New folder\India_Agri_Intelligence_Final\AgriGuard_Enriched_Dataset.csv"

LAT_COL = "Lat"
LON_COL = "Lon"
DATE_COL = "date"        # optional
YEAR_COL = "Year"        # fallback
CROP_COL = "Recommended_Crop"

# ======================================================
# 1️⃣ SEASON + MONTH EXTRACTION
# ======================================================

def add_month_and_season(df):

    if DATE_COL in df.columns:
        df["parsed_date"] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["month"] = df["parsed_date"].dt.month

    elif YEAR_COL in df.columns:
        df["month"] = np.nan
        print("WARNING: Only year present - month left empty")

    else:
        df["month"] = np.nan

    def season_map(m):
        if pd.isna(m):
            return np.nan
        m = int(m)
        if m in [6,7,8,9]:
            return "Kharif"
        if m in [10,11,12,1,2,3]:
            return "Rabi"
        if m in [4,5]:
            return "Zaid"
        return "Unknown"

    df["season"] = df["month"].apply(season_map)

    return df


# ======================================================
# 2️⃣ CROP SOWING WINDOW INFERENCE
# ======================================================

CROP_CALENDAR = {

    "rice": (6,9,"Kharif"),
    "paddy": (6,9,"Kharif"),
    "maize": (6,7,"Kharif"),
    "cotton": (6,7,"Kharif"),
    "soybean": (6,7,"Kharif"),
    "groundnut": (6,7,"Kharif"),

    "wheat": (10,12,"Rabi"),
    "gram": (10,12,"Rabi"),
    "mustard": (10,12,"Rabi"),
    "barley": (10,11,"Rabi"),

    "sugarcane": (10,3,"Year-round"),
    "banana": (1,12,"Year-round"),
}

def infer_sowing(crop, lat):

    if not isinstance(crop,str):
        return np.nan,np.nan,"Unknown"

    key = crop.lower().strip()

    for c in CROP_CALENDAR:
        if c in key:

            start,end,season = CROP_CALENDAR[c]

            # southern India adjustment
            if lat is not None and lat < 20 and season=="Kharif":
                start = start-1 if start>1 else 12
                end = end-1 if end>1 else 12

            return start,end,season

    return np.nan,np.nan,"Unknown"


# ======================================================
# 3️⃣ NASA POWER WEATHER + ET0
# ======================================================

def fetch_nasa_weather(lat, lon, date):

    params = ["PRECTOT","T2M","T2M_MAX","T2M_MIN","RH2M","WS2M","ALLSKY_SFC_SW_DWN"]

    d = pd.to_datetime(date)
    dstr = d.strftime("%Y%m%d")

    data = power.get_data(
        latitude=lat,
        longitude=lon,
        start=dstr,
        end=dstr,
        community="ag",
        parameters=params,
        temporal_average="daily"
    )

    weather = data["properties"]["parameter"]

    return weather


# ======================================================
# 4️⃣ FAO-56 ET0 Calculation
# ======================================================

def compute_eto(weather):

    try:
        tmax = weather["T2M_MAX"][0]
        tmin = weather["T2M_MIN"][0]
        tmean = weather["T2M"][0]
        rh = weather["RH2M"][0]
        wind = weather["WS2M"][0]
        rad = weather["ALLSKY_SFC_SW_DWN"][0] * 0.0864   # W/m2 → MJ/m2/day

        eto = (0.408*rad + 0.063*(tmean+17)*wind) / (tmean+273)

        return round(float(eto),2)

    except:
        return np.nan


# ======================================================
# 5️⃣ SOILGRIDS TEXTURE API
# ======================================================

def fetch_soil_texture(lat,lon):

    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    params = {
        "lat":lat,
        "lon":lon,
        "property":"sand,silt,clay",
        "depth":"0-5cm"
    }

    r = requests.get(url,params=params,timeout=20)

    if r.status_code!=200:
        return np.nan,np.nan,np.nan

    js = r.json()

    def extract(prop):
        try:
            return js["properties"][prop]["depths"][0]["values"]["mean"]
        except:
            return np.nan

    return extract("sand"),extract("silt"),extract("clay")


# ======================================================
# 6️⃣ CHIRPS SATELLITE RAINFALL
# ======================================================

def fetch_chirps_rain(lat,lon,date):

    d = pd.to_datetime(date)
    fname = f"CHIRPS_daily_{d.strftime('%Y%m%d')}.tif.gz"

    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{d.year}/{fname}"

    try:
        r = requests.get(url,timeout=60)
        if r.status_code!=200:
            return np.nan

        buf = io.BytesIO(gzip.decompress(r.content))

        with rasterio.open(buf) as src:
            row,col = src.index(lon,lat)
            val = src.read(1)[row,col]
            return float(val)

    except:
        return np.nan


# ======================================================
# 7️⃣ GROUNDWATER (GLOBAL FALLBACK)
# ======================================================

def fetch_groundwater(lat,lon):

    try:
        url = f"https://globalwaterwatch.org/api/point?lat={lat}&lon={lon}"
        r = requests.get(url,timeout=10)
        if r.status_code!=200:
            return np.nan

        return r.json().get("groundwater_depth_m",np.nan)

    except:
        return np.nan


# ======================================================
# 8️⃣ MAIN PIPELINE
# ======================================================

def enrich_dataset():

    df = pd.read_csv(INPUT_FILE)

    df = add_month_and_season(df)

    # new columns
    df["sowing_start_month"] = np.nan
    df["sowing_end_month"] = np.nan
    df["sowing_season_predicted"] = ""
    df["nasa_precip_mm"] = np.nan
    df["chirps_precip_mm"] = np.nan
    df["eto_mm_day"] = np.nan
    df["soil_sand_pct"] = np.nan
    df["soil_silt_pct"] = np.nan
    df["soil_clay_pct"] = np.nan
    df["groundwater_depth_m"] = np.nan

    for i,row in df.iterrows():

        lat = row[LAT_COL]
        lon = row[LON_COL]
        crop = row[CROP_COL]

        # choose date
        if DATE_COL in df.columns:
            date = row[DATE_COL]
        else:
            date = f"{int(row[YEAR_COL])}-06-15"

        # sowing inference
        s,e,season = infer_sowing(crop,lat)
        df.at[i,"sowing_start_month"] = s
        df.at[i,"sowing_end_month"] = e
        df.at[i,"sowing_season_predicted"] = season

        # NASA POWER
        try:
            weather = fetch_nasa_weather(lat,lon,date)
            df.at[i,"nasa_precip_mm"] = weather["PRECTOT"][0]
            df.at[i,"eto_mm_day"] = compute_eto(weather)
        except:
            pass

        # CHIRPS
        df.at[i,"chirps_precip_mm"] = fetch_chirps_rain(lat,lon,date)

        # SoilGrids
        sand,silt,clay = fetch_soil_texture(lat,lon)
        df.at[i,"soil_sand_pct"] = sand
        df.at[i,"soil_silt_pct"] = silt
        df.at[i,"soil_clay_pct"] = clay

        # Groundwater
        df.at[i,"groundwater_depth_m"] = fetch_groundwater(lat,lon)

        print(f"Processed row {i+1}/{len(df)}")

    df.to_csv(OUTPUT_FILE,index=False)
    print("\nSUCCESS: ENRICHMENT COMPLETE")
    print("Saved to:",OUTPUT_FILE)


# ======================================================
# RUN
# ======================================================

if __name__=="__main__":
    enrich_dataset()
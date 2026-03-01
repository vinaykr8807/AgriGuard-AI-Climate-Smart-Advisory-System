import sys
import os

# Add parent directory to path so it can find imd_scraper.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import imd_scraper
except ImportError:
    # If run from root
    sys.path.append('.')
    import imd_scraper

print('Seeding cache with state defaults and CSV data...')
seed_data = {}

# Add state-level defaults
for state, crops in imd_scraper._STATE_DEFAULT_CROPS.items():
    seed_data[state] = {'__default__': crops}

# Enrich from local CSV district-by-district
df = imd_scraper._get_csv_df()
if not df.empty:
    state_col = next((c for c in df.columns if 'state' in c.lower()), None)
    dist_col  = next((c for c in df.columns if 'district' in c.lower()), None)
    crop_col  = next((c for c in df.columns if 'crop' in c.lower()), None)
    if all([state_col, dist_col, crop_col]):
        for (st, dist), grp in df.groupby([state_col, dist_col]):
            st   = str(st).strip()
            dist = str(dist).strip()
            crops_list = sorted(set(
                str(c).strip().title()
                for c in grp[crop_col].dropna().tolist()
                if str(c).strip()
            ))
            if crops_list:
                seed_data.setdefault(st, {})[dist] = crops_list
        print(f'CSV enriched: {len(df)} rows processed')

imd_scraper._save_cache(seed_data)
status = imd_scraper.get_cache_status()
print(f'Cache seeded: {status["states"]} states, fresh={status["is_fresh"]}')

# Test lookups
tests = [
    ('Tamil Nadu', 'Ariyalur'),
    ('Uttar Pradesh', 'Meerut'),
    ('Maharashtra', 'Pune'),
    ('Punjab', 'Ludhiana'),
    ('Karnataka', 'Mysuru'),
]
for state, district in tests:
    result = imd_scraper.get_crops_for_location(state, district)
    print(f'  {state}/{district}: {result[:5]}')

"""
India Agriculture Intelligence - Feature Store Usage Examples
===============================================================

Quick-start code examples for working with the merged feature store.
"""

import pandas as pd
import json
from pathlib import Path

# ============================================================
# 1. LOADING THE FEATURE STORE
# ============================================================

def load_feature_store(use_parquet=True):
    """
    Load the merged feature store
    
    Args:
        use_parquet: If True, loads from .parquet (5x faster)
                    If False, loads from .csv (more compatible)
    
    Returns:
        pandas.DataFrame with 6,720 rows × 39 columns
    """
    store_dir = Path("india_feature_store")
    
    if use_parquet:
        df = pd.read_parquet(store_dir / "merged_feature_store.parquet")
        print(f"✓ Loaded {len(df):,} rows from Parquet (fast)")
    else:
        df = pd.read_csv(store_dir / "merged_feature_store.csv")
        print(f"✓ Loaded {len(df):,} rows from CSV")
    
    return df


# ============================================================
# 2. QUERY BY LOCATION AND CROP
# ============================================================

def get_advisories_for_district(df, state, district, crop=None, year=None):
    """
    Get advisories for a specific district
    
    Example:
        advisories = get_advisories_for_district(
            df, 
            state='Maharashtra', 
            district='Yavatmal',
            crop='Cotton',
            year=2024
        )
    """
    # Normalize inputs for matching
    state_norm = state.lower().strip()
    district_norm = district.lower().strip()
    
    # Build query
    query = (df['state_norm'] == state_norm) & (df['district_norm'] == district_norm)
    
    if crop:
        crop_norm = crop.lower().strip()
        query = query & (df['crop_norm'] == crop_norm)
    
    if year:
        query = query & (df['year'] == year)
    
    results = df[query]
    
    print(f"Found {len(results)} advisory records")
    return results


# ============================================================
# 3. MULTILINGUAL ADVISORY RETRIEVAL
# ============================================================

def get_advisory_in_language(df, advisory_id, language='hindi'):
    """
    Retrieve advisory in specified language
    
    Args:
        advisory_id: e.g., 'ADV_0001'
        language: 'hindi', 'telugu', 'marathi', 'punjabi', 'tamil', or 'english'
    
    Returns:
        str: Advisory text in requested language
    """
    row = df[df['advisory_id'] == advisory_id]
    
    if row.empty:
        return None
    
    row = row.iloc[0]
    
    if language == 'english':
        return row['advisory_text_en']
    else:
        col = f'advisory_{language.lower()}'
        if col in df.columns:
            return row[col]
        else:
            print(f"Language '{language}' not available")
            return None


def print_advisory_all_languages(df, advisory_id):
    """Print advisory in all available languages"""
    row = df[df['advisory_id'] == advisory_id]
    
    if row.empty:
        print(f"Advisory {advisory_id} not found")
        return
    
    row = row.iloc[0]
    
    print(f"\n{'='*80}")
    print(f"Advisory ID: {advisory_id}")
    print(f"Location: {row['district']}, {row['state']}")
    print(f"Crop: {row['recommended_crop']}")
    print(f"Year: {row['year']}")
    print(f"{'='*80}\n")
    
    print("ENGLISH:")
    print(f"  {row['advisory_text_en']}\n")
    
    for lang in ['hindi', 'telugu', 'marathi', 'punjabi', 'tamil']:
        col = f'advisory_{lang}'
        if col in df.columns and pd.notna(row[col]):
            print(f"{lang.upper()}:")
            print(f"  {row[col]}\n")


# ============================================================
# 4. CLIMATE-BASED FILTERING
# ============================================================

def find_suitable_regions(df, crop, min_rainfall=None, max_rainfall=None, 
                          min_ndvi=None, soil_ph_range=None, year=2024):
    """
    Find regions suitable for a crop based on climate criteria
    
    Example:
        # Find high-rainfall regions for rice
        suitable = find_suitable_regions(
            df,
            crop='Rice',
            min_rainfall=1200,
            min_ndvi=0.6,
            soil_ph_range=(6.0, 7.5),
            year=2024
        )
    """
    crop_norm = crop.lower().strip()
    
    query = (df['crop_norm'] == crop_norm) & (df['year'] == year)
    
    if min_rainfall:
        query = query & (df['rainfall_imd_mm'] >= min_rainfall)
    
    if max_rainfall:
        query = query & (df['rainfall_imd_mm'] <= max_rainfall)
    
    if min_ndvi:
        query = query & (df['ndvi_vegetation_index'] >= min_ndvi)
    
    if soil_ph_range:
        query = query & (df['soil_ph'] >= soil_ph_range[0]) & (df['soil_ph'] <= soil_ph_range[1])
    
    results = df[query]
    
    print(f"Found {len(results)} suitable regions for {crop}")
    return results[['state', 'district', 'rainfall_imd_mm', 'ndvi_vegetation_index', 
                    'soil_ph', 'advisory_text_en']].drop_duplicates()


# ============================================================
# 5. TEMPORAL ANALYSIS
# ============================================================

def analyze_climate_trends(df, state, district, start_year=2015, end_year=2024):
    """
    Analyze climate trends for a district over time
    """
    district_norm = district.lower().strip()
    state_norm = state.lower().strip()
    
    query = (
        (df['district_norm'] == district_norm) & 
        (df['state_norm'] == state_norm) &
        (df['year'] >= start_year) &
        (df['year'] <= end_year)
    )
    
    trends = df[query].groupby('year').agg({
        'rainfall_imd_mm': 'mean',
        'mean_temp_historical': 'mean',
        'ndvi_vegetation_index': 'mean',
        'soil_moisture_historical': 'mean'
    }).round(2)
    
    print(f"\nClimate Trends for {district}, {state} ({start_year}-{end_year})")
    print("="*80)
    print(trends)
    
    return trends


# ============================================================
# 6. ADVISORY RECOMMENDATION SYSTEM
# ============================================================

def recommend_advisory(df, user_query, top_k=3):
    """
    Recommend advisories based on user query
    
    This is a simple keyword-based system. For production,
    use semantic embeddings (e.g., all-MiniLM-L6-v2)
    
    Example:
        advisories = recommend_advisory(
            df,
            user_query="high rainfall cotton cultivation black soil",
            top_k=3
        )
    """
    # Simple keyword matching (upgrade to embeddings in production)
    keywords = user_query.lower().split()
    
    def score_advisory(text):
        if pd.isna(text):
            return 0
        text_lower = text.lower()
        return sum(1 for kw in keywords if kw in text_lower)
    
    df_copy = df[df['advisory_id'].notna()].copy()
    df_copy['relevance_score'] = df_copy['advisory_text_en'].apply(score_advisory)
    
    top_advisories = df_copy.nlargest(top_k, 'relevance_score')
    
    return top_advisories[['advisory_id', 'state', 'district', 'recommended_crop', 
                           'advisory_text_en', 'relevance_score']]


# ============================================================
# 7. DATA QUALITY CHECKS
# ============================================================

def validate_data_quality(df):
    """Run data quality checks"""
    print("\n" + "="*80)
    print("DATA QUALITY VALIDATION REPORT")
    print("="*80 + "\n")
    
    # 1. Check for missing advisories
    advisory_coverage = (df['advisory_id'].notna()).sum() / len(df) * 100
    print(f"1. Advisory Coverage: {advisory_coverage:.1f}%")
    assert advisory_coverage == 100.0, "❌ Missing advisories detected!"
    print("   ✓ PASS: All rows have advisories\n")
    
    # 2. Translation coverage
    print("2. Translation Coverage:")
    for lang in ['hindi', 'telugu', 'marathi', 'punjabi', 'tamil']:
        col = f'advisory_{lang}'
        if col in df.columns:
            coverage = df[col].notna().sum() / len(df) * 100
            print(f"   {lang.capitalize()}: {coverage:.1f}%")
    print()
    
    # 3. Numeric ranges
    print("3. Numeric Field Validation:")
    
    checks = [
        ('rainfall_imd_mm', 0, 10000, 'mm'),
        ('soil_ph', 3, 12, 'pH'),
        ('ndvi_vegetation_index', -1, 1, 'index'),
        ('soil_moisture_historical', 0, 1, 'index'),
        ('mean_temp_historical', -10, 55, '°C')
    ]
    
    for field, min_val, max_val, unit in checks:
        if field in df.columns:
            valid = df[field].between(min_val, max_val).all()
            status = "✓ PASS" if valid else "❌ FAIL"
            print(f"   {field}: {status} (range: {min_val}-{max_val} {unit})")
    
    print("\n" + "="*80)
    print("✅ DATA QUALITY VALIDATION COMPLETE")
    print("="*80 + "\n")


# ============================================================
# 8. EXPORT FILTERED DATA
# ============================================================

def export_subset(df, output_file, **filters):
    """
    Export filtered subset of data
    
    Example:
        export_subset(
            df,
            'cotton_maharashtra_2024.csv',
            state='Maharashtra',
            crop='Cotton',
            year=2024
        )
    """
    query = pd.Series([True] * len(df))
    
    if 'state' in filters:
        query = query & (df['state_norm'] == filters['state'].lower())
    
    if 'crop' in filters:
        query = query & (df['crop_norm'] == filters['crop'].lower())
    
    if 'year' in filters:
        query = query & (df['year'] == filters['year'])
    
    subset = df[query]
    subset.to_csv(output_file, index=False)
    print(f"✓ Exported {len(subset)} rows to {output_file}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INDIA AGRICULTURE INTELLIGENCE - FEATURE STORE EXAMPLES")
    print("="*80 + "\n")
    
    # Load data
    df = load_feature_store(use_parquet=True)
    
    print("\n--- Example 1: Get advisories for a district ---")
    advisories = get_advisories_for_district(
        df,
        state='Maharashtra',
        district='Yavatmal',
        crop='Cotton',
        year=2024
    )
    if len(advisories) > 0:
        print(f"\nSample advisory (English):")
        print(advisories.iloc[0]['advisory_text_en'][:200], "...")
    
    print("\n--- Example 2: Multilingual advisory ---")
    # Get first advisory ID
    sample_id = df[df['advisory_id'].notna()].iloc[0]['advisory_id']
    print(f"\nRetrieving {sample_id} in Hindi...")
    hindi_text = get_advisory_in_language(df, sample_id, 'hindi')
    if hindi_text:
        print(hindi_text[:150], "...")
    
    print("\n--- Example 3: Find suitable regions for Rice ---")
    suitable = find_suitable_regions(
        df,
        crop='Rice',
        min_rainfall=1000,
        min_ndvi=0.5,
        year=2024
    )
    print(f"\nTop 5 suitable districts:")
    if len(suitable) > 0:
        print(suitable.head()[['state', 'district', 'rainfall_imd_mm', 'ndvi_vegetation_index']])
    
    print("\n--- Example 4: Climate trends ---")
    if len(df) > 0:
        sample_district = df.iloc[0]['district']
        sample_state = df.iloc[0]['state']
        trends = analyze_climate_trends(df, sample_state, sample_district, 2020, 2024)
    
    print("\n--- Example 5: Data quality validation ---")
    validate_data_quality(df)
    
    print("\n✅ All examples completed successfully!")
    print("\nNext steps:")
    print("  1. Integrate live weather data (OpenWeather/Open-Meteo)")
    print("  2. Implement semantic search using embeddings")
    print("  3. Create train/test split for ML models")
    print("  4. Build API endpoints for real-time queries")

import pandas as pd
import os
import time
from tqdm import tqdm
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_CSV = "India_Agri_Intelligence_Final/Smart_Advisory_Reports_All.csv"
OUTPUT_CSV = "India_Agri_Intelligence_Final/Multilingual_Expert_Advisory.csv"

# Target Languages (ISO codes)
# Prioritized for speed - reduce to core 3 if needed, or keeping all but increasing workers
LANGUAGES = {
    'Advisory_Hindi': 'hi',
    'Advisory_Telugu': 'te',
    'Advisory_Marathi': 'mr',
    'Advisory_Punjabi': 'pa',
    'Advisory_Tamil': 'ta',
    'Advisory_Haryanvi': 'hi' # Mapping Haryanvi to Hindi as Google Translator lacks 'har' support
}

MAX_WORKERS = 30 # Aggressive parallelism for Google Translate
BATCH_SAVE_SIZE = 50 # Save every 50 records to reduce I/O overhead

def translate_text(text, target_lang):
    """Translates text using Google Translator via deep-translator."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    try:
        # We use a short retry logic for network stability
        for _ in range(3):
            try:
                translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
                return translated
            except:
                time.sleep(1)
        return "Translation Error"
    except Exception:
        return ""

def process_row_multilingual(row_data):
    """Translates the 'Expert_Advisory' column for a single row into all target languages."""
    idx = row_data[0]
    advisory = row_data[1]
    
    translations = {'index': idx}
    for col_name, lang_code in LANGUAGES.items():
        translations[col_name] = translate_text(advisory, lang_code)
    
    return translations

def main():
    print("🇮🇳 INDIC LANGUAGE ENGINE v1.1 (Live Watch Mode)")
    print("👀 Monitoring 'Smart_Advisory_Reports_All.csv' for new advice...")
    
    while True:
        if not os.path.exists(INPUT_CSV):
            print("⏳ Waiting for input file to be created...")
            time.sleep(10)
            continue

        try:
            # Load English advisory data with a slight delay to avoid 'File in use' errors
            df = pd.read_csv(INPUT_CSV)
            
            if 'Expert_Advisory' not in df.columns or df['Expert_Advisory'].isna().all():
                time.sleep(15) 
                continue

            target_col = 'Expert_Advisory'

            # Load or initialize output
            if os.path.exists(OUTPUT_CSV):
                df_out = pd.read_csv(OUTPUT_CSV)
                for lang_col in LANGUAGES.keys():
                    if lang_col not in df_out.columns:
                        df_out[lang_col] = None
                    # Treat existing 'Error' results as None to trigger re-translation
                    err_mask = df_out[lang_col].astype(str).str.contains('Error', case=False)
                    df_out.loc[err_mask, lang_col] = None
            else:
                df_out = df.copy()
                for lang_col in LANGUAGES.keys():
                    df_out[lang_col] = None

            # SYNC LOGIC: If English advice has been updated/fixed in source, update it here too
            # We check for mismatches between df and df_out for the target_col
            for idx in df.index:
                if idx < len(df_out):
                    if df.at[idx, target_col] != df_out.at[idx, target_col]:
                        df_out.at[idx, target_col] = df.at[idx, target_col]
                        # Reset Indic columns for this row so they get re-translated
                        for lang_col in LANGUAGES.keys():
                            df_out.at[idx, lang_col] = None

            # Identify rows ready for translation
            ready_to_translate = df[df[target_col].notna() & ~df[target_col].astype(str).str.contains('Error', case=False)].index.tolist()
            missing_translation = df_out[df_out[list(LANGUAGES.keys())].isna().any(axis=1)].index.tolist()
            
            # Intersection: Indices that are fixed/ready AND missing translations
            work_indices = list(set(ready_to_translate) & set(missing_translation))

            if not work_indices:
                # Check if the main process is actually finished
                total_english_done = df[target_col].count()
                if total_english_done == len(df):
                    print("🎉 All records translated and synced! Task complete.")
                    break
                
                print(f"😴 Waiting for new records (Progress: {total_english_done}/{len(df)} English advice ready)...")
                time.sleep(60)
                continue

            print(f"🚀 Found {len(work_indices)} new records. Translating...")

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                work_items = [(idx, df.at[idx, target_col]) for idx in work_indices]
                futures = {executor.submit(process_row_multilingual, item): item[0] for item in work_items}
                
                processed_in_batch = 0
                for future in tqdm(as_completed(futures), total=len(work_items), desc="Translating"):
                    result = future.result()
                    idx = result['index']
                    for col_name in LANGUAGES.keys():
                        df_out.at[idx, col_name] = result[col_name]
                    
                    processed_in_batch += 1
                    if processed_in_batch % BATCH_SAVE_SIZE == 0:
                        df_out.to_csv(OUTPUT_CSV, index=False)

            # Save final state of the batch
            df_out.to_csv(OUTPUT_CSV, index=False)
            print(f"📊 Progress checkpoint saved. Current translation count: {df_out['Advisory_Hindi'].count()}")

        except PermissionError:
            print("⚠️ File currently being written by main engine. Retrying in 10s...")
            time.sleep(10)
        except Exception as e:
            print(f"⚠️ Error: {e}. Retrying...")
            time.sleep(30)

if __name__ == "__main__":
    main()

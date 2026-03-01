import pandas as pd
import os
import json
import time
from tqdm import tqdm
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
INPUT_CSV = "India_Agri_Intelligence_Final/Unified_Decadal_Master_2015_2024.csv"
OUTPUT_CSV = "India_Agri_Intelligence_Final/Smart_Advisory_Reports_All.csv"

# GROQ API KEYS (Rotating Pool for Parallel Speed)
# Set one or more comma-separated keys in GROQ_API_KEYS.
GROQ_API_KEYS = [
    key.strip() for key in os.getenv("GROQ_API_KEYS", "").split(",") if key.strip()
]

# API Settings
BATCH_SIZE = 15   
MAX_WORKERS = 2    # Balanced for 2 active keys
DELAY_SECONDS = 15 # Adjusted for larger batch size to stay under TPM limits

# Initialize Client Pool
if not GROQ_API_KEYS:
    raise ValueError("No Groq API keys found. Set GROQ_API_KEYS environment variable.")

CLIENTS = [Groq(api_key=key) for key in GROQ_API_KEYS]
MODEL = "llama-3.1-8b-instant" 

def get_expert_advice_batch(records_batch, client):
    """Processes a small batch for speed + reliability with aggressive retry logic."""
    batch_json = records_batch.to_json(orient='records')
    
    prompt = f"""
    You are an expert Indian Agricultural Scientist. 
    Provide a JSON ARRAY of strings. Each string must be exactly 2 sentences of professional advice for the corresponding record.
    
    Data:
    {batch_json}
    
    Return ONLY a JSON array of strings. No keys, no markdown.
    """
    
    for attempt in range(7):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000,
            )
            content = completion.choices[0].message.content.strip()
            # Clean possible markdown
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"): content = content[4:]
            
            advice_list = json.loads(content)
            if len(advice_list) >= len(records_batch):
                time.sleep(DELAY_SECONDS) # Respect rate limits
                return advice_list[:len(records_batch)]
            else:
                # Fill missing indices with None so they remain as NaN in CSV
                advice_list.extend([None] * (len(records_batch) - len(advice_list)))
                time.sleep(DELAY_SECONDS) 
                return advice_list
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate limit" in err_msg.lower():
                print(f"⚠️ Rate limit hit. Cooling down for 60s (Attempt {attempt+1}/7)...")
                time.sleep(60)
            else:
                print(f"⚠️ API Error: {err_msg[:100]} (Attempt {attempt+1}/7)...")
                time.sleep(10 * (attempt + 1))
    return [None] * len(records_batch)

def main():
    print(f"🚀 AI ADVISORY ENGINE v4.0 (Multi-Key Parallel Mode)")
    print(f"🔑 Using {len(CLIENTS)} Rotating API Keys")
    
    if not os.path.exists(INPUT_CSV):
        print(f"❌ ERROR: {INPUT_CSV} not found!")
        return

    df = pd.read_csv(INPUT_CSV)
    total_records = len(df)
    
    # --- ID SYNCHRONIZATION ---
    if 'Advisory_ID' not in df.columns:
        print("🆔 Generating Advisory_ID for records...")
        df.insert(0, 'Advisory_ID', [f"ADV_{i:04d}" for i in range(1, total_records + 1)])

    # --- RESUME & ERROR FIX LOGIC ---
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        
        # If advisory_id does not exist in the Smart_Advisory file, attempt to inject it
        if 'Advisory_ID' not in df_existing.columns:
            print("⚠️ advisory_id missing in output file. Aligning by record position...")
            if len(df_existing) == total_records:
                df_existing.insert(0, 'Advisory_ID', df['Advisory_ID'])
            else:
                print("❌ FATAL: Record counts mismatch. Cannot safely resume without Advisory_ID.")
                return

        if 'Expert_Advisory' in df_existing.columns:
            df['Expert_Advisory'] = df_existing['Expert_Advisory']
            # Treat 'Error' strings as NaN so they get re-processed
            error_mask = df['Expert_Advisory'].astype(str).str.contains('Error', case=False)
            df.loc[error_mask, 'Expert_Advisory'] = None
            
            done_count = df['Expert_Advisory'].count()
            error_count = error_mask.sum()
            print(f"📂 Progress: {done_count} valid records found. {error_count} errors flagged for re-processing.")
        else:
            df['Expert_Advisory'] = None
    else:
        df['Expert_Advisory'] = None

    # Identify batches that need processing (Missing or Error)
    print(f"📁 Identifying records to process in batches of {BATCH_SIZE}...")

    # Define processing task for parallel execution
    def process_batch_wrapper(batch_info, client_idx):
        batch_idx, batch_df = batch_info
        client = CLIENTS[client_idx % len(CLIENTS)]
        return batch_idx, get_expert_advice_batch(batch_df, client)

    # Group missing/error work into batches
    work_batches = []
    for i in range(0, total_records, BATCH_SIZE):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        if batch_df['Expert_Advisory'].isna().any():
            work_batches.append((i, batch_df))

    if not work_batches:
        print("✅ All records are complete and error-free! Nothing to do.")
        return

    # Parallel processing of batches
    count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_batch_wrapper, wb, i): i for i, wb in enumerate(work_batches)}
        
        for future in tqdm(as_completed(futures), total=len(work_batches), desc="Groq Processing"):
            start_idx, advices = future.result()
            
            for offset, advice in enumerate(advices):
                target_idx = start_idx + offset
                if target_idx < total_records and advice is not None:
                    df.at[target_idx, 'Expert_Advisory'] = advice
            
            count += 1
            # Save every batch for immediate progress visibility
            df.to_csv(OUTPUT_CSV, index=False)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS: Final AI-Augmented Report saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

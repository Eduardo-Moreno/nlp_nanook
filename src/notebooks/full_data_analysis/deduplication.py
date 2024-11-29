# Save this entire script as 'deduplication.py'
from rapidfuzz import fuzz, process
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import math

def prefilter_messages(messages):
    length_groups = {}
    
    for idx, msg in enumerate(messages):
        if pd.isna(msg) or msg.strip() == '':
            continue
            
        msg_len = len(msg)
        min_len = math.floor(msg_len * 0.85)
        max_len = math.ceil(msg_len * 1.15)
        length_key = f"{min_len}-{max_len}"
        
        for length in range(min_len, max_len + 1):
            group_key = f"{length // 10 * 10}"
            if group_key not in length_groups:
                length_groups[group_key] = set()
            length_groups[group_key].add(idx)
    
    return length_groups

def process_message_group(args):
    group_indices, messages, similarity_threshold = args
    indices_to_drop = set()

    group_messages = [messages[idx] for idx in group_indices if pd.notna(messages[idx]) and messages[idx].strip()]
    group_indices_filtered = [idx for idx in group_indices if pd.notna(messages[idx]) and messages[idx].strip()]

    similarities = process.cdist(group_messages, group_messages, scorer=fuzz.ratio, workers=-1)

    for i in range(len(group_messages)):
        if group_indices_filtered[i] in indices_to_drop:
            continue
        for j in range(i + 1, len(group_messages)):
            if similarities[i][j] >= similarity_threshold:
                idx_j = group_indices_filtered[j]
                if len(group_messages[i]) >= len(group_messages[j]):
                    indices_to_drop.add(idx_j)
                else:
                    indices_to_drop.add(group_indices_filtered[i])
                    break
    return indices_to_drop

def remove_similar_messages_parallel(df, similarity_threshold=95, n_processes=None):
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    df_clean = df.copy()
    messages = df_clean['Message'].tolist()
    
    print("Pre-filtering messages...")
    length_groups = prefilter_messages(messages)
    
    group_args = [
        (indices, messages, similarity_threshold)
        for indices in length_groups.values()
        if len(indices) > 1
    ]
    
    print(f"Processing {len(group_args)} groups using {n_processes} processes...")
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_message_group, group_args),
            total=len(group_args),
            desc="Processing message groups"
        ))
    
    indices_to_drop = set().union(*results)
    df_clean = df_clean.drop(index=list(indices_to_drop))
    df_clean = df_clean.reset_index(drop=True)
    
    print(f"Removed {len(indices_to_drop)} similar messages.")
    return df_clean

def analyze_duplicates(df_original, df_deduplicated):
    total_original = len(df_original)
    total_dedup = len(df_deduplicated)
    removed = total_original - total_dedup
    
    print(f"\nDeduplication Analysis:")
    print(f"Original messages: {total_original:,}")
    print(f"Unique messages: {total_dedup:,}")
    print(f"Removed messages: {removed:,}")
    print(f"Reduction percentage: {(removed/total_original)*100:.2f}%")

if __name__ == '__main__':
    # Example usage
    test_subset = selected_data.head(100)  # Try with 100 rows first
    test_dedup = remove_similar_messages_parallel(test_subset)
    analyze_duplicates(test_subset, test_dedup)
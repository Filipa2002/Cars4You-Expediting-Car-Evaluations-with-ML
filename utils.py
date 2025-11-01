import numpy as np
import pandas as pd
import unicodedata
import re
import difflib

# Normalization function
def norm(s):
    if pd.isnull(s) or s == '':
        return None
    # Separate accents, lowercase, removes spaces
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    # Remove accents, keeping only the base characters
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Replace non-alphanumeric (e.g. punctuation) characters with a single space
    return re.sub(r"[^a-z0-9]+", " ", s).strip()



# Finds the best match using difflib similarity ratio
def get_best_match(query, choices, min_score=0.6):
    # Handle None or NaN safely
    if query is None or pd.isna(query):
        return []

    # Ensure input is string
    query = str(query)
    
    # Initialize variables to track the best score and matches
    best_score = 0
    best_matches = []

    # Special condition for short words: allow all matches
    if len(query) <= 3:
        min_score = 0.0  # ignore cutoff for short strings

    # Iterate over all candidate choices
    for choice in choices:
        # Compute similarity ratio between 0 and 1
        score = difflib.SequenceMatcher(None, query, choice).ratio()
        # Only consider matches that meet or exceed the minimum score
        if score >= min_score:
            if score > best_score:
                best_score = score
                best_matches = [choice]
            elif score == best_score:
                best_matches.append(choice)

    return best_matches



def analyze_flagged_data(df, flag_column):
    # Check if the flag column exists in the DataFrame to avoid errors
    if flag_column not in df.columns:
        print(f"Error: Column '{flag_column}' not found in the DataFrame.")
        return pd.DataFrame()

    # Define the list of flags that store lists
    list_based_flags = [
        'multiple_models', 
        'multiple_models_and_brands', 
        'multiple_brands', 
        'multiple_models_from_a_diff_brand'
    ]

    if flag_column in list_based_flags:
        # If flag_column is one of the list flags, use the .notnull() method to find flagged rows
        flagged_rows = df[df[flag_column].notnull()]
    else:
        # Otherwise, assume it's a binary flag
        flagged_rows = df[df[flag_column] == 1]
    

    # Create a summary DataFrame. Include the flag column itself for context,
    # which is useful for both types of flags.
    summary = flagged_rows[['Brand', 'model', flag_column]]
    
    # Print the total count
    print(f"Count of rows flagged in '{flag_column}': {summary.shape[0]}")
    
    # Return the summary DataFrame for display
    return summary

def print_dup_info(df, exclude_groups=None, name="df"):
    print(f"Total duplicates in {name}: {df.duplicated().sum()}")
    if not exclude_groups:
        return
    for cols in exclude_groups:
        cols = list(cols)
        n = df.drop(columns=cols, errors='ignore').duplicated().sum()
        print(f"Duplicates without {' and '.join(cols)} in {name}: {n}")

def describe_num_with_skew_kurtosis(df):
    num = df.select_dtypes(include='number')
    out = df.describe().T
    out['skew'] = num.skew()    
    out['kurtosis'] = num.kurtosis()  # Fisher (normal=0)
    return out       

def describe_cats(df, include=('object')):
    cats = df.select_dtypes(include=include)
    out = cats.describe().T
    out['top_freq_ratio'] = out['freq'] / out['count']
    return out


def apply_to_datasets(func, *dfs, **kwargs):
    return [func(df, **kwargs) for df in dfs]


def var_report(df, col, target=None, label=None, decimals=2):
    pfx = f"[{label}] " if label else ""
    s = pd.to_numeric(df[col], errors="coerce")

    neg  = (s < 0).mean() * 100
    pos  = (s > 0).mean() * 100
    zero = (s == 0).mean() * 100
    nan  = s.isna().mean() * 100

    print(f"{pfx}Negative {col} values: {neg:.{decimals}f}%")
    print(f"{pfx}Positive {col} values: {pos:.{decimals}f}%")
    print(f"{pfx}Zero {col} values: {zero:.{decimals}f}%")
    print(f"{pfx}NaN  {col} values: {nan:.{decimals}f}%")

    if target and target in df.columns:
        t = pd.to_numeric(df[target], errors="coerce")
        thr = abs(s.min(skipna=True))
        g_neg  = s < 0
        g_rng  = (s > 0) & (s <= thr)

        med_neg, mean_neg, std_neg = t[g_neg].median(), t[g_neg].mean(), t[g_neg].std()
        med_rng, mean_rng, std_rng = t[g_rng].median(), t[g_rng].mean(), t[g_rng].std()

        print(f"{pfx}{target} stats where {col} < 0:")
        print(f"{pfx}Median: {med_neg}  Mean: {mean_neg}  Std: {std_neg}")
        print(f"{pfx}{target} stats where 0 < {col} ≤ abs(min({col})):")
        print(f"{pfx}Median: {med_rng}  Mean: {mean_rng}  Std: {std_rng}")
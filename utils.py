import numpy as np
import pandas as pd
import unicodedata
import re
import difflib
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from wordcloud import WordCloud
from scipy.stats import ttest_ind, chi2_contingency

palette = ['#5C4212','#a92f02', '#a55b1bf9', '#b08972', '#e3a76c', '#e5c120','#f39c06','#f2e209']

def plot_box_hist(df, cols, title, n_bins=50):
    """
    The function creates a multi-row figure where each feature is represented by 
    a box plot (top row) and a histogram (bottom row) arranged in two columns.
    The histogram also includes vertical lines for mean, median, Q1, and Q3.

    Parameters
    ----------
    df: DataFrame
        The DataFrame containing the data to plot.
    cols: List
        A list of features to be plotted.
    title: str
        A string suffix to be added to the main plot title.
    n_bins: int, optional
        The number of bins for the histogram. Defaults to 50.
    
    Returns
    -------
    None
    """
    # Calculate number of subplot rows 
    sp_rows = ceil(len(cols) / 2)
    
    # Create figure with double rows: one for boxplots, one for histograms
    fig, axes = plt.subplots(
        sp_rows * 2, 2,
        figsize=(14, sp_rows * 4.5),
        gridspec_kw={'height_ratios': [0.3, 0.85] * sp_rows},
        tight_layout=False
    )
    
    # Ensure axes is a 2D array for consistent indexing
    axes = np.array(axes).reshape(sp_rows * 2, 2)

    # Loop through each feature to plot
    for idx, feat in enumerate(cols):
        row, col = (idx // 2) * 2, idx % 2 # Determine that variable's plot position
        ax_box, ax_hist = axes[row, col], axes[row + 1, col]

        # Boxplot
        sns.boxplot(x=df[feat], color=palette[-2], ax=ax_box, orient='h', width=0.58)
        
        # Remove axis labels
        ax_box.set_xlabel(None)
        ax_box.set_ylabel(None)
        # Remove top and right borders
        sns.despine(ax=ax_box, top=True, right=True)

        # Histogram
        sns.histplot(df[feat], bins=n_bins, color=palette[-2], kde=True, stat='percent', alpha=0.6, ax=ax_hist)
            
        # Calculate and plot statistics
        stats = {
            'mean': df[feat].mean(),
            'median': df[feat].median(),
            'q1': df[feat].quantile(0.25),
            'q3': df[feat].quantile(0.75)
        }
        # Plot vertical lines for each statistic
        for stat_name, stat_val in stats.items():
            ax_hist.axvline(
                stat_val, color=palette[list(stats.keys()).index(stat_name)], linestyle='--',
                linewidth=1.5, alpha=0.8, label=f'{stat_name.capitalize()}: {stat_val:.1f}'
            )
        # Remove top and right borders
        ax_hist.legend(loc='best', fontsize=7)
        sns.despine(ax=ax_hist, top=True, right=True)
    
    # Remove any unused subplots
    for i in range(len(cols) * 2+1, sp_rows * 4):
            fig.delaxes(axes.flatten()[i]) 

    # Title
    plt.suptitle("Distribution of Numerical Car Features " + title, fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_bar_wordcloud(df, cols):
    """
    For each column (categorical feature) provided:
    - Bar Plot (Left): Displays the percentage of the top 10 most frequent unique values.
    - Word Cloud (Right): Visualizes all unique values, where the size of the 
       text is proportional to the frequency of the value.

    Parameters
    ----------
    df : DataFrame
        The DataFrame we want to plot the categorical data from.
    cols : list
        A list of categorical features to plot.
    
    Returns
    -------
    None
    """
    # Loop through each column to create individual plots
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(16,6), gridspec_kw={'width_ratios':[1,1]}) #1 row, 2 columns
        
        # Barplot
        ax = axes[0]
        
        value_counts = df[col].value_counts(normalize=True) * 100 # percentage of each unique value in the column
        # Up to top 10 unique values
        if len(value_counts) > 10:
            value_counts = value_counts.head(10)

        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, color=palette[2])

        # Add percentage above each bar
        for i, v in enumerate(value_counts.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9)
        
        ax.set_title(f"{col} - Barplot", fontsize=12, fontweight='bold')
        ax.set_xlabel("Percentage")
        ax.set_ylabel(None)
        sns.despine(ax=ax, top=True, right=True)

        # WordCloud
        ax_wc = axes[1]
        word_counts = df[col].value_counts()
        wc = WordCloud(width=800, height=400,
                       background_color='white',
                       colormap='YlOrBr',
                       contour_width=1).generate_from_frequencies(word_counts)
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        ax_wc.set_title(f"{col} - WordCloud", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()


def missing_data(train_df, val_df=None, test_df=None):
    """
    Calculates the total count (n) and percentage (%) of missing values (NaN) 
    for each column across one or more specified DataFrames (Train, Validation, Test).
    
    Parameters
    ----------
    train_df : DataFrame
        The training DataFrame.
    val_df : DataFrame, optional
        The validation DataFrame.
    test_df : DataFrame, optional
        The test DataFrame.
    
    Returns
    -------
    pd.DataFrame
        A summary table where each row is a feature and columns show the count 
        ('n') and percentage ('%') of NaN values for each provided DataFrame 
        (e.g., 'Train n', 'Train %').
    """

    frames = [("Train", train_df), ("Validation", val_df), ("Test", test_df)]
    out = pd.DataFrame()
    for name, df in frames:
        if df is None:
            continue
        out[f"{name} n"] = df.isna().sum()
        out[f"{name} %"] = (df.isna().mean() * 100).round(2)
    return out

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def norm(s):
    """
    Standardizes a string by converting it to lowercase, removing accents, 
    stripping whitespace, and replacing any non-alphanumeric characters 
    with a single space.

    Parameters
    ----------
    s : str
        The input string to normalize. Handles None/NaN.
    
    Returns
    -------
    str or None
        The normalized string, or None if the input was null or empty.
    """
    if pd.isnull(s) or s == '':
        return None
    # Separate accents, lowercase, removes spaces
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    # Remove accents, keeping only the base characters
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Replace non-alphanumeric (e.g. punctuation) characters with a single space
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def get_best_match(query, choices, min_score=0.6):
    """
    Finds the best matching strings in a list of choices based on the 
    SequenceMatcher similarity ratio (Fuzzy Matching).

    Note: Short queries (<= 3 chars) ignore the minimum score threshold
    to prevent discarding relevant short codes.

    Parameters
    ----------
    query : str
        The string to search for.
    choices : list
        A list of candidate strings to match against.
    min_score : float, optional
        The minimum similarity ratio (0 to 1) required for a match. 
        Defaults to 0.6.
    
    Returns
    -------
    list
        A list of the best matching strings from `choices`.
    """
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


def impute_model_by_engine_size(df_in, reference_df=None):
    """
    Impute model for 'needs_review' rows using KNN on engineSize.
    Uses reference_df for finding neighbors (to prevent data leakage).

    Parameters
    ----------
    df_in : DataFrame
        The DataFrame containing the data to impute.
    reference_df : DataFrame, optional
        The DataFrame used as the source of clean 'model' values (should be 
        the cleaned Training set). Defaults to df_in if not provided.
    
    Returns
    -------
    DataFrame
        The DataFrame with the 'model' column updated for the 
        rows where imputation was successful, and status updated 
        to 'model_imputed_by_esize'.
    """
    df = df_in.copy()
    
    # Use self as reference if not provided
    if reference_df is None:
        reference_df = df
    
    mask_needs_review = df['bm_status'] == 'needs_review'
    df_needs_review = df[mask_needs_review]
    df_reference = reference_df[reference_df['bm_status'] != 'needs_review'].copy()
    
    corrected_models = {}

    for car_id, row in df_needs_review.iterrows():
        current_brand = row['Brand']
        current_model = row['model']
        current_engine_size = row['engineSize']
        
        if pd.isna(current_brand) or pd.isna(current_engine_size):
            continue

        # Filter references by same Brand
        ref_by_brand = df_reference[df_reference['Brand'] == current_brand].copy()

        # For short models (≤2 chars), match by prefix
        if len(str(current_model)) <= 2:
            ref_by_brand = ref_by_brand[
                ref_by_brand['model'].str.startswith(str(current_model), na=False)
            ]
        
        ref_by_brand = ref_by_brand.dropna(subset=['engineSize'])
        if ref_by_brand.empty:
            continue

        # Find K nearest neighbors by engineSize
        distance = np.abs(ref_by_brand['engineSize'] - current_engine_size)
        nearest_neighbors = distance.nsmallest(5).index
        
        # Use mode of neighbors' models
        inferred_model = ref_by_brand.loc[nearest_neighbors, 'model'].mode().iloc[0]
        corrected_models[car_id] = inferred_model

    # Update DataFrame
    for car_id, new_model in corrected_models.items():
        df.loc[car_id, 'model'] = new_model
        df.loc[car_id, 'bm_status'] = 'model_imputed_by_esize'
        df.loc[car_id, 'bm_note'] = f"Model imputed to '{new_model}' using engineSize/prefix similarity."
        
    return df


def force_missing_model(df_in):
    """
    Set model to None for remaining unresolved cases.
    
    Parameters
    ----------
    df_in : DataFrame
        The DataFrame containing the data to process.
    
    Returns
    -------
    DataFrame
        The DataFrame with 'model' set to None for rows where 
        'bm_status' is 'needs_review', and status updated to 
        'forced_missing'.
    """
    df = df_in.copy()
    mask = df['bm_status'] == 'needs_review'
    
    df.loc[mask, 'model'] = None
    df.loc[mask, 'bm_status'] = 'forced_missing'
    df.loc[mask, 'bm_note'] = "model set to missing due to unresolved review status."
    
    return df
#####################
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
    """
    Prints the percentage of negative, positive, zero, and NaN values for a
    specified column in a DataFrame. If a target column is provided, it also
    computes and prints summary statistics (median, mean, std) of the target
    for two groups based on the specified column's values.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    col : str
        The column name to analyze.
    target : str, optional
        The target column name for which to compute summary statistics.
    label : str, optional
        A label to prefix the print statements.
    decimals : int, optional
        Number of decimal places to display in the output.
    
    Returns
    -------
    None
    """
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


def correct_categorical_value(input_value, valid_values, min_score=0.6, fallback='unknown'):
    """
    Correct categorical value using fuzzy matching.

    Parameters
    ----------
    input_value : str
        The input categorical value to correct.
    valid_values : list of str
        The list of valid categorical values to match against.
    min_score : float, optional
        The minimum similarity score required to consider a match. Default is 0.6.
    fallback : str, optional
        The value to return if no suitable match is found. Default is 'unknown'.

    Returns
    -------
    str
        The corrected categorical value, or the fallback if no match is found.
    """
    # Start by normalizing the input to have a clean base for comparison
    normalized = norm(input_value)
    # Use get_best_match (similarity-based function) to find the best matches
    matches = get_best_match(normalized, valid_values, min_score=min_score)
    
    # Return the best match if exactly one match is found
    if isinstance(matches, list) and len(matches) == 1:
        return matches[0]
    # Return None if multiple matches are found
    elif isinstance(matches, list) and len(matches) > 1:
        print(f"Tie detected for value '{input_value}'. Multiple equally good matches found: {matches}") #APAGAR
        return None 
    # Return fallback if no matches are found
    return fallback


def test_missingness(df, target, num_cols):
    """
    Tests whether the missingness of a target variable is related to other
    variables in the DataFrame, indicating Missing At Random (MAR) mechanism.
    Uses t-tests for numeric and chi-square for categorical predictors.

    Parameters
    ----------
    df : DataFrame
        The DataFrame containing the data.
    target : str
        The target column name for which to test missingness.
    num_cols : list of str
        List of numeric column names to test against the missingness indicator.
    
    Returns
    -------
    DataFrame
        A DataFrame containing the results of the tests, including p-values.
    """
    df = df.copy()
    # Create missingness indicator (1 = missing, 0 = present)
    miss_col = target + '_missing'
    df[miss_col] = df[target].isna().astype(int)
    
    results = []
    
    # Numeric predictors: t-tests
    for col in num_cols:
        if col != target and col in df.columns:
            group0 = df.loc[df[miss_col] == 0, col].dropna()
            group1 = df.loc[df[miss_col] == 1, col].dropna()
            if len(group0) > 1 and len(group1) > 1:
                _, p = ttest_ind(group0, group1, equal_var=False)
                results.append((target, col, 't-test', p))
    
    # Categorical predictors: chi-square
    cat_cols = df.select_dtypes(exclude='number').columns
    for col in cat_cols:
        if col != miss_col:
            table = pd.crosstab(df[col], df[miss_col])
            if table.shape[0] > 1:
                _, p, _, _ = chi2_contingency(table)
                results.append((target, col, 'chi-square', p))
    return pd.DataFrame(results, columns=['Target', 'Variable', 'Test', 'p_value'])

    
def plot_violin_comparison(X_train, X_val, X_test, cat_col, num_cols, top_n=None):
    """
    Plots violin plots comparing the distribution of numerical columns across
    training, validation, and test datasets, segmented by a categorical column.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training dataset.
    X_val : pd.DataFrame
        Validation dataset.
    X_test : pd.DataFrame
        Test dataset.
    cat_col : str
        Name of the categorical column to segment by.
    num_cols : list of str
        List of numerical columns to plot.
    top_n : int, optional
        Number of top categories to include based on frequency. If None, include all categories.

    """
    # Organize into a dictionary for convenience
    dfs_dict = {'Train': X_train, 'Validation': X_val, 'Test': X_test}
    n_rows = len(num_cols)
    # Create subplots: 3 columns (Train, Val, Test) and rows equal to number of numerical columns
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(3 * 5.5, n_rows * 5),
                             squeeze=False) # Ensures axes is always a 2D array

    if top_n is not None:
        # If top_n is a number, use nlargest
        category_order = X_train[cat_col].value_counts().nlargest(top_n).index
        title_detail = f'Top {top_n} {cat_col.capitalize()}s'
    else:
        # If top_n is None, get all unique categories and sort them
        category_order = sorted(X_train[cat_col].unique())
        title_detail = f'All {cat_col.capitalize()}s'

    # Iterate through each numerical column to create a new row of plots
    for row_idx, num_col in enumerate(num_cols):
        # Iterate through the three dataframes for the columns
        for col_idx, (df_name, df) in enumerate(dfs_dict.items()):
            ax = axes[row_idx, col_idx]
            # Filter the dataframe to only include the top N categories
            df_filtered = df[df[cat_col].isin(category_order)]

            custom_palette = [palette[-3]] * len(category_order)

            # Create the violin plot
            sns.violinplot(
                data=df_filtered,
                x=num_col,
                y=cat_col,
                order=category_order,
                ax=ax,
                palette=custom_palette,
                inner='quartile',
                linewidth=1.2
            )
            
            ax.set_title(f'{num_col} Distribution ({df_name} Set)', fontweight='bold', fontsize=12)
            ax.set_xlabel(num_col.capitalize(), fontsize=10)
            
            if col_idx == 0:
                ax.set_ylabel(cat_col.capitalize(), fontsize=10)
            else:
                ax.set_ylabel('')
                
            sns.despine(ax=ax, top=True, right=True)

    plt.suptitle(f'Distribution Comparison for {title_detail}', 
                 fontweight='bold', fontsize=18, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


def plot_boxplot_by_category(df, num_col, cat_col, top_n=None, sort_by_median=True, ascending=False):
    """
    Plots a boxplot showing the distribution of a numerical variable across different categories.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    num_col : str
        The name of the numerical column to plot.
    cat_col : str
        The name of the categorical column to group by.
    top_n : int, optional
        If specified, only the top N most frequent categories will be plotted.
    sort_by_median : bool, optional
        Whether to sort categories by the median of the numerical column.
    ascending : bool, optional
        Whether the sorting should be in ascending order.

    Returns
    -------
    None
    """
    # Create a copy to avoid modifying the original dataframe
    df_plot = df.copy()

    # Filter for the top N most frequent categories if top_n is specified
    if top_n is not None:
        top_categories = df_plot[cat_col].value_counts().nlargest(top_n).index
        df_plot = df_plot[df_plot[cat_col].isin(top_categories)]
        title_prefix = f'Top {top_n} '
    else:
        title_prefix = ''

    # Determine the order of categories on the plot
    order = None
    if sort_by_median:
        # Order categories by the median for better visualization
        order = df_plot.groupby(cat_col)[num_col].median().sort_values(ascending=ascending).index

    if order is not None:
        num_categories = len(order)
    else:
        num_categories = df_plot[cat_col].nunique()
    custom_palette = [palette[-2]] * num_categories

    # Create the boxplot
    plt.figure(figsize=(12, 7))
    sns.boxplot(
        data=df_plot,
        x=cat_col,
        y=num_col,
        order=order,
        palette=custom_palette,
        width=0.6
    )

    # Set titles and labels
    plt.title(f'Distribution of {num_col.capitalize()} by {title_prefix}{cat_col.capitalize()}',
              fontweight='bold', fontsize=16)
    plt.xlabel(cat_col.capitalize(), fontsize=12)
    plt.ylabel(num_col.capitalize(), fontsize=12)
    
    # Adjustments for better readability
    plt.xticks(rotation=25, ha='right')
    sns.despine(top=True, right=True)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
        

def plot_importance_unified(palette, importance_series, name, is_tree_model=False):
    """Plot top 20 features by importance"""
    imp_coef = importance_series.sort_values(ascending=False).head(20)
    
    color = palette[1] if is_tree_model else palette[0]
    plt.figure(figsize=(9, 6))
    imp_coef.sort_values().plot(kind="barh", color=color) 
    
    plt.title(f"Feature Importance - {name} (Top 20)", fontsize=15)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_grouped_bar_with_percentage(df, cat_col1, cat_col2):
    """
    Plots a grouped bar chart showing the percentage distribution of cat_col2 within each category of cat_col1.
    Each bar is annotated with the absolute count of occurrences for that category combination.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the categorical columns.
    cat_col1 : str
        The name of the first categorical column (X-axis).
    cat_col2 : str
        The name of the second categorical column (hue).
    
    Returns
    -------
    None
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Pre-calculate percentages for the Y-axis and counts for the annotations
    # 1. Calculate percentages
    df_perc = (df.groupby(cat_col1)[cat_col2]
               .value_counts(normalize=True)
               .mul(100)
               .rename('percentage')
               .reset_index())

    # 2. Calculate absolute counts (for annotation)
    df_counts = (df.groupby(cat_col1)[cat_col2]
                 .value_counts(normalize=False)
                 .rename('count')
                 .reset_index())

    # 3. Merge them for plotting
    df_plot = df_perc.merge(df_counts, on=[cat_col1, cat_col2])

    # 4. Create a lookup dictionary for counts: key=(cat_col1, cat_col2), value=count
    count_lookup = df_counts.set_index([cat_col1, cat_col2])['count'].to_dict()

    # Plot the bar chart using the pre-calculated percentages
    ax = sns.barplot(data=df_plot, x=cat_col1, y='percentage', hue=cat_col2, palette='YlOrBr')

    # Map bar colors back to hue names for the count lookup
    try:
        legend_handles = ax.get_legend().get_patches()
        legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
        color_to_hue = {tuple(np.round(h.get_facecolor(), 4)): l for h, l in zip(legend_handles, legend_labels)}
    except AttributeError:
        color_to_hue = {}

    # Iterate over each bar in the plot to add the absolute count annotation
    for p in ax.patches:
        # Get the height of the bar (which is the PERCENTAGE)
        height_perc = p.get_height()

        # Ignore bars with no height (NaN or 0)
        if pd.isna(height_perc) or height_perc == 0:
            continue

        # Get the name of the x-axis group this bar belongs to
        group_name = None
        for xtick in ax.get_xticklabels():
            # Use np.isclose for robust float comparison
            if np.isclose(xtick.get_position()[0], p.get_x() + p.get_width() / 2.):
                 group_name = xtick.get_text()
                 break
        # If we didn't find a matching x-tick, we need to find the closest one
        if group_name is None:
             x_coords = [tick.get_position()[0] for tick in ax.get_xticklabels()]
             closest_tick_index = (pd.Series(x_coords) - (p.get_x() + p.get_width() / 2.)).abs().idxmin()
             group_name = ax.get_xticklabels()[closest_tick_index].get_text()

        # We now need to find the HUE name to look up the count
        patch_color = tuple(np.round(p.get_facecolor(), 4))
        hue_name = color_to_hue.get(patch_color)

        # If the group name and hue name are found, get the count
        if group_name and hue_name and (group_name, hue_name) in count_lookup:
            # Calculate the correct percentage
            count = count_lookup[(group_name, hue_name)]

            # Add the count text label slightly above the bar
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height_perc + (ax.get_ylim()[1] * 0.01), # Position based on percentage height
                f'{count}', # Display the absolute count
                ha='center',
                va='bottom',
                color='black',
                fontsize=9
            )

    # Adjust y-axis limit to make space for the labels
    ax.set_ylim(0, 105)

    # Format Y-axis ticks as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Set titles and labels
    ax.set_title(f'Percentage of {cat_col2.capitalize()} by {cat_col1.capitalize()}', fontweight='bold', fontsize=16)
    ax.set_xlabel(cat_col1.capitalize(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')

    # Improve layout and legend
    plt.xticks(rotation=15, ha='right')
    plt.legend(title=cat_col2.capitalize(), loc='upper right')
    sns.despine(top=True, right=True)
    plt.tight_layout()
    plt.show()


def compare_outlier_detection(df, columns, iqr_factor=1.5, mad_threshold=3.5, return_indices=False):
    """
    Compares outlier detection using IQR and MAD methods for specified numerical columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    columns : list of str or str
        The numerical columns to analyze for outliers.
    iqr_factor : float, optional
        The multiplier for the IQR method to define outlier bounds. Default is 1.5.
    mad_threshold : float, optional
        The threshold for the MAD method to define outlier bounds. Default is 3.5.
    return_indices : bool, optional
        If True, returns the indices of detected outliers for each method. Default is False.
    
    Returns
    -------
    pd.DataFrame
        A summary DataFrame with outlier counts and percentages for each method.
    dict, optional
        If return_indices is True, also returns a dictionary with outlier indices for each method.
    """

    # Just to ensure columns is a list
    if isinstance(columns, str):
        columns = [columns]
    
    results_list = []
    all_indices = {'iqr': set(), 'mad': set(), 'both': set()}

    # Iterate through each column
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in the DataFrame. Skipping.")
            continue
        
        data_series = df[column]

        ### IQR Method Calculations ###
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3-Q1
        iqr_lower_bound = Q1- (iqr_factor * IQR)
        iqr_upper_bound = Q3+ (iqr_factor * IQR)
        iqr_outliers_filter = (data_series < iqr_lower_bound) | (data_series > iqr_upper_bound)
        iqr_outlier_indices = set(df[iqr_outliers_filter].index)
        
        ### MAD Method Calculations ###
        median = data_series.median()
        mad = (data_series - median).abs().median()
        
        # Handle case where MAD is zero
        if mad == 0:
            mad_lower_bound = median
            mad_upper_bound = median
            # If MAD is zero, consider all values as outliers that are not equal to the median
            mad_outliers_filter = (data_series != median)
        else:
            mad_lower_bound = median - (mad_threshold * mad)
            mad_upper_bound = median + (mad_threshold * mad)
            mad_outliers_filter = (data_series < mad_lower_bound) | (data_series > mad_upper_bound)
        
        mad_outlier_indices = set(df[mad_outliers_filter].index)

        # Comparison Calculations
        both_indices = iqr_outlier_indices.intersection(mad_outlier_indices)
        
        column_summary = {
            'Column': column,
            'Outlier Count (IQR)': len(iqr_outlier_indices),
            'Outlier % (IQR)': round(len(iqr_outlier_indices) / len(df) * 100, 2),
            'Outlier Count (MAD)': len(mad_outlier_indices),
            'Outlier % (MAD)': round(len(mad_outlier_indices) / len(df) * 100, 2),
            'Outlier Count (Both)': len(both_indices),
            'Outlier % (Both)': round(len(both_indices) / len(df) * 100, 2),
        }
        results_list.append(column_summary)

        # Store indices if requested
        if return_indices:
            all_indices['iqr'].update(iqr_outlier_indices)
            all_indices['mad'].update(mad_outlier_indices)
                    
    if return_indices:
        all_indices['both'] = all_indices['iqr'].intersection(all_indices['mad'])
        for key in all_indices:
            all_indices[key] = sorted(list(all_indices[key]))

    summary_df = pd.DataFrame(results_list).set_index('Column')
    
    if return_indices:
        return summary_df, all_indices
        
    return summary_df


def fit_keep_models_per_brand(train, brand_col="Brand", model_col="model",
                              min_count=20, min_freq=0.01):
    """
    Fit a table of frequent (brand, model) combinations to keep.
    
    Parameters:
    ----------
    train : DataFrame
        The training DataFrame containing brand and model columns.
    brand_col : str, optional
        The name of the brand column. Default is "Brand".
    model_col : str, optional
        The name of the model column. Default is "model".
    min_count : int, optional
        Minimum count threshold to keep a (brand, model) combination. Default is 20.
    min_freq : float, optional
        Minimum frequency threshold within the brand to keep a (brand, model) combination. Default is 0.01.
    
    Returns:
    -------
    DataFrame with (brand, model) combinations to keep.
    """
    tmp = train[[brand_col, model_col]].copy()

    # Exclude 'unknown' entries
    tmp = tmp[~tmp[brand_col].isin(["unknown"])]
    tmp = tmp[~tmp[model_col].isin(["unknown"])]

    # Count combinations and calculate frequencies within each brand
    g = (tmp.groupby([brand_col, model_col], dropna=False)
           .size().rename("n").reset_index())
    g["brand_total"] = g.groupby(brand_col)["n"].transform("sum")
    g["freq"] = g["n"] / g["brand_total"]

    # Keep only sufficiently frequent combinations
    keep = g.loc[
        (g["n"] >= min_count) & (g["freq"] >= min_freq),
        [brand_col, model_col]
    ].copy()
    keep["keep"] = True

    return keep


def collapse_rare_models(df, keep_table, brand_col="Brand", model_col="model",
                         other_label="other"):
    """
    Collapse rare models into a generic 'other' label based on a keep table.

    Parameters:
    ----------
    df : DataFrame
        The DataFrame containing brand and model columns to process.
    keep_table : DataFrame
        The table of (brand, model) combinations to keep.
    brand_col : str, optional
        The name of the brand column. Default is "Brand".
    model_col : str, optional
        The name of the model column. Default is "model".
    other_label : str, optional
        The label to use for rare models. Default is "other".
    
    Returns:
    -------
    DataFrame with rare models replaced by 'other_label'.    
    """
    # Save original index
    original_index = df.index

    out = df.copy()

    # Merge with keep_table (preserving all rows from df)
    out = out.merge(
        keep_table, how="left",
        on=[brand_col, model_col]
    )

    keep_mask = out["keep"].fillna(False)

    # Only replace rare models, skip if brand or model == 'unknown'
    mask_to_replace = (
        (~keep_mask) &
        (~out[brand_col].isin(["unknown", None, np.nan])) &
        (~out[model_col].isin(["unknown", None, np.nan]))
    )

    out.loc[mask_to_replace, model_col] = other_label

    # Restore original index
    out.index = original_index

    # Drop auxiliary column
    out = out.drop(columns=["keep"], errors="ignore")

    return out

def print_selection_results(importance_series, model_name, threshold=None):
    """Print selected features based on importance threshold"""
    if threshold is None:
        threshold = importance_series.mean()
    
    selected = importance_series[importance_series > threshold].index.tolist()
    
    print(f"\n{'-'*60}")
    print(f"MODEL: {model_name}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Selected: {len(selected)} features")
    print(f"Features: {selected}")
    print(f"{'-'*60}\n")
    
    return selected


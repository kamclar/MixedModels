import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, wls
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import streamlit as st
from tabulate import tabulate
import io
import base64
from io import StringIO

def analyze_standard_anova(data, groups):
    df = pd.DataFrame(data.T, columns=groups * 3)

    normalized_values = []
    for i in range(0, len(groups) * 3, 3):
        avg_first_row = df.iloc[:, i].mean()
        for j in range(3):
            normalized_values.append(df.iloc[:, i + j] / avg_first_row)

    all_normalized_values = []
    group_labels = []
    for i in range(len(groups)):
        for j in range(i, len(normalized_values), 3):
            valid_values = normalized_values[j].dropna()
            all_normalized_values.extend(valid_values)
            group_labels.extend([groups[i]] * len(valid_values))

    anova_df = pd.DataFrame({'value': all_normalized_values, 'group': group_labels})

    model = ols('value ~ C(group)', data=anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    tukey = pairwise_tukeyhsd(endog=anova_df['value'], groups=anova_df['group'], alpha=0.05)
    significant_pairs = tukey.reject

    means = []
    std_devs = []

    for group in groups:
        group_values = anova_df[anova_df['group'] == group]['value']
        means.append(np.mean(group_values))
        std_devs.append(np.std(group_values))

    return anova_df, anova_table, tukey, significant_pairs, means, std_devs, "Standard ANOVA"

def analyze_weighted_anova(data, groups):
    df = pd.DataFrame(data.T, columns=groups * 3)

    normalized_values = []
    for i in range(0, len(groups) * 3, 3):
        avg_first_row = df.iloc[:, i].mean()
        for j in range(3):
            normalized_values.append(df.iloc[:, i + j] / avg_first_row)

    all_normalized_values = []
    group_labels = []
    weights = []
    for i in range(len(groups)):
        for j in range(i, len(normalized_values), 3):
            valid_values = normalized_values[j].dropna()
            row_length = len(valid_values)
            weight = 1 / row_length
            all_normalized_values.extend(valid_values)
            group_labels.extend([groups[i]] * row_length)
            weights.extend([weight] * row_length)

    total_observations = len(all_normalized_values)
    weights = [w * total_observations / sum(weights) for w in weights]

    anova_df = pd.DataFrame({'value': all_normalized_values, 'group': group_labels, 'weights': weights})

    model = wls('value ~ C(group)', data=anova_df, weights=anova_df['weights']).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    tukey = pairwise_tukeyhsd(endog=anova_df['value'], groups=anova_df['group'], alpha=0.05)
    significant_pairs = tukey.reject

    means = []
    std_devs = []

    for group in groups:
        group_values = anova_df[anova_df['group'] == group]['value']
        means.append(np.mean(group_values))
        std_devs.append(np.std(group_values))

    return anova_df, anova_table, tukey, significant_pairs, means, std_devs, "Weighted ANOVA"

def plot_results(groups, anova_df, tukey, significant_pairs, means, std_devs, analysis_type):
    def add_significance(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color='black', fontsize=12)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(groups, means, yerr=std_devs, capsize=10, color='#88c7dc')

    ax.set_title(f'Comparison of Group Means ({analysis_type})', fontsize=15)
    ax.set_ylabel('Mean Values', fontsize=12)

    if np.any(significant_pairs):
        max_val = max(means) + max(std_devs)
        h = max_val * 0.05
        gap = max_val * 0.02
        whisker_gap = max_val * 0.02

        comparisons = np.array(tukey.summary().data[1:])
        significant_comparisons = comparisons[significant_pairs]

        for comp in significant_comparisons:
            if 'siRNA_ctrl' in comp[:2]:
                group1 = groups.index(comp[0])
                group2 = groups.index(comp[1])
                add_significance(ax, group1, group2, max_val + whisker_gap, h, '*')
                whisker_gap += h + gap

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode()

    plt.close(fig)

    return plot_url

def display_table(anova_table, tukey):
    anova_table_html = anova_table.to_html(classes='table table-striped')
    tukey_summary_html = tukey.summary().as_html()
    return anova_table_html, tukey_summary_html

def parse_pasted_data(pasted_data, delimiter):
    # Split the data into lines
    lines = pasted_data.strip().split("\n")
    # Split each line into columns
    data = [line.split(delimiter) for line in lines]
    # Find the maximum number of columns
    max_cols = max(len(row) for row in data)
    # Pad the rows to have the same number of columns
    padded_data = [row + [''] * (max_cols - len(row)) for row in data]
    # Convert to DataFrame
    df = pd.DataFrame(padded_data).replace('', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

st.title('ANOVA Analysis')

delimiter = st.selectbox('Select delimiter', (';', '\t', ','))

input_method = st.radio("Select input method", ('File Upload', 'Copy-Paste'))

if input_method == 'File Upload':
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
else:
    pasted_data = st.text_area("Paste your data here (use selected delimiter)")

if (input_method == 'File Upload' and uploaded_file is not None) or (input_method == 'Copy-Paste' and pasted_data):
    try:
        if input_method == 'File Upload':
            data = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)
        else:
            data = parse_pasted_data(pasted_data, delimiter)

        st.write("Data Preview:", data.head())

        data_values = data.values
        st.text_area('Data (numpy array format):', str(data_values))

        groups_input = st.text_area('Groups (list format):', "['siRNA_ctrl', 'siRNA1_VTN', 'siRNA2_VTN']")

        if st.button('Run Analysis and Plot'):
            groups = eval(groups_input)

            # Check if any row contains NaN values indicating varying number of columns
            contains_nan = data.isna().any(axis=1).any()

            if contains_nan:
                anova_df, anova_table, tukey, significant_pairs, means, std_devs, analysis_type = analyze_weighted_anova(data_values, groups)
            else:
                anova_df, anova_table, tukey, significant_pairs, means, std_devs, analysis_type = analyze_standard_anova(data_values, groups)

            st.write(f"Analysis Type: {analysis_type}")

            anova_table_html, tukey_summary_html = display_table(anova_table, tukey)
            plot_url = plot_results(groups, anova_df, tukey, significant_pairs, means, std_devs, analysis_type)

            st.markdown(anova_table_html, unsafe_allow_html=True)
            st.markdown(tukey_summary_html, unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{plot_url}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")

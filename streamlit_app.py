import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tabulate import tabulate
import io
import base64
from io import StringIO
from pymer4.models import Lmer
from scipy.stats import kruskal
import scikit_posthocs as sp


def analyze_mixed_effects(data, groups):
    # Reshape data into long format for mixed-effects modeling
    group_labels = []
    biorep_labels = []
    techrep_labels = []
    values = []

    biorep_count = len(data) // (len(groups) * 3)

    for i, group in enumerate(groups):
        for biorep in range(biorep_count):
            for techrep in range(3):
                idx = biorep * (len(groups) * 3) + i * 3 + techrep
                values.append(data[idx, techrep])
                group_labels.append(group)
                biorep_labels.append(f'biorep_{biorep + 1}')
                techrep_labels.append(f'techrep_{techrep + 1}')
    
    anova_df = pd.DataFrame({
        'value': values,
        'group': group_labels,
        'biorep': biorep_labels,
        'techrep': techrep_labels
    })

    # Fit a mixed-effects model using pymer4
    model = Lmer("value ~ group + (1|biorep)", data=anova_df)
    result = model.fit()

    # Perform post-hoc analysis using estimated marginal means (EMMs)
    emmeans = model.emmeans('group')
    comparisons = emmeans.compare(method='holm')
    
    return anova_df, result, comparisons

def plot_results(groups, anova_df, comparisons):
    means = anova_df.groupby('group')['value'].mean()
    std_devs = anova_df.groupby('group')['value'].std()

    def add_significance(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color='black', fontsize=12)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(groups, means, yerr=std_devs, capsize=10, color='#88c7dc')

    ax.set_title('Comparison of Group Means (Mixed Effects)', fontsize=15)
    ax.set_ylabel('Mean Values', fontsize=12)

    max_val = max(means) + max(std_devs)
    h = max_val * 0.05
    gap = max_val * 0.02
    whisker_gap = max_val * 0.02

    for i, row in comparisons.iterrows():
        if row['p-value'] < 0.05:
            group1_idx = groups.index(row['group1'])
            group2_idx = groups.index(row['group2'])
            add_significance(ax, group1_idx, group2_idx, max_val + whisker_gap, h, '*')
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

def display_table(summary, comparisons):
    summary_html = summary.to_html(classes='table table-striped')
    comparisons_html = comparisons.to_html(classes='table table-striped')
    return summary_html, comparisons_html

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

st.title('Mixed Model Analysis')

delimiter = st.selectbox('Select delimiter', ( '\t', ',', ';'))
input_method = st.radio("Select input method", ( 'Copy-Paste', 'File Upload'))

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
        groups_input = st.text_area('Groups (list format):', "['siRNA_ctrl', 'siRNA1_VTN', 'siRNA2_VTN']")

        if st.button('Run Analysis and Plot'):
            groups = eval(groups_input)
            anova_df, summary, comparisons = analyze_mixed_effects(data_values, groups)

            st.write("Analysis Type: Mixed Effects Model")
            summary_html, comparisons_html = display_table(summary, comparisons)
            plot_url = plot_results(groups, anova_df, comparisons)

            st.markdown(summary_html, unsafe_allow_html=True)
            st.markdown(comparisons_html, unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{plot_url}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
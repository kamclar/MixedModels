import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.formula.api import mixedlm
import streamlit as st
from tabulate import tabulate
import io
import base64
from io import StringIO
import scikit_posthocs as sp

def analyze_mixed_model_anova(data, groups):
    num_technical_reps = data.shape[1] // len(groups)
    biological_replicates = np.repeat(range(1, num_technical_reps + 1), len(groups))
    technical_replicates = np.tile(range(1, len(groups) + 1), num_technical_reps)

    df = pd.DataFrame(data.T.flatten(), columns=['value'])
    df['group'] = np.tile(groups, num_technical_reps)
    df['biological_replicate'] = biological_replicates
    df['technical_replicate'] = technical_replicates

    model = mixedlm("value ~ group", df, groups=df["biological_replicate"])
    result = model.fit()

    means = df.groupby('group')['value'].mean().tolist()
    std_devs = df.groupby('group')['value'].std().tolist()

    return df, result.summary(), means, std_devs, "Mixed Model ANOVA"

def dunnett_test(anova_df, control_group):
    comp = sp.posthoc_dunn(anova_df, val_col='value', group_col='group', p_adjust='bonferroni')
    control_comp = comp.loc[control_group]
    return control_comp

def plot_results(groups, anova_df, dunnett_results, means, std_devs, analysis_type):
    def add_significance(ax, x1, x2, y, h, text):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')
        ax.text((x1 + x2) * .5, y + h, text, ha='center', va='bottom', color='black', fontsize=12)

    # Bar plot
    fig, ax = plt.subplots(figsize=(8, 8))
    bars = ax.bar(groups, means, yerr=std_devs, capsize=10, color='#88c7dc')

    ax.set_title(f'Comparison of Group Means ({analysis_type})', fontsize=15)
    ax.set_ylabel('Mean Values', fontsize=12)

    control_group = groups[0]
    other_groups = groups[1:]

    if not dunnett_results.empty:
        max_val = max(means) + max(std_devs)
        h = max_val * 0.05
        gap = max_val * 0.02
        whisker_gap = max_val * 0.02

        for group in other_groups:
            p_value = dunnett_results[group]
            if p_value < 0.05:  # If p-value is significant
                group1 = groups.index(control_group)
                group2 = groups.index(group)
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

def display_table(anova_table, dunnett_results):
    anova_table_html = anova_table.as_html()
    dunnett_html = dunnett_results.to_frame().to_html(classes='table table-striped')
    return anova_table_html, dunnett_html

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

st.title('Mixel Model Analysis')

delimiter = st.selectbox('Select delimiter', ( '\t', ',', ';'))

input_method = st.radio("Select input method", ('Copy-Paste', 'File Upload'))

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

            anova_df, anova_table, means, std_devs, analysis_type = analyze_mixed_model_anova(data_values, groups)
            dunnett_results = dunnett_test(anova_df, groups[0])

            st.write(f"Analysis Type: {analysis_type}")

            anova_table_html, dunnett_html = display_table(anova_table, dunnett_results)
            plot_url = plot_results(groups, anova_df, dunnett_results, means, std_devs, analysis_type)

            st.markdown(anova_table_html, unsafe_allow_html=True)
            st.markdown(dunnett_html, unsafe_allow_html=True)
            st.image(f"data:image/png;base64,{plot_url}")
    except Exception as e:
        st.error(f"Error processing the file: {e}")

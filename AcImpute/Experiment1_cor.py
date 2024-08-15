#Experiments to calculate correlations
#sc_10x_5cl dataset
import pandas as pd
from scipy.stats import spearmanr, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

bulk_expr = pd.read_csv('./datasets/cellbench/cellbench/GSE86337_processed_count_average_replicates.csv',
                        index_col=0)

AcImpute_expr = pd.read_csv('./datasets/cellbench/sc_10x_5cl/methods/AcImpute/sc_10x_5cl.csv', index_col=0)
raw_expr = pd.read_csv('./datasets/cellbench/sc_10x_5cl/genebycell.csv', index_col=0)

cell_types = ["A549", "H1975", "H2228", "H838","HCC827"]

def calculate_correlations(single_cell_df, bulk_df, method_name):
    results = []

    common_genes = bulk_df.index.intersection(single_cell_df.index)
    print(f"Number of common genes: {len(common_genes)}")

    for cell_type in cell_types:
        single_cell_cols = [col for col in single_cell_df.columns if cell_type in col]
        bulk_data = bulk_df[cell_type].loc[common_genes]

        for sc_col in single_cell_cols:
            single_cell_data = single_cell_df[sc_col].loc[common_genes]

            corr, _ = spearmanr(bulk_data, single_cell_data)
            results.append({
                'Cell_Type': cell_type,
                'Single_Cell_Column': sc_col,
                'Spearman_Correlation': corr,
                'Method': method_name
            })

    result_df = pd.DataFrame(results)
    return result_df

AcImpute_corr = calculate_correlations(AcImpute_expr, bulk_expr, method_name='AcImpute')
raw_corr = calculate_correlations(raw_expr, bulk_expr, method_name='raw')

correlation_results = pd.concat([AcImpute_corr, raw_corr], ignore_index=True)

print(correlation_results.head())
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Cell_Type', y='Spearman_Correlation', hue='Method', data=correlation_results, inner="box",
               palette="muted")
for i, cell_type in enumerate(cell_types):
    data_AcImpute = correlation_results[
        (correlation_results['Cell_Type'] == cell_type) & (correlation_results['Method'] == 'AcImpute')][
        'Spearman_Correlation']
    data_raw = \
    correlation_results[(correlation_results['Cell_Type'] == cell_type) & (correlation_results['Method'] == 'raw')][
        'Spearman_Correlation']

    t_stat, p_value = ttest_ind(data_AcImpute, data_raw)

    if p_value < 0.0001:
        sig = '****'
    elif p_value < 0.001:
        sig = '***'
    elif p_value < 0.01:
        sig = '**'
    elif p_value < 0.05:
        sig = '*'
    else:
        sig = 'ns'  # 不显著

    y, h, col = max(data_AcImpute.max(), data_raw.max()) + 0.02, 0.02, 'k'
    ax.text(i, y, sig, ha='center', va='bottom', color=col, fontsize=12)

ax.legend(title='Method', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.ylabel("Correlation")
plt.xlabel("Cell Type")
plt.xticks(rotation=45)
plt.grid(True)

plt.savefig("./datasets/cellbench/sc_10x_5cl/cor/t0810spearman_correlation_comparison_violinplot.pdf")
plt.show()


########################################################
##Ziegenhain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def corercc(ercc, methodss):
    corpearson = []
    for i in range(methodss.shape[1]):
        cor, _ = pearsonr(ercc, methodss[:, i])
        corpearson.append(cor)
    return corpearson


datasets = ["SmartSeq2", "CELseq2", "MARSseq", "SCRBseq", "SmartSeq"]
dataercclong = pd.DataFrame()

def corercc(ercc, methodss):
    corpearson = []
    for i in range(methodss.shape[1]):
        cor, _ = pearsonr(ercc, methodss[:, i])
        corpearson.append(cor)
    return corpearson


datasets = ["SmartSeq2", "CELseq2", "MARSseq", "SCRBseq", "SmartSeq"]
dataercclong = pd.DataFrame()

for dataset in datasets:
    # read data
    AcImpute = pd.read_csv(f'./code/data/{dataset}_AcImpute.csv')
    rawdata = pd.read_csv(f'./datasets/Ziegenhain/{dataset}.csv')

    erccrawloc = rawdata.iloc[:, 0].str.contains('ERCC')
    erccraw = rawdata[erccrawloc]

    ercc = pd.read_excel("./datasets/Ziegenhain/ERCC.xlsx", usecols=[1, 3])
    ercc = ercc.values

    erccAcImpute = AcImpute[AcImpute.iloc[:, 0].str.contains('ERCC')]
    common_rows = np.intersect1d(erccraw.iloc[:, 0].values, erccAcImpute.iloc[:, 0].values)

    erccAcImpute = erccAcImpute[erccAcImpute.iloc[:, 0].isin(common_rows)].sort_values(by=erccAcImpute.columns[0])
    erccraw = erccraw[erccraw.iloc[:, 0].isin(common_rows)].sort_values(by=erccraw.columns[0])
    ercc = ercc[np.isin(ercc[:, 0], [row.replace("g", "") for row in common_rows])]
    ercc = ercc[np.argsort(ercc[:, 0])]

    erccAcImpute_values = erccAcImpute.iloc[:, 1:].apply(pd.to_numeric).values
    erccraw_values = erccraw.iloc[:, 1:].values
    ercc_values = ercc[:, 1].astype(float)

    erccrawcor = corercc(np.log(ercc_values), np.log(erccraw_values + 1))
    erccAcImputecor = corercc(np.log(ercc_values), np.log(erccAcImpute_values + 1))

    dataercc = pd.DataFrame({
        "raw": erccrawcor,
        "AcImpute": erccAcImputecor
    }).melt(var_name="Ziegenhain", value_name="correlation")

    dataercc["methods"] = dataset
    dataercclong = pd.concat([dataercclong, dataercc], ignore_index=True)

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

ax = sns.boxplot(x="methods", y="correlation", hue="Ziegenhain", data=dataercclong, palette="Paired")
plt.legend(title="Methods", loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
plt.xticks(rotation=45)
plt.show()

plt.savefig("./datasets/Ziegenhain/result/ercclonggroup.pdf", bbox_inches='tight')
dataercclong.to_csv("./datasets/Ziegenhain/result/dataercclong.csv", index=False)

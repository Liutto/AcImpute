#Used to calculate the inverse relationship between gene expression and dropout rate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

data_dir = "E:/学习箱/论文/2/datasets/Usoskin_silver/"

rawdata = pd.read_csv(f"{data_dir}Usoskin_RAW.csv")
labels = pd.read_table(f"{data_dir}Usoskin_RAW.txt", header=None).iloc[0, :].astype(int)

datarate = []

for i in np.unique(labels):
    type_col = np.where(labels == i)[0]

    gene_exp = rawdata.iloc[:, 1:].values[:, type_col]

    row_mean = np.mean(gene_exp, axis=1)
    zero_proportion = np.sum(gene_exp == 0, axis=1) / gene_exp.shape[1]

    sorted_indices = np.argsort(row_mean)
    row_means_sorted = row_mean[sorted_indices]
    zero_proportion_sorted = zero_proportion[sorted_indices]

    group_type = np.full(len(row_mean), i)
    merged_data = np.column_stack((group_type, row_means_sorted, zero_proportion_sorted))

    datarate.append(merged_data)

datarate = np.vstack(datarate)
datarate_df = pd.DataFrame(datarate, columns=["group_type", "row_means_sorted", "zero_proportion_sorted"])

result = datarate_df.groupby(["group_type", "row_means_sorted"]).agg(
    Mean_zero_proportion=("zero_proportion_sorted", "mean"),
    countn=("zero_proportion_sorted", "size")
).reset_index()


result["group_type"] = result["group_type"].astype(int)

correlations = []
for group in result["group_type"].unique():
    group_data = result[result["group_type"] == group]
    corr, _ = pearsonr(group_data["row_means_sorted"], group_data["Mean_zero_proportion"])
    correlations.append(corr)

average_correlation = np.mean(correlations)

print("Each type of correlation:")
for idx, corr in enumerate(correlations):
    print(f"The correlation of type {result['group_type'].unique()[idx]} : {corr}")

print(f"The average of all types of correlations: {average_correlation}")
result.to_csv("./output_plot_Usoskin_zero_proportion.csv", index=False)

#
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))
# sns.lineplot(
#     data=result,
#     x="row_means_sorted",
#     y="Mean_zero_proportion",
#     hue="group_type",
#     marker="o",
#     palette="Paired"
# )
# plt.xlabel("Mean Expression Level (row_means_sorted)", fontsize=14)
# plt.ylabel("Mean Zero Proportion", fontsize=14)
# plt.legend(title="Group Type")
# plt.show()

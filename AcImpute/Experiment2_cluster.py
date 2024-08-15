import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering


def matrix_fusion(true_lab, pre_lab):
    true_labels = pd.Categorical(true_lab)
    pred_labels = pd.Categorical(pre_lab)
    contingency_matrix = pd.crosstab(true_labels, pred_labels)
    return contingency_matrix.to_numpy()


def calculate_ari(true_lab, pre_lab):
    if len(true_lab) != len(pre_lab):
        raise ValueError("The length of true_lab and pre_lab must be equal.")

    A = matrix_fusion(true_lab, pre_lab)
    a = A.sum(axis=1)
    b = A.sum(axis=0)
    n = len(true_lab)

    c1 = np.sum(a * (a - 1)) / 2
    c2 = np.sum(b * (b - 1)) / 2
    c0 = np.sum(A * (A - 1)) / 2

    ari = (n * (n - 1) * c0 / 2 - c1 * c2) / (n * (n - 1) * (c1 + c2) / 4 - c1 * c2)
    return ari


def calculate_purity(true_lab, pre_lab):
    if len(true_lab) != len(pre_lab):
        raise ValueError("The length of true_lab and pre_lab must be equal.")

    N = len(true_lab)
    unique_true = np.unique(true_lab)
    unique_pred = np.unique(pre_lab)
    sumnum = 0

    for v in unique_true:
        maxnum = 0
        for u in unique_pred:
            num = len(np.intersect1d(np.where(pre_lab == u)[0], np.where(true_lab == v)[0]))
            if num > maxnum:
                maxnum = num
        sumnum += maxnum

    return sumnum / N


def calculate_nmi(true_lab, pre_lab):
    if len(true_lab) != len(pre_lab):
        raise ValueError("The length of true_lab and pre_lab must be equal.")

    N = len(true_lab)
    v = np.unique(true_lab)
    u = np.sort(np.unique(pre_lab))

    jointprobability = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            jointprobability[i, j] = len(
                np.intersect1d(np.where(pre_lab == u[i])[0], np.where(true_lab == v[j])[0])) / N

    eps = 1.4e-100
    A = jointprobability.sum(axis=1)
    B = jointprobability.sum(axis=0)
    Hx = -np.sum(A * np.log2(A + eps))
    Hy = -np.sum(B * np.log2(B + eps))
    C = -np.sum(jointprobability * np.log2(jointprobability + eps))

    nmi = 2 * (Hx + Hy - C) / (Hx + Hy)
    return nmi


data_dir = "E:/学习箱/论文/2/datasets/Chu/"
rawdata = pd.read_csv(f"{data_dir}Chu_celltype_raw.csv")

scImpute = pd.read_csv(f"{data_dir}scimpute/scimpute_count.csv")
saver = pd.read_csv(f"{data_dir}saver.csv")
magic = pd.read_csv(f"{data_dir}magic1.csv")
scRMD = pd.read_csv(f"{data_dir}scRMD.csv")
ALRA = pd.read_csv(f"{data_dir}ALRA.csv")
VIPER = pd.read_csv(f"{data_dir}VIPER.csv")
DrImpute = pd.read_csv(f"{data_dir}DrImpute.csv")
scRecover = pd.read_csv(f"{data_dir}outDir_scRecover/scRecover+scImpute.csv")
deepimpute = pd.read_csv(f"{data_dir}deepimpute.csv")
AcImpute = pd.read_csv(f"{data_dir}Acmagic_np_100.csv")

label = rawdata.columns[1:].str.extract(r'^(.*?)_')[0]
label_map = {label: idx for idx, label in enumerate(np.unique(label))}
label = label.map(label_map).astype(int)

methods = {
    "rawdata": rawdata,
    "magic": magic,
    "saver": saver,
    "scImpute": scImpute,
    "VIPER": VIPER,
    "scRMD": scRMD,
    "scRecover": scRecover,
    "deepimpute": deepimpute,
    "DrImpute": DrImpute,
    "AcImpute": AcImpute,
    "ALRA": ALRA
}

results = {"ARI": [], "NMI": [], "Purity": []}

for method_name, method_data in methods.items():
    print(f"Processing {method_name}...")
    method_data = method_data.iloc[:, 1:]
    totalCounts_by_cell = method_data.sum(axis=0)
    normalized_data = method_data.div(totalCounts_by_cell, axis=1) * 1e6
    log_data = np.log10(normalized_data + 1.01)

    # PCA降维
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(log_data.T)

    # Spectral Clustering
    specc_res = SpectralClustering(n_clusters=7, affinity='rbf', random_state=666).fit(pca_data)
    pred_labels = specc_res.labels_

    ari = calculate_ari(label, pred_labels)
    nmi = calculate_nmi(label, pred_labels)
    purity = calculate_purity(label, pred_labels)

    results["ARI"].append(ari)
    results["NMI"].append(nmi)
    results["Purity"].append(purity)

databar = pd.DataFrame({
    "variable": ["ARI"] * len(methods) + ["NMI"] * len(methods) + ["Purity"] * len(methods),
    "methods": list(methods.keys()) * 3,
    "value": results["ARI"] + results["NMI"] + results["Purity"]
})

databar_pivot = databar.pivot_table(index="methods", columns="variable", values="value")

databar_pivot['total'] = databar_pivot.sum(axis=1)
databar_pivot = databar_pivot.sort_values(by='total', ascending=False)
databar_pivot = databar_pivot.drop(columns=['total'])

databar_pivot.plot(kind="bar", stacked=True, color=sns.color_palette("Paired", len(databar["variable"].unique())), edgecolor="black", width=0.8)

plt.title('Chu_celltype5 1018cells × 17559 genes', fontsize=13)
plt.xlabel('Methods', fontsize=12)
plt.ylabel('Value', fontsize=12)

plt.xticks(rotation=45, ha="right", fontsize=10, color="black")
plt.yticks(fontsize=14, color="black")

plt.tight_layout()
plt.legend(title='Metrics', title_fontsize='13', fontsize='10')
plt.show()
databar.to_csv("E:/学习箱/论文/2/result/0814Usoskin_AcImpute_npall_databar.csv", index=False)

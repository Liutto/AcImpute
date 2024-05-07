
import array

import utils
from scipy import sparse
from scipy import spatial
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

import graphtools
print("AcImpute")
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
import scprep
import tasklogger
import warnings
import heapq
import math
import time
_logger = tasklogger.get_tasklogger("graphtools")

class AcImpute(BaseEstimator):

    global data_imputed_pre
    data_imputed_pre = None
    global high_var_genes
    high_var_genes = None
    global normalize_data
    normalize_data = None
    def __init__(
        self,
        knn=5,
        knn_max=None,
        decay=1,
        t="auto",
        n_pca=100,
        solver="exact",
        knn_dist="euclidean",
        n_jobs=1,
        random_state=None,
        verbose=1,
    ):
        self.knn = knn
        self.knn_max = knn_max
        self.decay = decay
        self.t = t
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.solver = solver
        self.graph = None
        self.X = None
        self.X_AcImpute = None
        self._check_params()
        self.verbose = verbose

        _logger.set_level(verbose)

    @property
    def knn_max(self):
        if self._knn_max is not None:
            return self._knn_max
        else:
            return self.knn * 3
    @knn_max.setter
    def knn_max(self, value):
        self._knn_max = value

    @property
    def diff_op(self):
        """The diffusion operator calculated from the data"""
        if self.graph is not None:
            return self.graph.diff_op
        else:
            raise NotFittedError(
                "This AcImpute instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )

    def _check_params(self):

        utils.check_positive(knn=self.knn)
        utils.check_int(knn=self.knn, n_jobs=self.n_jobs)
        utils.check_if_not(
            None,
            utils.check_positive,
            utils.check_int,
            n_pca=self.n_pca,
            knn_max=self.knn_max,
        )
        utils.check_if_not(None, utils.check_positive, decay=self.decay)
        utils.check_if_not("auto", utils.check_positive, utils.check_int, t=self.t)
        utils.check_in(["exact", "approximate"], solver=self.solver)
        if not callable(self.knn_dist):
            utils.check_in(
                [
                    "euclidean",
                    "cosine",
                    "correlation",
                    "cityblock",
                    "l1",
                    "l2",
                    "manhattan",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "dice",
                    "hamming",
                    "jaccard",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "yule",
                ],
                knn_dist=self.knn_dist,
            )

    def _set_graph_params(self, **params):
        try:
            self.graph.set_params(**params)
        except AttributeError:
            pass

    def set_params(self, **params):

        reset_kernel = False
        reset_imputation = False
        # diff potential parameters
        if "t" in params and params["t"] != self.t:
            self.t = params["t"]
            reset_imputation = True
            del params["t"]

        # kernel parameters
        if "knn" in params and params["knn"] != self.knn:
            self.knn = params["knn"]
            reset_kernel = True
            del params["knn"]
        if "knn_max" in params and params["knn_max"] != self.knn_max:
            self.knn_max = params["knn_max"]
            reset_kernel = True
            del params["knn_max"]
        if "decay" in params and params["decay"] != self.decay:
            self.decay = params["decay"]
            reset_kernel = True
            del params["decay"]
        if "n_pca" in params and params["n_pca"] != self.n_pca:
            self.n_pca = params["n_pca"]
            reset_kernel = True
            del params["n_pca"]
        if "knn_dist" in params and params["knn_dist"] != self.knn_dist:
            self.knn_dist = params["knn_dist"]
            reset_kernel = True
            del params["knn_dist"]

        # parameters that don't change the embedding
        if "solver" in params and params["solver"] != self.solver:
            self.solver = params["solver"]
            reset_imputation = True
            del params["solver"]
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
            self._set_graph_params(n_jobs=params["n_jobs"])
            del params["n_jobs"]
        if "random_state" in params:
            self.random_state = params["random_state"]
            self._set_graph_params(random_state=params["random_state"])
            del params["random_state"]
        if "verbose" in params:
            self.verbose = params["verbose"]
            tasklogger.set_level(self.verbose)
            self._set_graph_params(verbose=params["verbose"])
            del params["verbose"]

        if reset_kernel:
            # can't reset the graph kernel without making a new graph
            self.graph = None
            reset_imputation = True
        if reset_imputation:
            self.X_AcImpute = None

        self._check_params()
        return self

    def fit(self, X, graph=None):

        if self.n_pca is None or X.shape[1] <= self.n_pca:
            n_pca = None
        else:
            n_pca = self.n_pca
        # num
        if X.shape[0]<1000:
            celln = ((X.shape[0] - 30) / (1000 - 30)) * (3 - 1) + 1
            self._knn_max = round(self.knn * celln)




        if graph is None:
            graph = self.graph
            if self.X is not None and not utils.matrix_is_equivalent(X, self.X):
                """
                If the same data is used, we can reuse existing kernel and
                diffusion matrices. Otherwise we have to recompute.
                """
                _logger.debug("Reset graph due to difference in input data")
                graph = None
            elif graph is not None:
                try:
                    graph.set_params(
                        decay=self.decay,
                        knn=self.knn,
                        knn_max=self.knn_max,
                        distance=self.knn_dist,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose,
                        n_pca=n_pca,
                        thresh=1e-4,
                        random_state=self.random_state,
                    )
                except ValueError as e:
                    # something changed that should have invalidated the graph
                    _logger.debug("Reset graph due to {}".format(str(e)))
                    graph = None
        else:
            self.knn = graph.knn
            self.alpha = graph.decay
            self.n_pca = graph.n_pca
            self.knn_dist = graph.distance
            try:
                self.knn_max = graph.knn_max
            except AttributeError:
                # not all graphs have knn_max
                self.knn_max = None

        self.X = X

        if utils.has_empty_columns(X):
            warnings.warn(
                "Input matrix contains unexpressed genes. "
                "Please remove them prior to running AcImpute."
            )

        if graph is not None:
            _logger.info("Using precomputed graph and diffusion operator...")
            self.graph = graph
        else:
            # reset X_AcImpute in case it was previously set
            self.X_AcImpute = None

            with _logger.task("graph and diffusion operator"):


                global libsize
                # 先归一化再筛选高变异基因
                libsize = np.sum(X, axis=1)  #消除细胞大小的影响，对每行求和
                datat = np.array(X)
                libsize = np.array(libsize)
                datat = datat / libsize[:, np.newaxis] * np.median(libsize)
                global normalize_data
                normalize_data = datat
                means = np.mean(datat, axis=0)
                stds = np.std(datat, axis=0)
                cv = stds / means
                global high_var_genes
                high_var_genes = np.where((means >= 0.01) & (cv >= np.quantile(cv, 0.25)))[0]
                datat = datat[:, high_var_genes]
                self.graph = graphtools.Graph(
                    datat,
                    n_pca=n_pca,
                    knn=self.knn,
                    knn_max=self.knn_max,
                    decay=self.decay,
                    thresh=1e-4,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                )

        return self

    def _parse_genes(self, X, genes):
        X_sparse = sparse.issparse(X) or scprep.utils.is_sparse_dataframe(X)
        X_large = np.prod(X.shape) > 5000 * 20000
        if genes is None and X_sparse and X_large:
            warnings.warn(
                "Returning imputed values for all genes on a ({} x "
                "{}) matrix will require approximately {:.2f}GB of "
                "memory. Suppress this warning with "
                "`genes='all_genes'`".format(
                    X.shape[0], X.shape[1], np.prod(X.shape) * 8 / (1024 ** 3)
                ),
                UserWarning,
            )
        if isinstance(genes, str) and genes == "all_genes":
            genes = None
        elif isinstance(genes, str) and genes == "pca_only":
            if not hasattr(self.graph, "data_pca"):
                raise RuntimeError("Cannot return PCA as PCA is not performed.")
        elif genes is not None:
            genes = np.array([genes]).flatten()
            if not issubclass(genes.dtype.type, numbers.Integral):
                # gene names
                if isinstance(X, pd.DataFrame):
                    gene_names = X.columns
                elif utils.is_anndata(X):
                    gene_names = X.var_names
                else:
                    raise ValueError(
                        "Non-integer gene names only valid with pd.DataFrame "
                        "or anndata.AnnData input. "
                        "X is a {}, genes = {}".format(type(X).__name__, genes)
                    )
                if not np.all(np.isin(genes, gene_names)):
                    warnings.warn(
                        "genes {} missing from input data".format(
                            genes[~np.isin(genes, gene_names)]
                        )
                    )
                genes = np.argwhere(np.isin(gene_names, genes)).reshape(-1)
        return genes

    def transform(self, X=None, genes=None, t_max=20, plot_optimal_t=False, ax=None):
        """Computes the values of genes after diffusion

        """
        if self.graph is None:
            if self.X is not None:
                self.fit(self.X)
            else:
                raise NotFittedError(
                    "This AcImpute instance is not fitted yet. Call "
                    "'fit' with appropriate arguments before "
                    "using this method."
                )

        if X is not None and not utils.matrix_is_equivalent(X, self.graph.data):
            extrapolation = True
            store_result = False
            warnings.warn(
                "Running AcImpute.transform on different "
                "data to that which was used for AcImpute.fit may not "
                "produce sensible output, unless it comes from the "
                "same manifold.",
                UserWarning,
            )
        else:
            extrapolation = False
            X = self.X
            data = self.graph
            store_result = True


        genes = self._parse_genes(X, genes)
        if genes is None:
            genes_is_short = False
        else:
            genes_is_short = len(genes) < self.graph.data_nu.shape[1]

        if isinstance(genes, str) and genes == "pca_only":
            # have to use PCA to return it
            solver = "approximate"
        elif self.X_AcImpute is None and genes_is_short:
            # faster to skip PCA
            solver = "exact"
            store_result = False
        else:
            solver = self.solver

        if store_result and self.X_AcImpute is not None:
            X_AcImpute = self.X_AcImpute
        else:
            if extrapolation:
                n_pca = self.n_pca if solver == "approximate" else None
                data = graphtools.base.Data(X, n_pca=n_pca)
            if solver == "approximate":
                # select PCs
                X_input = data.data_nu
            else:
                X_input = scprep.utils.to_array_or_spmatrix(data.data)# 将data.data转换为NumPy数组或稀疏矩阵
                if genes is not None and not (
                    isinstance(genes, str) and genes != "pca_only"
                ):
                    X_input = scprep.select.select_cols(X_input, idx=genes)
            if solver == "exact" and X_input.shape[1] > 6000:
                _logger.warning(
                    "Running AcImpute with `solver='exact'` on "
                    "{}-dimensional data may take a long time. "
                    "Consider denoising specific genes with `genes=<list-like>` "
                    "or using `solver='approximate'`.".format(X_input.shape[1])
                )
            X_AcImpute = self._impute(X_input, t_max=t_max, plot=plot_optimal_t, ax=ax)
            if store_result:
                self.X_AcImpute = X_AcImpute

        # return selected genes
        if isinstance(genes, str) and genes == "pca_only":
            X_AcImpute = PCA().fit_transform(X_AcImpute)
            genes = ["PC{}".format(i + 1) for i in range(X_AcImpute.shape[1])]
        elif solver == "approximate":
            X_AcImpute = data.inverse_transform(X_AcImpute, columns=genes)
        elif genes is not None and len(genes) != X_AcImpute.shape[1]:
            # select genes
            X_AcImpute = scprep.select.select_cols(X_AcImpute, idx=genes)

        # # convert back to pandas dataframe, if necessary
        # X_AcImpute = utils.convert_to_same_format(
        #     X_AcImpute, X, columns=genes, prevent_sparse=True
        # )
        return X_AcImpute

    def fit_transform(self, X, graph=None, **kwargs):

        with _logger.task("AcImpute"):
            global X_pri
            X_pri = X.copy()     #未筛选之前
            _logger.info(
                "Running AcImpute on {} cells and {} genes.".format(X_pri.shape[0], X_pri.shape[1])
            )
            global selected_columns
            selected_columns = (X.sum(axis=0) > 0.001) & ((X != 0).sum(axis=0) > 3)
            X = X.loc[:, selected_columns]
            global data_imputed_pre
            data_imputed_pre = X
            self.fit(X, graph=graph)
            X_AcImpute = self.transform(**kwargs)
        return X_AcImpute

    def _calculate_error(
        self, data, data_prev=None, weights=None, subsample_genes=None
    ):

        if subsample_genes is not None:
            data = data[:, subsample_genes]
        if weights is None:
            weights = np.ones(data.shape[1]) / data.shape[1]
        if data_prev is not None:
            _, _, error = spatial.procrustes(data_prev, data)
        else:
            error = None
        return error, data

    def _impute(
        self,
        data,
        t_max=20,
        plot=True,
        ax=None,
        max_genes_compute_t=500,
        threshold=0.001,
    ):


        global data_imputed_pre
        global high_var_genes
        data_imputed_pre =data_imputed_pre.values
        if data_imputed_pre.shape[1] > max_genes_compute_t:
            subsample_genes = np.random.choice(
                data_imputed_pre.shape[1], max_genes_compute_t, replace=False
            )
        else:
            subsample_genes = None
        if hasattr(data, "data_pca"):
            weights = None  # data.data_pca.explained_variance_ratio_
        else:
            weights = None
        if self.t == "auto":
            _, data_prev = self._calculate_error(
                data_imputed_pre,
                data_prev=None,
                weights=weights,
                subsample_genes=subsample_genes,
            )
            error_vec = []
            t_opt = None
        else:
            t_opt = self.t

        with _logger.task("get t_opt"):

            if (t_opt is not None) and (self.diff_op.shape[1] < data_imputed_pre.shape[1]):

                #
                diff_op_t = np.linalg.matrix_power(
                    scprep.utils.toarray(self.diff_op), t_opt
                )   #M^t
                data_imputed_pre = diff_op_t.dot(data_imputed_pre)  # (M^t) * D
            else:
                i = 0
                global normalize_data
                data_imputedt = normalize_data
                while (t_opt is None and i < t_max) or (
                    t_opt is not None and i < t_opt
                ):
                    i += 1
                    data_imputedt = self.diff_op.dot(data_imputedt)
                    if self.t == "auto":
                        error, data_prev = self._calculate_error(
                            data_imputedt,
                            data_prev,
                            weights=weights,
                            subsample_genes=subsample_genes,
                        )
                        error_vec.append(error)
                        _logger.debug("{}: {}".format(i, error_vec))
                        if error < threshold and t_opt is None:
                            t_opt = i + 1
                            _logger.info("Automatically selected t = {}".format(t_opt))
                        if i==t_max:
                            t_opt=t_max

        if plot:
            # continue to t_max
            with _logger.task("optimal t plot"):
                if t_opt is None:
                    # never converged
                    warnings.warn(
                        "optimal t > t_max ({})".format(t_max), RuntimeWarning
                    )
                else:
                    data_overimputed = data_imputedt
                    while i < t_max:
                        i += 1
                        data_overimputed = self.diff_op.dot(data_overimputed)
                        error, data_prev = self._calculate_error(
                            data_overimputed,
                            data_prev,
                            weights=weights,
                            subsample_genes=subsample_genes,
                        )
                        error_vec.append(error)

                # create axis
                if ax is None:
                    fig, ax = plt.subplots()
                    show = True
                else:
                    show = False

                # plot
                x = np.arange(len(error_vec)) + 1
                ax.plot(x, error_vec)
                if t_opt is not None:
                    ax.plot(
                        t_opt,
                        error_vec[t_opt - 1],
                        "ro",
                        markersize=10,
                    )
                ax.plot(x, np.full(len(error_vec), threshold), "k--")
                ax.set_xlabel("t")
                ax.set_ylabel("disparity(data_{t}, data_{t-1})")
                ax.set_xlim([1, len(error_vec)])
                plt.tight_layout()
            if show:
                plt.show(block=False)

        # time.sleep(10)
        with _logger.task("get P matrix"):
            diff_op_t = np.linalg.matrix_power(
                scprep.utils.toarray(self.diff_op), t_opt
            )  # M^t
            if self.X.shape[0] < 1000:
                maxn = ((self.X.shape[0] - 15) / (1000 - 15)) * 30
                maxn = math.ceil(maxn)
                if maxn < 5:
                    maxn = 5
            else:
                maxn = 100

            genem_norm = np.empty((self.diff_op.shape[1], data_imputed_pre.shape[1]))
            for i in range(self.diff_op.shape[1]):
                nonzero_indices = np.where(diff_op_t[i, :] != 0)[0]

                max_30_indices = heapq.nlargest(maxn, range(len(nonzero_indices)),
                                                key=diff_op_t[i, nonzero_indices].__getitem__)
                max_30_indices = nonzero_indices[max_30_indices]
                genem = np.mean(normalize_data[max_30_indices, :], axis=0)  # 使用归一化后的数据在进行后面的扩散

                genem = self.maxminnorm(genem, max(genem), min(genem))
                genem_norm[i, :] = genem
        with _logger.task("imputation"):
            data_imputed_g = np.empty((self.diff_op.shape[1], data_imputed_pre.shape[1]))
            #cells genes
            for i in range(data_imputed_pre.shape[0]):
                for j in range(data_imputed_pre.shape[1]):
                    if(genem_norm[i, j] == 1):
                        continue
                    diff_p = np.power(diff_op_t[i, :], genem_norm[i, j])
                    diff_p = diff_p / np.sum(diff_p)
                    data_imputed_g[i, j] = diff_p.dot(normalize_data[:, j])

            data_imputed_end = data_imputed_g
            nan_positions = np.isnan(data_imputed_end)

            if np.isnan(nan_positions).any():
                print("nan_positions 中包含 NaN，需要处理")
        with _logger.task("重缩放"):
            # 归一化后的重缩放
            max_values = np.max(data_imputed_end, axis=0)
            percentiles = np.percentile(data_imputed_pre, 99, axis=0)
            if np.isnan(percentiles).any():
                print("percentiles 中包含 NaN，需要处理")
            mask = max_values != 0  # max_values为0时不进行计算
            data_imputed_e = np.divide((data_imputed_end * percentiles), max_values, where=mask,
                                       out=np.zeros_like(data_imputed_end))
        with _logger.task("final imputation result"):
            global libsize
            data_imputed_e = data_imputed_e / np.median(libsize) * libsize[:, np.newaxis]
            data_imputed_e = np.where(data_imputed_pre == 0, data_imputed_e, data_imputed_pre)
            if np.isnan(data_imputed_e).any():
                print("data_imputed_e 中包含 NaN，需要处理")
            X_pri.loc[:, selected_columns] = data_imputed_e

        print("Porportion of zero:", (X_pri == 0).mean().mean())
        return X_pri

    def knnDREMI(
        self, gene_x, gene_y, k=10, n_bins=20, n_mesh=3, n_jobs=1, plot=False, **kwargs
    ):

        data = self.transform(genes=[gene_x, gene_y])
        dremi = scprep.stats.knnDREMI(
            data[gene_x],
            data[gene_y],
            k=k,
            n_bins=n_bins,
            n_mesh=n_mesh,
            n_jobs=n_jobs,
            plot=plot,
            **kwargs,
        )
        return dremi
    def maxminnorm(self,x,max,min):
        x = ((x-min)/(max-min))*(3-1)+1
        return x
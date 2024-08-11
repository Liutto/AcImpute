# AcImpute:A constraint-enhancing smooth-based approach for imputing single-cell RNA sequencing data

 An unsupervised method that enhances imputation accuracy by constraining the smoothing weights among cells for genes with different expression levels

## Quick start
'''python

new_path = r'D:\AcImpute'
sys.path.insert(0, new_path)
import AcImpute
import pandas as pd
import time
import numpy as np
start_time = time.time()

X = pd.read_csv("D：\datasets\Usoskin_silver\Usoskin_RAW.csv",header = 0,index_col=0)
X = X.transpose()  #转置函数
AcImpute_operator = AcImpute.AcImpute()
X_AcImpute = AcImpute_operator.fit_transform(X)
print("--- %s seconds ---" % (time.time() - start_time))
pd.DataFrame.to_csv(X_AcImpute.transpose(), "D：\datasets\Usoskin_silver\AcImpute.csv")
'''

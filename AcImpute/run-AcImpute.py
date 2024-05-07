import AcImpute
import pandas as pd
import time
import numpy as np
start_time = time.time()

X = pd.read_csv("E:/学习箱/论文/2/datasets/Usoskin_silver/Usoskin_RAW.csv",header = 0,index_col=0)
X = X.transpose()  #转置函数
magic_operator = AcImpute.AcImpute()
X_magic = magic_operator.fit_transform(X)
print("--- %s seconds ---" % (time.time() - start_time))
pd.DataFrame.to_csv(X_magic.transpose(), "E:/学习箱/论文/2/datasets/Usoskin_silver/test.csv")

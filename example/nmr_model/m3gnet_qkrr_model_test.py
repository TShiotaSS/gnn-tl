from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

import time
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_ridge import KernelRidge

import pandas as pd
import sys

from skqulacs.qnn import QNNRegressor
from skqulacs.circuit.pre_defined import create_qcl_ansatz,create_ibm_embedding_circuit,create_npqc_ansatz
from skqulacs.qnn.solver import Bfgs
from skqulacs.qnn.solver import Adam
from skqulacs.qkrr import QKRR

import numpy as np
import optuna


# 引数を取得
args = sys.argv
# print(args)
csv_name = args[1]
atomic_number = float(args[2])
n_qubit = int(args[3])
depth = int(args[4])
c = float(args[5])
alpha = 0.005846326121643412 # 適宜調整

if atomic_number == 6.0:
    element = "C"
    alpha = 0.006750312521595923 # 適宜調整
elif atomic_number == 7.0:
    element = "N"
    alpha = 0.005846326121643412 # 適宜調整
elif atomic_number == 8.0:
    element = "O"
    alpha = 0.005846326121643412 # 適宜調整
elif atomic_number == 9.0:
    element = "F"
    alpha = 0.005846326121643412 # 適宜調整
elif atomic_number == 1.0:
    element = "H"
    alpha = 0.005846326121643412 # 適宜調整
else:
    element = "H"

# drug_40のcsv
test_type_name = "pah"
# test_type_name = "12drug"


df_train =  pd.read_csv(f'/home/k-ishihara/sklearn/data/m3gnet/m3gnet_train_{element}_{csv_name}.csv') # 
df_test =  pd.read_csv(f'/home/k-ishihara/sklearn/data/m3gnet/m3gnet_{test_type_name}_total.csv') # 

df_train = df_train.sort_values(["dsgdb_idx","nmr_shift"],ascending=False)
df_test = df_test.sort_values(["dsgdb_idx","nmr_shift"],ascending=False)

with open(f"qkrr_{csv_name}_m3_test_{test_type_name}.log" , "a") as f:
    f.write(str(len(set(df_train["dsgdb_idx"]))))
    f.write("\n")
    f.write(str(len(set(df_test["dsgdb_idx"]))))
    f.write("\n")

# ここもいじる
# atomic_number = 6.0
# atomic_number = 1.0
with open(f"qkrr_{csv_name}_m3_test_{test_type_name}.log" , "a") as f:
    f.write(f"atomic_number : {str(atomic_number)}")
    f.write("\n")

df_train_element = df_train[df_train["atomic_number"]==atomic_number]
df_test_element = df_test[df_test["atomic_number"]==atomic_number]

# 説明変数と目的変数に分割する
X_train = df_train_element.loc[:, "atom_feature_vector_1":"atom_feature_vector_64"]
X_test = df_test_element.loc[:, "atom_feature_vector_1":"atom_feature_vector_64"]
y_train = df_train_element[["nmr_shift"]]
y_test = df_test_element[["nmr_shift"]]

# numpyにしとく
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy().reshape(-1,)
y_test = y_test.to_numpy().reshape(-1,)

with open(f"qkrr_{csv_name}_m3_test_{test_type_name}.log" , "a") as f:
    f.write(f"train:{str(len(y_train))}")
    f.write("\n")
    f.write(f"test:{str(len(y_test))}")
    f.write("\n")

start_time = time.time()

# n_qubit = 6
# depth = 30
# time_step = 0.5
# solver = Adam()
# maxiter= 1000
# circuit = create_qcl_ansatz(n_qubit, depth, time_step, 0)
circuit = create_npqc_ansatz(n_qubit, depth, c)
# qnn = QNNRegressor(circuit, solver)
qkrr = QKRR(circuit=circuit,alpha=alpha)
qkrr.fit_with_no_optimize_kernel(X_train, y_train)

with open(f"qkrr_{csv_name}_m3_test_{test_type_name}.log" , "a") as f:
    f.write(f"Time for qkrr fitting: {str(time.time() - start_time)} seconds")
    f.write("\n")
    f.write(f"n_qubit: {str(n_qubit)}")
    f.write("\n")
    f.write(f"depth: {str(depth)}")
    f.write("\n")
    f.write(f"scaling factor: {str(c)}")
    f.write("\n")
    f.write(f"alpha: {str(alpha)}")
    f.write("\n")
    # f.write(f"maxiter: {str(maxiter)}")
    # f.write("\n")

# 推論
import time
start_time_pre = time.time()

# 推論
predictions_qkrr_train = qkrr.predict_with_no_optimize_kernel(X_train)
predictions_qkrr_test = qkrr.predict_with_no_optimize_kernel(X_test)

from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import mean_squared_error  # MSE

# 平均平方二乗誤差(RMSE)
# print('RMSE 学習: %.2f, テスト: %.2f' % (
#         mean_squared_error(y_train, y_train_pred, squared=False), # 学習
#         mean_squared_error(y_test, y_test_pred, squared=False)    # テスト
#       ))

# print(kernel_ridge_tuned.best_params_)
# 平均絶対誤差(MAE)
with open(f"qkrr_{csv_name}_m3_test_{test_type_name}.log" , "a") as f:
    f.write(f"Time for Quantum KernelRidge prediction: {str(time.time() - start_time_pre)} seconds")
    f.write("\n")
    f.write(f'MAE 学習:{str(mean_absolute_error(y_train, predictions_qkrr_train))}')
    f.write("\n")
    f.write(f'MAE テスト:{str(mean_absolute_error(y_test, predictions_qkrr_test))}')
    f.write("\n")

# print(kernel_ridge_tuned.best_params_)
# 平均平方二乗誤差(RMSE)
with open(f"qkrr_{csv_name}_m3_test_{test_type_name}.log" , "a") as f:
    f.write(f'RMSE 学習:{str(np.sqrt(mean_squared_error(y_train, predictions_qkrr_train)))}')
    f.write("\n")
    f.write(f'RMSE テスト:{str(np.sqrt(mean_squared_error(y_test, predictions_qkrr_test)))}')
    f.write("\n")

### 予測結果をcsvファイルに出力する。
# trainデータの予測結果
df_train_element["nmr_shift_prediction"] = predictions_qkrr_train
atom_vector_col_name = [f"atom_feature_vector_{i+1}" for i in range(64)]
df_train_element.drop(atom_vector_col_name,axis=1).to_csv(f"{element}_prediction_train_result.csv")

# df_test_element["nmr_shift_prediction"] = predictions_kr_test
# atom_vector_col_name = [f"atom_feature_vector_{i+1}" for i in range(64)]
# df_test_element.drop(atom_vector_col_name,axis=1).to_csv("prediction_test_result_lap_opt.csv")
df_test_element["nmr_shift_prediction"] = predictions_qkrr_test
atom_vector_col_name = [f"atom_feature_vector_{i+1}" for i in range(64)]
df_test_element.drop(atom_vector_col_name,axis=1).to_csv(f"{element}_prediction_test_result_test_{test_type_name}.csv")


# 決定係数(R^2)
# print('R^2 学習: %.2f, テスト: %.2f' % (
#         r2_score(y_train, y_train_pred), # 学習
#         r2_score(y_test, y_test_pred)    # テスト
#       ))
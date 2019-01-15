#coding:utf-8
import numpy as np
from DataPreparation import get_data
#def OurMethod():
#读入数据
filepath = r"E:\Documents\learn_file\gradute\dataset\SmartwatchDataFilesAnonymized\SmartwatchDataFilesAnonymized"
valid_user_list = [1]
user_id = 1
feat_norm_flag = 1
pca_app_flag = 1
#num_pcomp是看pca中要保留的主成分个数
#None代表保留全部
num_pcomp = None
num_samples_imp = 1
gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = get_data(filepath, valid_user_list, user_id, feat_norm_flag, pca_app_flag, num_pcomp, num_samples_imp)
print(gen_trdata)
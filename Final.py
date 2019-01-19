#coding:utf-8
import numpy as np
from DataPreparation import get_data
from EvaluateOneClassClassifiers import *
#def OurMethod():
#读入数据
filepath = r"E:\Documents\learn_file\gradute\dataset\SmartwatchDataFilesAnonymized\SmartwatchDataFilesAnonymized"
valid_user_list = [1,2]
user_id = 1
feat_norm_flag = 1
pca_app_flag = 1
#num_pcomp是看pca中要保留的主成分个数
#None代表保留全部
num_pcomp = None
num_samples_imp = 1
gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = get_data(filepath, valid_user_list, user_id, feat_norm_flag, pca_app_flag, num_pcomp, num_samples_imp)

#初始化四个一类分类器
ee = Proposed('EEnvelop')
ifo = Proposed('Iforest')
lfo = Proposed('LOF')
svm1c = Proposed('SVM1C')
#--------输出生成的数据------------------------
ee.gen_tr_data = gen_tr_data
ee.gen_ts_data = gen_ts_data
ee.imp_tr_data = imp_tr_data
ee.imp_ts_data = imp_ts_data
#ee.print_input_data()
#--------生成真实用户训练数据的标签-------------
ee_gen_tr_label = ee.get_gen_tr_labels()
#print("ee_gen_tr_label的标签")
#print(ee_gen_tr_label)
#--------生成真实用户测试数据的标签-------------
ee_gen_ts_label = ee.get_gen_ts_labels()
#print("ee_gen_ts_label的标签")
#print(ee_gen_ts_label)
#--------生成虚假用户训练数据的标签（这里因为仅有真实用户所以就只能真实）-------------
ee_imp_tr_label = ee.get_imp_tr_labels()
#print("ee_imp_tr_label的标签")
#print(ee_imp_tr_label)
#--------生成虚假用户测试数据的标签（这里因为仅有真实用户所以就只能真实）-------------
ee_imp_ts_label = ee.get_imp_ts_labels()
#print("ee_imp_ts_label的标签")
#print(ee_imp_ts_label)
#--------错误率------------------------------
#norm_flag是数据要不要初始化，1为要初始化
single_norm_flag = 1
#n_components:  
#意义：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n
#类型：int 或者 string，缺省时默认为None，所有成分被保留。
#赋值为int，比如n_components=1，将把原始数据降到一个维度。
#赋值为string，比如n_components='mle'，将自动选取特征个数n，使得满足所要求的方差百分比。
#先设为None
single_num_comp = None
ee_far,ee_frr = ee.run_to_get_error_rates(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, single_norm_flag,single_num_comp)
print("ee_far")
print(ee_far)
print("ee_frr")
print(ee_frr)

#--------------融合后的错误率----------------------
# 先初始化一个
fus = Proposed('fusion')
#--------输出生成的数据------------------------
fus.gen_tr_data = gen_tr_data
fus.gen_ts_data = gen_ts_data
fus.imp_tr_data = imp_tr_data
fus.imp_ts_data = imp_ts_data
#fus.print_input_data()
#--------生成真实用户训练数据的标签-------------
fus_gen_tr_label = fus.get_gen_tr_labels()
#print("fus_gen_tr_label的标签")
#print(fus_gen_tr_label)
#--------生成真实用户测试数据的标签-------------
fus_gen_ts_label = fus.get_gen_ts_labels()
#print("fus_gen_ts_label的标签")
#print(fus_gen_ts_label)
#--------生成虚假用户训练数据的标签（这里因为仅有真实用户所以就只能真实）-------------
#fus_imp_tr_label = fus.get_imp_tr_labels()
#print("fus_imp_tr_label的标签")
#print(fus_imp_tr_label)
#--------生成虚假用户测试数据的标签（这里因为仅有真实用户所以就只能真实）-------------
#ee_imp_ts_label = ee.get_imp_ts_labels()
#print("ee_imp_ts_label的标签")
#print(ee_imp_ts_label)
#norm_flag,num_comp等参数设定
fus_norm_flag = 1
fus_num_comp = None
#weights是要看四个分类器是否都用
fus_weights =[1,1,1,1]
fus_threshold = 0.8
fus_far, fus_frr,fus_pr, fus_final_score_table = fus.run_fusion_to_get_error_rates(fus.gen_tr_data , fus.gen_ts_data , fus.imp_tr_data , fus.imp_ts_data , fus_norm_flag, fus_num_comp, fus_weights,fus_threshold)
print("fus_far")
print(fus_far)
print("fus_frr")
print(fus_frr)

def print_fus_final_score_table(fus_final_score_table):
        file = open(r'E:\Documents\learn_file\gradute\dataset\SmartwatchDataFilesAnonymized\result\result.txt','w')
        file.write(str(fus_final_score_table))
        file.close()
        return fus_final_score_table
print_fus_final_score_table(fus_final_score_table)
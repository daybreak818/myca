# Evaluating multi-class algorithms
import os
import warnings
import numpy as np
import pandas as pd
#from ISBAFinal import DataPreparation
#from ISBAFinal import ProposedMethod
warnings.filterwarnings("ignore")
# Loading data sets
dataset_name = 'Treadmill'
# dataset_name = 'ArmAcc40Users'
# dataset_name = 'ArmGyro40Users'
# dataset_name = 'SwipePMFusion33Users'
filepath = 'C:\\Users\\Rajesh\\OneDrive\\ISBA2017\\'+dataset_name+'\\FeatureFiles\\'
num_users= len(os.listdir(filepath+'Training'))
print('total users in the list: ',num_users)
user_list=range(1,num_users+1)
# invalid_user_list= [6,7,17,23,32] # for Swiping fusion data
invalid_user_list=[]
valid_user_list=set(user_list)-set(invalid_user_list)
user_counter=0
num_good_users = len(valid_user_list)
print('Processing total ',num_good_users, 'users')

# Data preprocessor setting and flags
feat_norm_flag =1 # feature normalization flag
pca_app_flag = 0 # to run PCA based dimensionality reduction
num_comp = 13  # number of components for PCA
num_samples_imp = 2 #

# Creating frame to storing errors
uletable = pd.DataFrame(columns=['clf','threshold','user','far','frr', 'hter'])
cletable = pd.DataFrame(columns=['clf', 'threshold','mfar','mfrr', 'mhter'])
# Prepare configuration for cross validation
seed = 13

# Configuring classification models
# classifier settings
num_neighbors = 7
rf_num_trees = 50
SVMKernel='rbf'
num_neurons = 100 # number of neurons per layer in MLP
# clf_to_run = ['RFC']
# models = []
# models.append(('ABoost', AdaBoostClassifier()))
# models.append(('DTree', DecisionTreeClassifier(max_depth=5)))
# models.append(('NBayes', GaussianNB()))
# models.append(('kNN', KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='auto', p=2)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('LReg', LogisticRegression()))
# models.append(('MLP', MLPClassifier(hidden_layer_sizes=(num_neurons,num_neurons, ), activation='relu', solver='adam', alpha=0.0001,
#                             batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
#                             max_iter=7000, shuffle=True, random_state=None, tol=0.0001, verbose=False,
#                             warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
#                             validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)))
# models.append(('RFC', RandomForestClassifier(n_estimators=rf_num_trees, criterion='gini', max_depth=None, min_samples_split=2,
#                                     min_samples_leaf=1,
#                                     min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None,
#                                     bootstrap=True, oob_score=False, n_jobs=1,
#                                     random_state=True,
#                                     verbose=0, warm_start=False, class_weight=None)))
# models.append(('SVC', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#                           decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
#                           max_iter=-1, probability=True, random_state=None, shrinking=True,
#                           tol=0.001, verbose=False)))
#
# Evaluate each model for each user
row_counter= -1
cls_counter= -1
# # Running traditional multi(two)-class classifiers
# for classifier, model in models:
#     # if classifier in clf_to_run:
#         print('Applying...', classifier)
#         for user in valid_user_list:
#             user_name = 'User' + str(user)
#             # print('Processing... user..', user)
#             gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = DataPreparation.get_data(filepath, valid_user_list,
#                                                                                           user, feat_norm_flag,
#                                                                                           pca_app_flag, num_comp,
#                                                                                           num_samples_imp)
#             tr_data = np.vstack((gen_tr_data, imp_tr_data))
#             tr_label = np.concatenate((DataPreparation.get_labels(gen_tr_data, 1), DataPreparation.get_labels(imp_tr_data, -1)))
#             ts_data = np.vstack((gen_ts_data, imp_ts_data))
#             ts_label = np.concatenate((DataPreparation.get_labels(gen_ts_data, 1), DataPreparation.get_labels(imp_ts_data, -1)))
#
#            model.fit(tr_data, tr_label)
#             pred_ts_lables = model.predict(ts_data)
#             tn, fp, fn, tp = confusion_matrix(ts_label, pred_ts_lables).ravel()
#
#             far = fp / (fp + tn)
#             frr = fn / (fn + tp)
#             row_counter = row_counter + 1
#             uletable.loc[row_counter] = [classifier, user_name, far, frr, (far + frr) / 2]
#
#         tempdf = uletable[uletable.clf == classifier]
#         cls_counter = cls_counter + 1
#         cletable.loc[cls_counter]= [classifier, tempdf['far'].mean(),tempdf['frr'].mean(), tempdf['hter'].mean()]
#

# #  Evaluating proposed model
# proposed_models = []
# proposed_models.append(('EEnvelop', ProposedMethod.Proposed('EEnvelop')))
# proposed_models.append(('Iforest', ProposedMethod.Proposed('Iforest')))
# proposed_models.append(('LOF', ProposedMethod.Proposed('LOF')))
# proposed_models.append(('SVM1C', ProposedMethod.Proposed('SVM1C')))
# for cls_name, classifier in proposed_models:
#     # if classifier in clf_to_run:
#         print('Applying...', cls_name)
#         for user in valid_user_list:
#             user_name = 'User' + str(user)
#             # print('Processing... user..', user)
#             gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = DataPreparation.get_data(filepath, valid_user_list,
#                                                                                           user, feat_norm_flag,
#                                                                                           pca_app_flag, num_comp,
#                                                                                           num_samples_imp)
#             far, frr = classifier.run_to_get_error_rates(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, feat_norm_flag, num_comp)
#             row_counter = row_counter + 1
#             uletable.loc[row_counter] = [cls_name, user_name, far, frr, (far+frr)/2]
#
#         tempdf = uletable[uletable.clf == cls_name]
#         cls_counter = cls_counter + 1
#         cletable.loc[cls_counter]= [cls_name, tempdf['far'].mean(),tempdf['frr'].mean(), tempdf['hter'].mean()]
#
# # print('userwise results')
# # print(uletable)
# print('clswise avg. results')
# print(cletable)

def getWeights(number):
    str = format(number, '04b')
    lw = [int(str[0]),int(str[1]),int(str[2]),int(str[3])]
    return lw

#  Evaluating proposed fusion model
num_classifiers = 4
#  All possible fusion combinations
threshold = 0.86
total_comb = pow(2,num_classifiers)
total_comb = 2
for comb in range(1,total_comb):
    weights = getWeights(comb)
    cls_name = ''.join(str(weights))
    print('Applying fusion with weights', cls_name, 'and threshold', threshold)
    for user in valid_user_list:
        cu_model = ProposedMethod.Proposed('Fusion')
        user_name = 'User' + str(user)
        # print('Processing... user..', user)
        gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data = DataPreparation.get_data(filepath, valid_user_list,
                                                                                      user, feat_norm_flag,
                                                                                      pca_app_flag, num_comp,
                                                                                      num_samples_imp)
        far, frr, final_score_table = cu_model.run_fusion_to_get_error_rates(gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data, feat_norm_flag, num_comp, weights,threshold)
        row_counter = row_counter + 1
        uletable.loc[row_counter] = [cls_name, threshold, user_name, far, frr, (far+frr)/2]
    tempdf = uletable[uletable.threshold == threshold]
    cls_counter = cls_counter + 1
    cletable.loc[cls_counter]= [cls_name, threshold, tempdf['far'].mean(),tempdf['frr'].mean(), tempdf['hter'].mean()]


print('userwise results')
print(uletable)
print('clswise avg. results')
print(cletable)

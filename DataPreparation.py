# coding=UTF-8
import numpy as np
#导入sklearn中StandardScaler,目的是为了去均值和方差归一化
from sklearn.preprocessing import StandardScaler
#PCA是主成分分析，用于非线性数据的降维
from sklearn.decomposition import PCA
#def是定义函数
#
def get_imp_data(filepath, user_id, valid_user_list, num_samples_imp):
    # select three samples of imposter from every other user than the candidate
    #从其他人而非候选人中选择三个骗子的样本
    #possible_imp_set应该是从所有用户列表中去除掉真实的候选者用户，其代表的是三个骗子样本可以进行选择的人员列表
    possible_imp_set = set(valid_user_list) - set([user_id])
    print(possible_imp_set)
    # print('valid_user_list',valid_user_list)
    # print('possible_imp_set',possible_imp_set)

    count_imp=-1
    for imp in possible_imp_set:
        current_imp = 'User' + str(imp)
       #构建其他人员的训练数据和测试数据
#        tr_file = filepath + 'Training\\' + current_imp + '.txt'
#        ts_file = filepath + 'Testing\\' + current_imp + '.txt'
        tr_file = filepath + '\\' + current_imp + '\\Training' + '\\Accelerometer_Data.txt'
        ts_file = filepath + '\\' + current_imp + '\Testing' + '\Accelerometer_Data.txt'
        #载入文件中的数据
        tr_data = np.loadtxt(tr_file, delimiter=',')
        ts_data = np.loadtxt(ts_file, delimiter=',')
        # slicing the imp data to avoid the class imbalance
        #对数据进行切片，为了避免类不平衡
        tr_data = tr_data[1:num_samples_imp, :]
        ts_data = ts_data[1:num_samples_imp, :]
        count_imp = count_imp + 1 # I had forgotten this
        #代表仅循环了一次，不需要切片
        if count_imp == 0:
            imp_data_tr = tr_data
            imp_data_ts = ts_data
#            return tr_data,ts_data
        else:
            imp_data_tr = np.vstack((imp_data_tr, tr_data))
            imp_data_ts = np.vstack((imp_data_ts, ts_data))
    return imp_data_tr, imp_data_ts
# Generating label column for the given data matrix
#对于已经被给的数据矩阵生成一个标签
def get_labels(data_matrix, label):
    if data_matrix.shape[0] > 1:
        label_column = np.empty(data_matrix.shape[0])
        label_column.fill(label)
    else:
        print('Warning! user data contains only one sample')
    return label_column

def get_data(filepath, valid_user_list, user_id, feat_norm_flag, pca_app_flag, num_pcomp, num_samples_imp):
    # print('Working on : ',CUser),
#    global imp_tr_data
#    global imp_ts_data
    candidate_user = 'User'+str(user_id)
#    tr_file =  filepath + 'Training\\' + candidate_user+ '.txt'
#    ts_file = filepath + 'Testing\\' + candidate_user + '.txt'
    tr_file = filepath + '\\' + candidate_user + '\\Training'+ '\\Accelerometer_Data.txt'
#    print(tr_file)
    ts_file = filepath + '\\' + candidate_user + '\Testing' + '\Accelerometer_Data.txt'
    # Preparing genuine train and test data
    #真实用户的数据
    gen_tr_data = np.loadtxt(tr_file, delimiter=',')
    gen_ts_data = np.loadtxt(ts_file, delimiter=',')
#    print(gen_tr_data)
    imp_tr_data, imp_ts_data = get_imp_data(filepath, user_id, valid_user_list, num_samples_imp)

    if feat_norm_flag == 1:
        ############## Feature scaling ##################
        #特征的处理
        scaler = StandardScaler()
        # Fit only on training data
        #仅仅对真实用户进行了这样的操作
        scaler.fit(gen_tr_data)
        gen_tr_data = scaler.transform(gen_tr_data)
        gen_ts_data = scaler.transform(gen_ts_data)
        # apply same transformation to test data
        #在对测试的数据进行操作
        imp_tr_data = scaler.transform(imp_tr_data)
        imp_ts_data = scaler.transform(imp_ts_data)

    if pca_app_flag == 1:
        ############## Dimensionality Reduction ##################
        #降维
        pca = PCA(n_components=num_pcomp)
        # Don't cheat - fit only on training data
        pca.fit(gen_tr_data)
        gen_tr_data = pca.transform(gen_tr_data)
        gen_ts_data = pca.transform(gen_ts_data)
        # apply same transformation to test data
        imp_tr_data = pca.transform(imp_tr_data)
        imp_ts_data = pca.transform(imp_ts_data)

    # if class_balancing == 1:
    #     ############## Feature scaling ##################
    #     sampler = SMOTE(random_state=random_state)
    #     # Don't cheat - fit only on training data
    #     print('gen_tr_data.shape',gen_tr_data.shape, 'before oversampling')
    #     print('gen_ts_data.shape', gen_ts_data.shape, 'before oversampling')
    #     gen_tr_data = sampler.fit_sample(gen_tr_data,act_gen_tr_labels)
    #     gen_tr_data = sampler.fit_sample(gen_tr_data, act_gen_tr_labels)
    #     print('gen_tr_data.shape', gen_tr_data.shape, 'after oversampling')
    #     print('gen_ts_data.shape', gen_ts_data.shape, 'after oversampling')
    #     # apply same transformation to test data
    #     imp_tr_data = scaler.transform(imp_tr_data)
    #     imp_ts_data = scaler.transform(imp_ts_data)
    return gen_tr_data, gen_ts_data, imp_tr_data, imp_ts_data
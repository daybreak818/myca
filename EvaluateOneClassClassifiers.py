# coding=UTF-8
#去均值和方差归一化
from sklearn.preprocessing import StandardScaler
#主成分分析
from sklearn.decomposition import PCA
#用于对数据的鲁棒协方差估计，从而将椭圆适配到中央数据点，忽略中央模式之外的点
from sklearn.covariance import EllipticEnvelope
#异常检测算法
from sklearn.ensemble import IsolationForest
#SVM分类器
from sklearn import svm
#局部异常因子
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import os
#混淆矩阵，计算分类的准确率
from sklearn.metrics import confusion_matrix
#from Final import OurMethod
class Proposed:
    def __init__(self, classifier):
        self.gen_tr_data = []
        self.gen_ts_data = []
        self.imp_tr_data = []
        self.imp_ts_data = []
        self.classifier = classifier

    def run_to_get_error_rates(self,gen_tr, gen_ts, imp_tr, imp_ts, norm_flag,num_comp):
        self.gen_tr_data = gen_tr
        self.gen_ts_data = gen_ts
        self.imp_tr_data = imp_tr
        self.imp_ts_data = imp_ts
        if norm_flag ==1:
            self.normalize_features()

        if self.classifier == 'EEnvelop':
            self.apply_pca(num_comp)
            far, frr = self.envelop()
        elif self.classifier == 'Iforest':
            far, frr = self.iforest()
        elif self.classifier == 'LOF':
            far, frr = self.lof()
        elif self.classifier == 'SVM1C':
            far, frr = self.svm1c()
        else:
            print('sorry!, entered classifier is unavailable')
        return far, frr

    def run_fusion_to_get_error_rates(self,gen_tr, gen_ts, imp_tr, imp_ts, norm_flag, num_comp, weights,threshold):
        self.gen_tr_data = gen_tr
        self.gen_ts_data = gen_ts
        self.imp_tr_data = imp_tr
        self.imp_ts_data = imp_ts
        self.threshold = threshold
        if norm_flag == 1:
            self.normalize_features()
        far, frr, final_score_table = self.fuse_to_get_results(weights,num_comp)
        return far, frr, final_score_table

    def print_input_data(self):
        print('gen_tr_data', self.gen_tr_data)
        print('gen_ts_data', self.gen_ts_data)
        print('imp_tr_data', self.imp_tr_data)
        print('imp_ts_data', self.imp_ts_data)

    def get_gen_tr_labels(self):
        if self.gen_tr_data.shape[0] > 1:
            label_column = np.empty(self.gen_tr_data.shape[0])
            label_column.fill(1)
        else:
            print('Warning! gen training data contains only one sample')
        return label_column

    def fill_sc_with_zero(self,ref_size):
        if  ref_size.shape[0] > 1:
            label_column = np.empty(ref_size.shape[0])
            label_column.fill(0)
        else:
            print('Warning! gen training data contains only one sample')
        return label_column

    def get_gen_ts_labels(self):
        if self.gen_ts_data.shape[0] > 1:
            label_column = np.empty(self.gen_ts_data.shape[0])
            label_column.fill(1)
        else:
            print('Warning! genuine testing data contains only one sample')
        return label_column

    def get_imp_tr_labels(self):
        if self.imp_tr_data.shape[0] > 1:
            label_column = np.empty(self.imp_tr_data.shape[0])
            label_column.fill(-1)
        else:
            print('Warning! imp training data contains only one sample')
        return label_column

    def get_imp_ts_labels(self):
        if self.imp_ts_data.shape[0] > 1:
            label_column = np.empty(self.imp_ts_data.shape[0])
            label_column.fill(-1)
        else:
            print('Warning! imp testing data contains only one sample')
        return label_column

    def normalize_features(self):
        ############## Feature scaling ##################
        scaler = StandardScaler()
        # Don't cheat - fit only on training data
        scaler.fit(self.gen_tr_data)
        gen_tr_data = scaler.transform(self.gen_tr_data)
        gen_ts_data = scaler.transform(self.gen_ts_data)
        # apply same transformation to test data
        imp_tr_data = scaler.transform(self.imp_tr_data)
        imp_ts_data = scaler.transform(self.imp_ts_data)

    def apply_pca(self, num_comp):
        ############## Dimensionality Reduction ##################
        pca = PCA(n_components=num_comp)
        # Don't cheat - fit only on training data
        pca.fit(self.gen_tr_data)
        self.gen_tr_data = pca.transform(self.gen_tr_data)
        self.gen_ts_data = pca.transform(self.gen_ts_data)
        # apply same transformation to test data
        self.imp_tr_data = pca.transform(self.imp_tr_data)
        self.imp_ts_data = pca.transform(self.imp_ts_data)

    def envelop(self):
        # Make sure you apply pca before using Envelop -- it is very sensitive to the feature dimensions
        clf_een = EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=0.25,
                                   contamination=0.1,
                                   random_state=True)

        # Fitting the model on reduced dimensionality
        clf_een.fit(self.gen_tr_data)

        # Prediction labels
        pred_gen_ts_labels = clf_een.predict(self.gen_ts_data)
        pred_imp_ts_labels = clf_een.predict(self.imp_ts_data)

        act_ts_labels = np.concatenate((self.get_gen_ts_labels(),self.get_imp_ts_labels()))
        pred_ts_labels = np.concatenate((pred_gen_ts_labels, pred_imp_ts_labels))

        tn, fp, fn, tp = confusion_matrix(act_ts_labels, pred_ts_labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        pr = tp / (tp + fp)
        return far, frr, pr

    def iforest(self):
        # Make sure you apply pca before using Envelop -- it is very sensitive to the feature dimensions
        clf_if = IsolationForest(max_samples="auto", contamination=0.2,random_state=True)

        # Fitting the model on reduced dimensionality
        clf_if.fit(self.gen_tr_data)

        # Prediction labels
        pred_gen_ts_labels = clf_if.predict(self.gen_ts_data)
        pred_imp_ts_labels = clf_if.predict(self.imp_ts_data)

        act_ts_labels = np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels()))
        pred_ts_labels = np.concatenate((pred_gen_ts_labels, pred_imp_ts_labels))
        tn, fp, fn, tp = confusion_matrix(act_ts_labels, pred_ts_labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        pr = tp / (tp + fp)
        return far, frr, pr

    def lof(self):
        clf_lof = LocalOutlierFactor(n_neighbors=35, metric='l2', contamination=0.25)
        X = np.concatenate((self.gen_tr_data, self.gen_ts_data))
        X_all = np.concatenate((X, self.imp_ts_data))
        pred_all_score = clf_lof.fit_predict(X_all)
        act_ts_labels = np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels()))
        pred_ts_labels = pred_all_score[range(len(self.gen_tr_data), len(pred_all_score)),]
        tn, fp, fn, tp = confusion_matrix(act_ts_labels, pred_ts_labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        pr = tp / (tp + fp)
        return far, frr, pr

    def svm1c(self):
        # Make sure you apply pca before using Envelop -- it is very sensitive to the feature dimensions
        clf_svm1c = svm.OneClassSVM(kernel='rbf', degree=3, gamma=0.001, coef0=0.0, tol=0.00001, nu=0.001,
                      shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=True)

        # Fitting the model on reduced dimensionality
        clf_svm1c.fit(self.gen_tr_data)

        # Prediction labels
        pred_gen_ts_labels = clf_svm1c.predict(self.gen_ts_data)
        pred_imp_ts_labels = clf_svm1c.predict(self.imp_ts_data)

        act_ts_labels = np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels()))
        pred_ts_labels = np.concatenate((pred_gen_ts_labels, pred_imp_ts_labels))
        tn, fp, fn, tp = confusion_matrix(act_ts_labels, pred_ts_labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        pr = tp / (tp + fp)
        return far, frr, pr

    def mymm_scaler(self,sup_scores):  # Scales supplied list to 0, 1
        minimum = min(sup_scores)
        maximum = max(sup_scores)
        scaled_scores = []
        for item in sup_scores:
            norm_item = (item - minimum) / (maximum - minimum)
            scaled_scores.append(norm_item)
        return scaled_scores

    def fuse_to_get_results(self, weights, num_comp):
        if weights[0] != 0:
            self.apply_pca(num_comp)
            # Make sure you apply pca before using Envelop -- it is very sensitive to the feature dimensions
            clf_een = EllipticEnvelope(store_precision=True, assume_centered=False, support_fraction=0.25,
                                       contamination=0.1,
                                       random_state=True)
            # Fitting the model on reduced dimensionality
            clf_een.fit(self.gen_tr_data)
            # The anomaly score of the input samples. The lower, the more abnormal.
            #输入样本的异常分数。越低越不正常。
            pred_gen_scores_ee = clf_een.decision_function(self.gen_ts_data)
            pred_imp_scores_ee = clf_een.decision_function(self.imp_ts_data)
            pred_scores_ts_ee = np.concatenate((pred_gen_scores_ee, pred_imp_scores_ee))
            norm_scores_ee = self.mymm_scaler(pred_scores_ts_ee)
        else:
            norm_scores_ee = self.fill_sc_with_zero(np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels())))
        if weights[1] != 0:
            # Make sure you apply pca before using envelop -- it is very sensitive to the feature dimensions
            clf_if = IsolationForest(max_samples="auto", contamination=0.2, random_state=True)
            # Fitting the model on reduced dimensionality
            clf_if.fit(self.gen_tr_data)
            # The anomaly score of the input samples. The lower, the more abnormal.
            pred_gen_scores_if = clf_if.decision_function(self.gen_ts_data)
            pred_imp_scores_if = clf_if.decision_function(self.imp_ts_data)
            # print('pred_gen_scores_if',self.mymm_scaler(pred_gen_scores_if))
            # print(clf_if.predict(self.gen_ts_data))
            # print('pred_imp_scores_if', self.mymm_scaler(pred_imp_scores_if))
            # print(clf_if.predict(self.imp_ts_data))

            pred_scores_ts_if = np.concatenate((pred_gen_scores_if, pred_imp_scores_if))
            norm_scores_if = self.mymm_scaler(pred_scores_ts_if)
            # print('norm_scores_if',norm_scores_if)
            # print('plabel',np.concatenate((clf_if.predict(self.gen_ts_data),clf_if.predict(self.imp_ts_data))))
        else:
            norm_scores_if = self.fill_sc_with_zero(np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels())))
        if weights[2] != 0:
            num_neighbors = 35
            clf_lof = LocalOutlierFactor(n_neighbors=num_neighbors, metric='l2', contamination=0.25)
            X = np.concatenate((self.gen_tr_data, self.gen_ts_data))
            X_all = np.concatenate((X, self.imp_ts_data))
            pred_all_score = clf_lof.fit_predict(X_all)
            #print('pred_all_score')
            #print(pred_all_score)
            pred_scores_ts_lof = pred_all_score[range(len(self.gen_tr_data), len(pred_all_score)),]
            norm_scores_lof = self.mymm_scaler(pred_scores_ts_lof)
        else:
            norm_scores_lof = self.fill_sc_with_zero(np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels())))

        if weights[3] != 0:
            # Make sure you apply pca before using envelop -- it is very sensitive to the feature dimensions
            clf_svm1c = svm.OneClassSVM(kernel='rbf', degree=3, gamma=0.001, coef0=0.0, tol=0.00001, nu=0.001,
                                        shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=True)
            # Fitting the model on reduced dimensionality
            clf_svm1c.fit(self.gen_tr_data)
            # The anomaly score of the input samples. The lower the more abnormal.
            pred_gen_scores_svm = clf_svm1c.decision_function(self.gen_ts_data)
            pred_imp_scores_svm = clf_svm1c.decision_function(self.imp_ts_data)
            pred_scores_ts_svm = np.concatenate((pred_gen_scores_svm, pred_imp_scores_svm))
            norm_scores_svm = self.mymm_scaler(pred_scores_ts_svm)
        else:
            norm_scores_svm = self.fill_sc_with_zero(np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels())))

        # Score level fusion
        pred_ts_labels = []
        fused_scores = []
        for ees, ifs, lofs, svms in zip(norm_scores_ee, norm_scores_if, norm_scores_lof, norm_scores_svm):
            cfscore = (weights[0] * ees + weights[1] * ifs + weights[2]*lofs + weights[3] * svms) / sum(weights)
            fused_scores.append(cfscore)
            if cfscore < self.threshold:
                pred_ts_labels.append(-1)
            else:
                pred_ts_labels.append(1)

        act_ts_labels = np.concatenate((self.get_gen_ts_labels(), self.get_imp_ts_labels()))
        tn, fp, fn, tp = confusion_matrix(act_ts_labels, pred_ts_labels).ravel()
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        pr = tp / (tp + fp)
        final_score_table = [norm_scores_ee, norm_scores_if, norm_scores_lof, norm_scores_svm, fused_scores, act_ts_labels]
        #ee分数
        print(norm_scores_ee)
        #if分数
        print(norm_scores_if)
        #lof是0，1标签
        print(norm_scores_lof)
        #svm分数
        print(norm_scores_svm)
        #混合后也是分数
        print(fused_scores)
        #标签
        print(act_ts_labels)
        return far, frr, pr, final_score_table

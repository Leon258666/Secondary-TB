from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import interp

from statsmodels.stats.proportion import proportion_confint
# svm/随机森林、决策树、knn、lda、adaboost、lr、lgbm、coxboost、rsf、glmboost、gbm、glmnet



rnd_clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=1, min_samples_split=2, oob_score=True, class_weight='balanced', random_state=16)
svm_clf = SVC(C=1.5, decision_function_shape='ovo', probability=True, random_state=16)
knn_clf = KNeighborsClassifier(n_neighbors=3, p=2, weights='uniform', leaf_size=50)
ada_clf = AdaBoostClassifier(n_estimators=20, base_estimator=rnd_clf, learning_rate=0.1, random_state=16)
lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
# solver=svd/lsqr/eigen
gbm_clf = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, n_estimators=200, min_samples_split=3, subsample=0.8, random_state=16)

lrl1_clf = LogisticRegression(penalty='l1', C=0.5, solver="liblinear", random_state=20)
lrl2_clf = LogisticRegression(penalty='l2', C=0.5, solver="liblinear", random_state=20)
lgbm_clf = lightgbm.LGBMClassifier(objective='binary', max_depth=4, num_leaves=25, learning_rate=0.1, n_estimators=1000, min_child_samples=80,
                                   subsample=0.8, colsample_bytree=1, reg_alpha=0, reg_lambda=0, random_state=20)
cat_clf = CatBoostClassifier(iterations=500, depth=3, learning_rate=0.1, loss_function='MultiClass', logging_level='Silent')
# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pre = clf.predict(X_test)
#     print(clf.__class__, accuracy_score(y_pre, y_test))




# df = pd.read_csv(r'C:\Users\69559\Desktop\deepfeatures\combine56.csv')
df = pd.read_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\combine56.csv')


# 56firsttime
# certain_feature_names = ['square_glcm_Imc2',
#                          'wavelet-LLL_firstorder_Skewness',
#                          'original_shape_Maximum2DDiameterRow',
#                          'wavelet-LLH_glszm_GrayLevelNonUniformityNormalized',
#                          'wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis',
#                          'wavelet-LLH_glszm_GrayLevelVariance']

# certain_feature_names = ['CC_deepfeature_377', 'CC_deepfeature_444', 'CC_deepfeature_907', 'CC_deepfeature_978', 'CC_deepfeature_1039', 'CC_deepfeature_1308', 'CC_deepfeature_1976', 'CC_deepfeature_1977']
# rfecvdeep
# certain_feature_names = ['CC_deepfeature_7' ,'CC_deepfeature_9' ,'CC_deepfeature_11',
#  'CC_deepfeature_13' ,'CC_deepfeature_14', 'CC_deepfeature_19',
#  'CC_deepfeature_23' ,'CC_deepfeature_24', 'CC_deepfeature_27',
#  'CC_deepfeature_30', 'CC_deepfeature_56', 'CC_deepfeature_59',
#  'CC_deepfeature_72', 'CC_deepfeature_93', 'CC_deepfeature_94',
#  'CC_deepfeature_97' ,'CC_deepfeature_98', 'CC_deepfeature_100',
#  'CC_deepfeature_107', 'CC_deepfeature_110', 'CC_deepfeature_111',
#  'CC_deepfeature_113' ,'CC_deepfeature_114', 'CC_deepfeature_116',
#  'CC_deepfeature_118', 'CC_deepfeature_128', 'CC_deepfeature_140',
#  'CC_deepfeature_141', 'CC_deepfeature_142', 'CC_deepfeature_145',
#  'CC_deepfeature_150' ,'CC_deepfeature_164', 'CC_deepfeature_167',
#  'CC_deepfeature_182' ,'CC_deepfeature_188', 'CC_deepfeature_210',
#  'CC_deepfeature_219', 'CC_deepfeature_232', 'CC_deepfeature_233',
#  'CC_deepfeature_236' ,'CC_deepfeature_237', 'CC_deepfeature_240',
#  'CC_deepfeature_243', 'CC_deepfeature_253', 'CC_deepfeature_254',
#  'CC_deepfeature_256' ,'CC_deepfeature_264', 'CC_deepfeature_268',
#  'CC_deepfeature_275', 'CC_deepfeature_278', 'CC_deepfeature_279',
#  'CC_deepfeature_285', 'CC_deepfeature_289', 'CC_deepfeature_292',
#  'CC_deepfeature_294', 'CC_deepfeature_304', 'CC_deepfeature_309',
#  'CC_deepfeature_310', 'CC_deepfeature_337', 'CC_deepfeature_343',
#  'CC_deepfeature_346', 'CC_deepfeature_348', 'CC_deepfeature_360',
#  'CC_deepfeature_361', 'CC_deepfeature_370', 'CC_deepfeature_373',
#  'CC_deepfeature_385', 'CC_deepfeature_394', 'CC_deepfeature_396',
#  'CC_deepfeature_418', 'CC_deepfeature_421', 'CC_deepfeature_423',
#  'CC_deepfeature_429', 'CC_deepfeature_430', 'CC_deepfeature_431',
#  'CC_deepfeature_432', 'CC_deepfeature_436', 'CC_deepfeature_438',
#  'CC_deepfeature_454', 'CC_deepfeature_457', 'CC_deepfeature_459',
#  'CC_deepfeature_465', 'CC_deepfeature_466', 'CC_deepfeature_470',
#  'CC_deepfeature_474', 'CC_deepfeature_483' ,'CC_deepfeature_493',
#  'CC_deepfeature_500', 'CC_deepfeature_504' ,'CC_deepfeature_506', 'square_glcm_Imc2',
#                          'wavelet-LLL_glcm_MCC',
#                          'wavelet-LLL_firstorder_Skewness',
#                          'original_shape_Maximum2DDiameterRow',
#                          'wavelet-LLH_glszm_GrayLevelNonUniformityNormalized',
#                          'wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis',
#                          'original_glcm_Imc2',
#                          'wavelet-LLH_glszm_HighGrayLevelZoneEmphasis']


certain_feature_names = ['CC_deepfeature_3', 'CC_deepfeature_4', 'CC_deepfeature_5', 'CC_deepfeature_6', 'CC_deepfeature_8', 'CC_deepfeature_9', 'CC_deepfeature_10', 'CC_deepfeature_11', 'CC_deepfeature_12', 'CC_deepfeature_13', 'CC_deepfeature_14', 'CC_deepfeature_16', 'CC_deepfeature_17', 'CC_deepfeature_19', 'CC_deepfeature_21', 'CC_deepfeature_24', 'CC_deepfeature_25', 'CC_deepfeature_29', 'CC_deepfeature_31', 'CC_deepfeature_33', 'CC_deepfeature_34', 'CC_deepfeature_35', 'CC_deepfeature_36', 'CC_deepfeature_37', 'CC_deepfeature_38', 'CC_deepfeature_40', 'CC_deepfeature_42', 'CC_deepfeature_49', 'CC_deepfeature_51', 'CC_deepfeature_52', 'CC_deepfeature_53', 'CC_deepfeature_54', 'CC_deepfeature_57', 'CC_deepfeature_59', 'CC_deepfeature_62', 'CC_deepfeature_63', 'CC_deepfeature_64', 'CC_deepfeature_65', 'CC_deepfeature_70', 'CC_deepfeature_71', 'CC_deepfeature_73', 'CC_deepfeature_74', 'CC_deepfeature_76', 'CC_deepfeature_77', 'CC_deepfeature_79', 'CC_deepfeature_80', 'CC_deepfeature_82', 'CC_deepfeature_84', 'CC_deepfeature_87', 'CC_deepfeature_90', 'CC_deepfeature_91', 'CC_deepfeature_92', 'CC_deepfeature_94', 'CC_deepfeature_95', 'CC_deepfeature_99', 'CC_deepfeature_100', 'CC_deepfeature_103', 'CC_deepfeature_104', 'CC_deepfeature_105', 'CC_deepfeature_109', 'CC_deepfeature_112', 'CC_deepfeature_114', 'CC_deepfeature_115', 'CC_deepfeature_116', 'CC_deepfeature_117', 'CC_deepfeature_118', 'CC_deepfeature_119', 'CC_deepfeature_120', 'CC_deepfeature_121', 'CC_deepfeature_122', 'CC_deepfeature_125', 'CC_deepfeature_127', 'CC_deepfeature_130', 'CC_deepfeature_133', 'CC_deepfeature_135', 'CC_deepfeature_136', 'CC_deepfeature_137', 'CC_deepfeature_139', 'CC_deepfeature_141', 'CC_deepfeature_143', 'CC_deepfeature_144', 'CC_deepfeature_146', 'CC_deepfeature_148', 'CC_deepfeature_149', 'CC_deepfeature_151', 'CC_deepfeature_152', 'CC_deepfeature_154', 'CC_deepfeature_155', 'CC_deepfeature_156', 'CC_deepfeature_158', 'CC_deepfeature_160', 'CC_deepfeature_162', 'CC_deepfeature_163', 'CC_deepfeature_165', 'CC_deepfeature_167', 'CC_deepfeature_168', 'CC_deepfeature_170', 'CC_deepfeature_171', 'CC_deepfeature_172', 'CC_deepfeature_173', 'CC_deepfeature_175', 'CC_deepfeature_177', 'CC_deepfeature_180', 'CC_deepfeature_181', 'CC_deepfeature_182', 'CC_deepfeature_183', 'CC_deepfeature_185', 'CC_deepfeature_187', 'CC_deepfeature_188', 'CC_deepfeature_189', 'CC_deepfeature_191', 'CC_deepfeature_192', 'CC_deepfeature_193', 'CC_deepfeature_194', 'CC_deepfeature_195', 'CC_deepfeature_196', 'CC_deepfeature_197', 'CC_deepfeature_202', 'CC_deepfeature_203', 'CC_deepfeature_204', 'CC_deepfeature_205', 'CC_deepfeature_208', 'CC_deepfeature_209', 'CC_deepfeature_210', 'CC_deepfeature_214', 'CC_deepfeature_216', 'CC_deepfeature_218', 'CC_deepfeature_219', 'CC_deepfeature_220', 'CC_deepfeature_223', 'CC_deepfeature_224', 'CC_deepfeature_226', 'CC_deepfeature_227', 'CC_deepfeature_228', 'CC_deepfeature_229', 'CC_deepfeature_231', 'CC_deepfeature_236', 'CC_deepfeature_238', 'CC_deepfeature_240', 'CC_deepfeature_241', 'CC_deepfeature_246', 'CC_deepfeature_247', 'CC_deepfeature_248', 'CC_deepfeature_250', 'CC_deepfeature_256', 'CC_deepfeature_259', 'CC_deepfeature_260', 'CC_deepfeature_264', 'CC_deepfeature_265', 'CC_deepfeature_266', 'CC_deepfeature_267', 'CC_deepfeature_268', 'CC_deepfeature_270', 'CC_deepfeature_271', 'CC_deepfeature_275', 'CC_deepfeature_276', 'CC_deepfeature_280', 'CC_deepfeature_283', 'CC_deepfeature_284', 'CC_deepfeature_285', 'CC_deepfeature_286', 'CC_deepfeature_288', 'CC_deepfeature_290', 'CC_deepfeature_292', 'CC_deepfeature_293', 'CC_deepfeature_299', 'CC_deepfeature_300', 'CC_deepfeature_302', 'CC_deepfeature_303', 'CC_deepfeature_304', 'CC_deepfeature_306', 'CC_deepfeature_308', 'CC_deepfeature_310', 'CC_deepfeature_311', 'CC_deepfeature_315', 'CC_deepfeature_318', 'CC_deepfeature_319', 'CC_deepfeature_320', 'CC_deepfeature_325', 'CC_deepfeature_327', 'CC_deepfeature_330', 'CC_deepfeature_331', 'CC_deepfeature_334', 'CC_deepfeature_335', 'CC_deepfeature_337', 'CC_deepfeature_339', 'CC_deepfeature_340', 'CC_deepfeature_343', 'CC_deepfeature_347', 'CC_deepfeature_349', 'CC_deepfeature_350', 'CC_deepfeature_351', 'CC_deepfeature_352', 'CC_deepfeature_357', 'CC_deepfeature_360', 'CC_deepfeature_361', 'CC_deepfeature_362', 'CC_deepfeature_363', 'CC_deepfeature_364', 'CC_deepfeature_366', 'CC_deepfeature_368', 'CC_deepfeature_369', 'CC_deepfeature_370', 'CC_deepfeature_373', 'CC_deepfeature_375', 'CC_deepfeature_376', 'CC_deepfeature_377', 'CC_deepfeature_378', 'CC_deepfeature_381', 'CC_deepfeature_382', 'CC_deepfeature_384', 'CC_deepfeature_385', 'CC_deepfeature_388', 'CC_deepfeature_390', 'CC_deepfeature_392', 'CC_deepfeature_393', 'CC_deepfeature_394', 'CC_deepfeature_395', 'CC_deepfeature_396', 'CC_deepfeature_397', 'CC_deepfeature_398', 'CC_deepfeature_399', 'CC_deepfeature_402', 'CC_deepfeature_403', 'CC_deepfeature_405', 'CC_deepfeature_407', 'CC_deepfeature_409', 'CC_deepfeature_410', 'CC_deepfeature_411', 'CC_deepfeature_413', 'CC_deepfeature_415', 'CC_deepfeature_417', 'CC_deepfeature_418', 'CC_deepfeature_422', 'CC_deepfeature_423', 'CC_deepfeature_424', 'CC_deepfeature_426', 'CC_deepfeature_429', 'CC_deepfeature_430', 'CC_deepfeature_431', 'CC_deepfeature_432', 'CC_deepfeature_434', 'CC_deepfeature_435', 'CC_deepfeature_436', 'CC_deepfeature_437', 'CC_deepfeature_438', 'CC_deepfeature_441', 'CC_deepfeature_442', 'CC_deepfeature_443', 'CC_deepfeature_447', 'CC_deepfeature_450', 'CC_deepfeature_451', 'CC_deepfeature_452', 'CC_deepfeature_453', 'CC_deepfeature_455', 'CC_deepfeature_456', 'CC_deepfeature_458', 'CC_deepfeature_459', 'CC_deepfeature_460', 'CC_deepfeature_461', 'CC_deepfeature_462', 'CC_deepfeature_465', 'CC_deepfeature_466', 'CC_deepfeature_467', 'CC_deepfeature_472', 'CC_deepfeature_475', 'CC_deepfeature_476', 'CC_deepfeature_480', 'CC_deepfeature_482', 'CC_deepfeature_485', 'CC_deepfeature_487', 'CC_deepfeature_488', 'CC_deepfeature_489', 'CC_deepfeature_490', 'CC_deepfeature_491', 'CC_deepfeature_492', 'CC_deepfeature_493', 'CC_deepfeature_496', 'CC_deepfeature_497', 'CC_deepfeature_498', 'CC_deepfeature_501', 'CC_deepfeature_502', 'CC_deepfeature_504', 'CC_deepfeature_505', 'CC_deepfeature_506'
                         , 'square_glcm_Imc2',
                         'wavelet-LLL_glcm_MCC',
                         'wavelet-LLL_firstorder_Skewness',

                         'wavelet-LLH_glszm_GrayLevelNonUniformityNormalized',

                         'original_glcm_Imc2',
                         'wavelet-LLH_glszm_HighGrayLevelZoneEmphasis']
# 'original_shape_Maximum2DDiameterRow',
#                          'wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis',

# 特征个数
#  51:791 751 2:80 78 3:80 79 7:81 78 12:81 78 21:80 79 26:80 80 29:81 80 34: 83 76 35:83 82 39:82 79 43:83 80 68:81 81 74:81 80 92:82 80
# trian, test = train_test_split(df, test_size=0.3, random_state=27)
trian = pd.read_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\combinetrain56pos.csv')
test = pd.read_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\combinetest56pos.csv')
# trian.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\combinetrain56pos.csv')  # 提取的特征写入excel中，提取的位置是什么
# test.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\combinetest56pos.csv')
train_Xshap = trian[certain_feature_names]
test_Xshap = test[certain_feature_names]
# RFC:20
# SVM:
feature = list(df.columns[24:2131])
# 1619
train_names = trian['names'].tolist()
train_X = trian[certain_feature_names]
train_X = train_X.values
train_Y = trian['label']
train_Y = train_Y.values
test_X = test[certain_feature_names]
test_Y = test['label']

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.fit_transform(test_X)
# 创建SVM分类器
# model = svm.SVC()


model = VotingClassifier(estimators=[('rf', rnd_clf), ('svc', svm_clf), ('knn', knn_clf), ('ada', ada_clf), ('lda', lda_clf), ('gbm', gbm_clf), ('lr1', lrl1_clf), ('lr2', lrl2_clf), ('lgbm', lgbm_clf)],

                                     # ('svc', svm_clf), ('knn', knn_clf), ('ada', ada_clf), ('lda', lda_clf), ('gbm', gbm_clf), ('lr1', lrl1_clf), ('lr2', lrl2_clf)],#estimators:子分类器
                              voting='soft')  #参数voting代表你的投票方式，hard,soft
# model = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, oob_score=True, class_weight='balanced', random_state=16)
# model = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, oob_score=True, class_weight='balanced', random_state=10)
# 用训练集做训练
i = 0
j = 0
k = 0
l = 0
m = 0
o = 0
n = 0
tprs = []
fprs = []
mean_fpr = np.linspace(0, 1, 100)
list1 = []
for train_index, test_index in kfold.split(train_X, train_Y):
    print('--------------')
    print(train_index)
    print(np.array(train_names)[train_index])
    print('----------------------------------')
    # print(test_index)
    # print(np.array(train_names)[test_index])
    # print('--------------')
    list2 = []
    list3 = []
    nameslist = []
    # print('训练集索引', train_index)
    # print('验证集索引', test_index)
    # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
    this_train_x, this_train_y = train_X[train_index], train_Y[train_index]  # 本组训练集
    this_test_x, this_test_y = train_X[test_index], train_Y[test_index]  # 本组验证集
    # 训练本组的数据，并计算准确率
    # model.fit(this_train_x, this_train_y)
    # prediction = model.predict(this_test_x)

    # score = accuracy_score(this_test_y, prediction)

    for clf in (rnd_clf, svm_clf, knn_clf, ada_clf, lda_clf, gbm_clf, lrl1_clf, lrl2_clf, lgbm_clf, model):
                # , svm_clf, knn_clf, ada_clf, lda_clf, gbm_clf, lrl1_clf, lrl2_clf, model):
        clf.fit(this_train_x, this_train_y)
        prediction = clf.predict(this_test_x)
        list1.extend(prediction)
        # print(clf.__class__, accuracy_score(y_pre, y_test))
        score = accuracy_score(this_test_y, prediction)
        print(clf.__class__.__name__, score)

        d = 0
    for b in prediction:

        if b != this_test_y[d]:
            list2.append(d)
        if b == this_test_y[d]:
            list3.append(d)
        d = d + 1

    model.fit(this_train_x, this_train_y)
    prediction1 = model.predict(this_test_x)
    score1 = accuracy_score(this_test_y, prediction1)
    y_score1 = model.predict_proba(this_test_x)[:, 1]
    print('Acc', score1)
    fpr, tpr, thread = roc_curve(this_test_y, y_score1, pos_label=1)

    print('fpr是', fpr)

    tprs.append(interp(mean_fpr, fpr, tpr))

    print('tpr是', tpr)
    print('mean_fpr是', mean_fpr)

    roc_auc = auc(fpr, tpr)
    print('auc', roc_auc)
    precision = precision_score(this_test_y, prediction1, average='binary')
    print('pre', precision)
    recall = recall_score(this_test_y, prediction1, average='binary')
    print('rec', recall)
    f1 = f1_score(this_test_y, prediction1, average='binary')
    print('f1', f1)
    tn, fp, fn, tp = confusion_matrix(this_test_y, prediction1).ravel()
    specificity = tn / (tn + fp)
    print('spe', specificity)
    sensitivity = tp / (tp + fn)
    print('sen', sensitivity)

    for e in list2:
        index_of_names = test_index[e]
        certain_name = train_names[index_of_names]
        nameslist.append(certain_name)
    print(nameslist)
    for f in list3:
        index_of_names = test_index[f]
        certain_name = train_names[index_of_names]
        nameslist.append(certain_name)
    print(nameslist)

        # print('训练', score)  # 得到预测结果区间[0,1]
    print('____________________________________________________________________________________________')
    i = i + score
    j = j + precision
    k = k + recall
    l = l + f1
    m = m + roc_auc
    n = n + specificity
    o = o + sensitivity
    train_ave_fpr56 = mean_fpr
    train_ave_tpr56 = np.mean(tprs, axis=0)
    print('tpr均值', train_ave_tpr56)

train_ave_acc = i / 10
train_ave_precision = j / 10
train_ave_recall = k / 10
train_ave_f1 = l / 10
train_ave_roc_auc56 = m / 10
train_ave_Sen = o / 10
train_ave_Spe = n / 10
# train_ave_fpr56 = n / 10
# train_ave_tpr56 = o / 10
print('交叉验证的平均acc：', train_ave_acc)
print('交叉验证的平均precision：', train_ave_precision)
print('交叉验证的平均recall：', train_ave_recall)
print('交叉验证的平均f1：', train_ave_f1)
print('交叉验证的平均auc：', train_ave_roc_auc56)
print('交叉验证的平均Sen：', train_ave_Sen)
print('交叉验证的平均Spe：', train_ave_Spe)
# model.fit(train_X, train_Y)
# 用测试集做预测
prediction = model.predict(test_X)
# y_score = model.decision_function(test_X)
y_score = model.predict_proba(test_X)[:, 1]
# print('y_score是：', y_score)
print('测试集的Acc：', metrics.accuracy_score(prediction, test_Y))
precision = precision_score(test_Y, prediction, average='binary')
recall = recall_score(test_Y, prediction, average='binary')
f1 = f1_score(test_Y, prediction, average='binary')
tnt, fpt, fnt, tpt = confusion_matrix(test_Y, prediction).ravel()
specificity = tnt / (tnt + fpt)
sensitivity = tpt / (tpt + fnt)
print('测试集的Precision: {:.2f}%'.format(precision * 100))
print('测试集的Recall: {:.2f}%'.format(recall * 100))
print('测试集的F1: {:.2f}%'.format(f1 * 100))
print('测试集的spe: {:.2f}%'.format(specificity * 100))
print('测试集的sen: {:.2f}%'.format(sensitivity * 100))

C2 = confusion_matrix(test_Y, prediction, labels=[0, 1])
print(C2)
# 绘制混淆矩阵
plt.figure(figsize=(6, 6))
plt.imshow(C2, interpolation='nearest', cmap=plt.cm.Blues)
plt.rcParams.update({'font.size':23})
plt.title('c1')
plt.colorbar()
tick_marks = np.arange(len([0, 1]))
plt.xticks(tick_marks, [0, 1])
plt.yticks(tick_marks, [0, 1])

# 在格子中添加数值
fmt = 'd'
thresh = C2.max() / 2.
for i in range(C2.shape[0]):
    for j in range(C2.shape[1]):
        plt.text(j, i, format(C2[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if C2[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()


test_Y56 = test_Y
y_score56 = y_score

print(prediction.shape)
print(test_Y.shape)
print(type(prediction))
print(type(test_Y))
print('ACCprediction', prediction)
print('ACCtest_Y', test_Y.index)
minor_list = []
for i in range(len(prediction)):
    minor_list.append(prediction[i] - test_Y[i])
zero_number = minor_list.count(0)
len_list = len(minor_list)
one_number = len_list - zero_number
print(zero_number)
print(len_list)
lower, upper = proportion_confint(zero_number, len_list, 0.05)
print('ACClower=%.3f, ACCupper=%.3f' % (lower, upper))

from scipy.stats import norm


def bootstrap_auc_ci(y_true, y_scores, n_bootstraps=2000, alpha=0.05):
    """
    Calculate AUC and its bootstrap confidence interval.

    Parameters:
    - y_true: 真实的标签。
    - y_scores: 预测的概率或分数。
    - n_bootstraps: 自助法重复的次数。
    - alpha: 置信区间的显著性水平，例如0.05对应95%的置信区间。

    Returns:
    - auc: AUC的值。
    - ci_lower: 置信区间的下界。
    - ci_upper: 置信区间的上界。
    """

    # 计算原始AUC值
    auc = roc_auc_score(y_true, y_scores)

    # 初始化一个空数组来存储每次bootstrap的AUC值
    bootstrapped_aucs = []

    # 使用自助法估计AUC的分布
    for _ in range(n_bootstraps):
        # 随机选择样本，允许重复选择（即自助法）
        indices = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        boot_y_true = y_true[indices]
        boot_y_scores = y_scores[indices]

        # 计算bootstrap样本的AUC，并存储到数组中
        boot_auc = roc_auc_score(boot_y_true, boot_y_scores)
        bootstrapped_aucs.append(boot_auc)

    # 计算percentile置信区间
    sorted_aucs = np.sort(bootstrapped_aucs)
    ci_lower = sorted_aucs[int((alpha / 2) * n_bootstraps)]
    ci_upper = sorted_aucs[int((1 - alpha / 2) * n_bootstraps)]

    return auc, ci_lower, ci_upper


auc_1, ci_lower_1, ci_upper_1 = bootstrap_auc_ci(test_Y56, y_score56)
print('AUC=%.3f, AUClower=%.3f, AUCupper=%.3f' % (auc_1, ci_lower_1, ci_upper_1))



# from sklearn.inspection import permutation_importance
# results = permutation_importance(model, train_X, train_Y, n_repeats=30, random_state=42)
#
# # 获取特征重要性并打印
# importance = results.importances_mean
# feature_names = certain_feature_names
# pd.set_option('display.max_rows', None)  # 显示所有行
# pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.width', None)  # 自动调整宽度
# pd.set_option('display.max_colwidth', None)  # 显示每列的完整内容
#
# # 创建一个 DataFrame 来显示特征重要性
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
# importance_df.sort_values(by='Importance', ascending=False, inplace=True)
# top_importance_df = importance_df.head(10)
# # 检查是否存在重要性值为0的数据
# if (top_importance_df['Importance'] == 0).any():
#     # 如果存在0值，所有重要性值加0.01
#     top_importance_df['Importance'] += 0.001
#
# print(top_importance_df)
# feature_names = top_importance_df['Feature']
# importance = top_importance_df['Importance']





# 使用方法：
# y_true = 真实标签数组
# y_scores = 预测概率数组
# auc_value, ci_l, ci_u = bootstrap_auc_ci(y_true, y_scores)

# fpr, tpr, thread = roc_curve(test_Y, y_score, pos_label=1)
# roc_auc = auc(fpr, tpr)
# 绘图
# plt.figure(figsize=(5, 4), dpi=100)
# plt.style.use('seaborn-darkgrid')
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
#
# # Plotting decision regions
# f, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
# for idx,clf,tt in zip(
#     product([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]),
#     [rnd_clf, svm_clf, knn_clf, ada_clf, lda_clf, gbm_clf, lrl1_clf, lrl2_clf, lgbm_clf, model],
#     ["rnd_clf", "svm_clf", "knn_clf", "ada_clf", "lda_clf", " gbm_clf", "lrl1_clf", "lrl2_clf", "lgbm_clf", "model"],
# ):
#     print(prediction)
#     DecisionBoundaryDisplay.from_estimator(
#         clf, prediction, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict"
#     )
#     axarr[idx[0], idx[1]].scatter(prediction[:, 0], prediction[:, 1], c=test_Y, s=20, edgecolor="k")
#     axarr[idx[0], idx[1]].set_title(tt)
#
# plt.show()


# import shap
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 假设您有特征数据 X 和标签 y
# # X = ...  # 特征数据
# # y = ...  # 标签数据
# certain_feature_names1 = pd.Series(certain_feature_names)
# # certain_feature_names1 = np.array(certain_feature_names)
# # 使用 KernelExplainer 计算 SHAP 值
# explainer = shap.KernelExplainer(model.predict_proba, train_Xshap)
#
# # 计算测试集的 SHAP 值
# shap_values = explainer.shap_values(test_Xshap)
# np.save(r'C:\Users\69559\Desktop\56_shap_val.npy', shap_values, allow_pickle=True, fix_imports=True)
#
#
# if isinstance(shap_values, list):
#     importance = np.mean([np.abs(shap_val).mean(axis=0) for shap_val in shap_values], axis=0)
# else:
#     importance = np.abs(shap_values).mean(axis=0)
#
#
# # # 获取特征的重要性排序
# # sorted_indices = np.argsort(importance)[::-1]  # 从大到小排序
# # # importance = np.abs(shap_values).mean(axis=0)
# # selected_indices = sorted_indices[:20]
# # # 将二维的 importance 转换为一维
# #
# # sorted_feature_names = certain_feature_names1[selected_indices]
# # sorted_importance_values = importance[selected_indices]
# # selet_list = list(sorted_feature_names)
# #
# # shapx = test_Xshap[sorted_feature_names]
# # shap_val_1 = shap_values[1]
# # shap_val_1dsortdf = pd.DataFrame(shap_val_1)
# # shap_valfinal_df = shap_val_1dsortdf[selected_indices]  #1
# # print('shap_val_1dsortdf',shap_val_1dsortdf)
# # print('selected_indices',selected_indices)
# # X_input = test_Xshap[sorted_feature_names]  #2
# # sorted_feature_names_list = list(sorted_feature_names)
# # shap.summary_plot(shap_valfinal_df.values, X_input.values, feature_names=sorted_feature_names_list, show=False)
# shap.summary_plot(shap_values, test_Xshap.values, feature_names=certain_feature_names1, max_display=20, plot_type="bar", show=False)
# # shap.summary_plot(shap_values, test_Xshap.values, feature_names=certain_feature_names1, max_display=20, plot_type="bar", show=True)
# save_path = r'C:\Users\69559\Desktop\shap\56bar.png'  # 自定义路径
# plt.savefig(save_path, dpi=800, bbox_inches='tight')  # 保存为 PDF 文件
# plt.close()  # 关闭当前图形
import shap
import numpy as np
import matplotlib.pyplot as plt


# 假设您有特征数据 X 和标签 y
# X = ...  # 特征数据
# y = ...  # 标签数据
certain_feature_names1 = pd.Series(certain_feature_names)
# certain_feature_names1 = np.array(certain_feature_names)
# 使用 KernelExplainer 计算 SHAP 值
explainer = shap.KernelExplainer(model.predict_proba, train_Xshap)

# 计算测试集的 SHAP 值
shap_values = explainer.shap_values(test_Xshap)
np.save(r'C:\Users\69559\Desktop\deepfeatureNonormalize\56_shap_val.npy', shap_values, allow_pickle=True, fix_imports=True)

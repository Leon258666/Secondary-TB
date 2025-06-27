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
from catboost import CatBoostClassifier

# svm/随机森林、决策树、knn、lda、adaboost、lr、lgbm、coxboost、rsf、glmboost、gbm、glmnet

rnd_clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, min_samples_split=3, oob_score=True, class_weight='balanced', random_state=42)
svm_clf = SVC(C=1.5, decision_function_shape='ovr', probability=True, random_state=16)
knn_clf = KNeighborsClassifier(n_neighbors=4, p=2, weights='distance')
ada_clf = AdaBoostClassifier(n_estimators=20, base_estimator=rnd_clf, learning_rate=0.01, random_state=16)
lda_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
# solver=svd/lsqr/eigen
gbm_clf = GradientBoostingClassifier(max_depth=3, learning_rate=0.01, n_estimators=200, min_samples_split=3, subsample=0.8, random_state=16)

lrl1_clf = LogisticRegression(penalty='l1', C=1, solver="liblinear", random_state=20)
lrl2_clf = LogisticRegression(penalty='l2', C=1, solver="liblinear", random_state=20)
lgbm_clf = lightgbm.LGBMClassifier(objective='binary', max_depth=3, num_leaves=30, learning_rate=0.05, n_estimators=1000, min_child_samples=8,
                                   subsample=0.8, colsample_bytree=1, reg_alpha=1, reg_lambda=1, random_state=20)
cat_clf = CatBoostClassifier(iterations=500, depth=4, learning_rate=0.1, loss_function='MultiClass', logging_level='Silent')


# for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
#     clf.fit(X_train, y_train)
#     y_pre = clf.predict(X_test)
#     print(clf.__class__, accuracy_score(y_pre, y_test))




df = pd.read_csv('C:/Users/Lenovo/Desktop/out56/63positive.csv')

certain_feature_names = ['original_glcm_ClusterShade',
                         'logarithm_glcm_Imc1',
                         'squareroot_glcm_Imc1',
                         'wavelet-LLL_glcm_ClusterShade',
                         'wavelet-LLL_ngtdm_Strength',
                         'original_glcm_Imc2',
                         'original_glcm_MCC',
                         'original_glrlm_LongRunLowGrayLevelEmphasis',
                         'lbp-3D-m2_glszm_SmallAreaHighGrayLevelEmphasis',
                         'lbp-3D-k_glszm_LargeAreaEmphasis',
                         'lbp-3D-k_glszm_LargeAreaHighGrayLevelEmphasis',
                         'logarithm_glcm_Correlation',
                         'square_glcm_MCC',
                         'squareroot_glcm_ClusterShade',
                         'squareroot_glcm_Imc2',
                         'wavelet-LLH_glrlm_ShortRunHighGrayLevelEmphasis',
                         'wavelet-LLH_glszm_ZoneVariance',
                         'wavelet-HLH_glszm_LargeAreaHighGrayLevelEmphasis']



# certain_feature_names = ['logarithm_glcm_Imc1',
#                          'wavelet-LLH_glrlm_ShortRunHighGrayLevelEmphasis',
#                          'wavelet-HHH_glrlm_LongRunHighGrayLevelEmphasis',
#                          'wavelet-LHH_gldm_DependenceNonUniformityNormalized',
#                          'square_ngtdm_Busyness']









trian, test = train_test_split(df, test_size=0.3, random_state=1)

# RFC:20
# SVM:
feature = list(df.columns[24:1619])
# 1619
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
model = VotingClassifier(estimators=[('rf', rnd_clf), ('svc', svm_clf), ('knn', knn_clf), ('ada', ada_clf), ('lda', lda_clf), ('gbm', gbm_clf), ('lr1', lrl1_clf), ('lr2', lrl2_clf), ('lgbm', lgbm_clf), ('cat', cat_clf)],

                                     # ('svc', svm_clf), ('knn', knn_clf), ('ada', ada_clf), ('lda', lda_clf), ('gbm', gbm_clf), ('lr1', lrl1_clf), ('lr2', lrl2_clf)],#estimators:子分类器
                              voting='soft')  #参数voting代表你的投票方式，hard,soft
# model = RandomForestClassifier(n_estimators=200, min_samples_leaf=1, oob_score=True, class_weight='balanced', random_state=16)
# model = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, oob_score=True, class_weight='balanced', random_state=10)
# 用训练集做训练
i = 0


for train_index, test_index in kfold.split(train_X, train_Y):
    # print('训练集索引', train_index)
    # print('验证集索引', test_index)
    # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
    this_train_x, this_train_y = train_X[train_index], train_Y[train_index]  # 本组训练集
    this_test_x, this_test_y = train_X[test_index], train_Y[test_index]  # 本组验证集
    # 训练本组的数据，并计算准确率
    # model.fit(this_train_x, this_train_y)
    # prediction = model.predict(this_test_x)

    # score = accuracy_score(this_test_y, prediction)

    for clf in (rnd_clf, svm_clf, knn_clf, lda_clf, gbm_clf, ada_clf, lrl1_clf, lrl2_clf, lgbm_clf, cat_clf ,model):
                # , svm_clf, knn_clf, ada_clf, lda_clf, gbm_clf, lrl1_clf, lrl2_clf, model):
        clf.fit(this_train_x, this_train_y)
        prediction = clf.predict(this_test_x)
        # print(clf.__class__, accuracy_score(y_pre, y_test))
        score = accuracy_score(this_test_y, prediction)
        print(clf.__class__.__name__, score)
        # print('训练', score)  # 得到预测结果区间[0,1]
    print('____________________________________________________________________________________________')
    i = i + score
train_ave = i / 10

print('交叉验证的平均准确率：', train_ave)
# model.fit(train_X, train_Y)
# 用测试集做预测
prediction = model.predict(test_X)
# y_score = model.decision_function(test_X)
y_score = model.predict_proba(test_X)[:, 1]
# print('y_score是：', y_score)
print('测试集的准确率：', metrics.accuracy_score(test_Y, prediction))

C2 = confusion_matrix(test_Y, prediction, labels=[0, 1])
print(C2)


fpr, tpr, thread = roc_curve(test_Y, y_score, pos_label=1)
roc_auc = auc(fpr, tpr)
# 绘图
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

from skimage import transform,exposure
from sklearn import model_selection, preprocessing, metrics, feature_selection
import os
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score


#加载npy数据和label
def load_npy_data(data_dir,split):
    data1np=[]                               #images
    # data2np=[]                               #images
    truenp=[]
    # A=os.path.join(data_dir,'128')#labels
    A = data_dir
    # B=os.path.join(data_dir,'128')
    for file1 in os.listdir(A):
        print(file1)
        data1=np.load(os.path.join(A,file1),allow_pickle=True)
        if(split =='train'):
            data_sug11= transform.rotate(data1[0][0], 60) #旋转60度，不改变大小
            # data_sug12= transform.rotate(data2[0][0], 60) #旋转60度，不改变大小
            data_sug21 = exposure.exposure.adjust_gamma(data1[0][0], gamma=0.5)#变亮
            # data_sug22 = exposure.exposure.adjust_gamma(data2[0][0], gamma=0.5)
            data1np.append(data_sug11)
            truenp.append(data1[0][1])
            data1np.append(data_sug21)
            truenp.append(data1[0][1])
            # data2np.append(data_sug12)
            # data2np.append(data_sug22)
        data1np.append(data1[0][0])
        # data2np.append(data2[0][0])
        truenp.append(data1[0][1])
    data1np = np.array(data1np)
    print(data1np.shape)
    #numpy.array可使用 shape。list不能使用shape。可以使用np.array(list A)进行转换。
    #不能随意加维度
    data1np=np.expand_dims(data1np,axis=3)  #加维度,from(1256,256,128)to(256,256,128,1),according the cnn tabel.png

    data1np = data1np.transpose(0,3,1,2)


    truenp = np.array(truenp)
    # print(data1np.shape,data2np.shape, truenp.shape)
    # print(np.min(data1np), np.max(data1np), np.mean(data1np), np.median(data1np))
    # print(data1np.shape,data2np.shape, truenp.shape)
    # print(np.min(data1np), np.max(data1np), np.mean(data1np), np.median(data1np))
    # print(np.min(data2np), np.max(data2np), np.mean(data2np), np.median(data2np))
    return data1np,truenp

def load_npy_data_for_cam(data_dir,split):
    data1np=[]                               #images
    data2np=[]                               #images
    truenp=[]
    image_names=[]
    A=os.path.join(data_dir,'64')#labels
    B=os.path.join(data_dir,'128')
    for file1 in os.listdir(A):
        data1=np.load(os.path.join(A,file1),allow_pickle=True)
        if file1 in os.listdir(B):
            image_names.append(file1)
            data2=np.load(os.path.join(B,file1),allow_pickle=True)
            if(split =='train'):
                data_sug11= transform.rotate(data1[0][0], 60) #旋转60度，不改变大小
                data_sug12= transform.rotate(data2[0][0], 60) #旋转60度，不改变大小
                data_sug21 = exposure.exposure.adjust_gamma(data1[0][0], gamma=0.5)#变亮
                data_sug22 = exposure.exposure.adjust_gamma(data2[0][0], gamma=0.5)
                data1np.append(data_sug11)
                truenp.append(data1[0][1])
                data1np.append(data_sug21)
                truenp.append(data1[0][1])
                data2np.append(data_sug12)
                data2np.append(data_sug22)
            data1np.append(data1[0][0])
            data2np.append(data2[0][0])
            truenp.append(data1[0][1])
    data1np = np.array(data1np)

    data2np = np.array(data2np)
    #numpy.array可使用 shape。list不能使用shape。可以使用np.array(list A)进行转换。
    #不能随意加维度
    data1np=np.expand_dims(data1np,axis=4)  #加维度,from(1256,256,128)to(256,256,128,1),according the cnn tabel.png
    data2np=np.expand_dims(data2np,axis=4)
    data1np = data1np.transpose(0,4,1,2,3)
    data2np = data2np.transpose(0,4,1,2,3)

    truenp = np.array(truenp)
    print(data1np.shape,data2np.shape, truenp.shape)
    # print(np.min(data1np), np.max(data1np), np.mean(data1np), np.median(data1np))
    # print(data1np.shape,data2np.shape, truenp.shape)
    # print(np.min(data1np), np.max(data1np), np.mean(data1np), np.median(data1np))
    # print(np.min(data2np), np.max(data2np), np.mean(data2np), np.median(data2np))
    return data1np,data2np,truenp,image_names

#定义随机种子
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def _init_fn(worker_id):
    np.random.seed(int(12) + worker_id)

#计算分类的各项指标
# from sklearn import metrics
# import numpy as np

# import numpy as np
# from sklearn import metrics
#

def calculate_multiclass(score, label, n_classes=5):
    #correct!!!    获取最高分数的索引作为预测类别
    score = np.array(score)
    label = np.array(label)
    pred = np.argmax(score, axis=1)

    # correct!!!   计算混淆矩阵
    confusion_matrix1 = metrics.confusion_matrix(label, pred)

    # 从混淆矩阵中提取TP, FP, FN
    TP = np.diag(confusion_matrix1)
    FP = confusion_matrix1.sum(axis=0) - TP
    FN = confusion_matrix1.sum(axis=1) - TP

    # 计算TN
    TN = np.zeros_like(TP)
    for i in range(len(TP)):
        TN[i] = confusion_matrix1.sum() - (FP[i] + FN[i] + TP[i])

    # 计算每类的指标，然后平均
    TPR = TP / (TP + FN)  # 真正例率，也称为敏感性
    TNR = TN / (TN + FP)  # 真负例率，也称为特异性

    # 计算分类精度
    # class_accuracy = TP / (TP + FP + FN)
    class_accuracy = confusion_matrix1.diagonal() / confusion_matrix1.sum(axis=1)

    # 计算加权平均（宏平均）的指标
    weights = (TP + FN) / confusion_matrix1.sum()
    macro_TPR = np.average(TPR, weights=weights)  # 加权敏感性
    macro_TNR = np.average(TNR, weights=weights)  # 加权特异性

    # 计算每一类的spe与sen需要改n_classes
    spe0 = spe1 = spe2 = 0
    sen0 = sen1 = sen2 = 0
    for i in range(n_classes):
        conf_matrix = metrics.confusion_matrix(label, pred, labels=[i, 1-i])
        tn, fp, fn, tp = conf_matrix.ravel()
        if i == 0:
            spe0 = tn / (tn + fp)
            sen0 = tp / (tp + fn)
        elif i == 1:
            spe1 = tn / (tn + fp)
            sen1 = tp / (tp + fn)
        elif i == 2:
            spe2 = tn / (tn + fp)
            sen2 = tp / (tp + fn)
        elif i == 3:
            spe3 = tn / (tn + fp)
            sen3 = tp / (tp + fn)
        elif i == 4:
            spe4 = tn / (tn + fp)
            sen4 = tp / (tp + fn)



    # 计算精确度
    # accuracy = (TP + TN).sum() / (confusion_matrix.sum() * len(TP))
    accuracy = accuracy_score(label, pred)
    # correct!!!  计算多类AUC，确保score是每个类别的概率得分
    AUC = metrics.roc_auc_score(label, score, multi_class='ovr')

    # 执行检查以确保所有计算都正确
    assert np.all(confusion_matrix1.sum(axis=1) == TP + FN), "Row sums do not match TP + FN!"
    assert np.all(confusion_matrix1.sum(axis=0) == TP + FP), "Column sums do not match TP + FP!"

    assert 0 <= accuracy <= 1, "Accuracy is out of bounds!"

    result = {
        'AUC': AUC,
        'acc': accuracy,
        'class_acc': class_accuracy,  # 分类精度
        'sen': macro_TPR,  # 宏平均敏感性
        'spe': macro_TNR,  # 宏平均特异性
        'pred': pred  ,# 预测的类别
        'class_0_sen': sen0,
        'class_0_spe': spe0,
        'class_1_sen': sen1,
        'class_1_spe': spe1,
        'class_2_sen': sen2,
        'class_2_spe': spe2,
        'class_3_sen': sen3,
        'class_3_spe': spe3,
        'class_4_sen': sen4,
        'class_4_spe': spe4
    }

    # 输出每个类别的精度值
    for i in range(len(class_accuracy)):
        print("Class", i, "accuracy:", class_accuracy[i])

    print(result)
    return result




# def calculate(score, label, th):
#     score = np.array(score)
#     label = np.array(label)
#     pred = np.zeros_like(label)
#     pred[score >= th] = 1
#     pred[score < th] = 0
#
#     TP = len(pred[(pred > 0.5) & (label > 0.5)])
#     FN = len(pred[(pred < 0.5) & (label > 0.5)])
#     TN = len(pred[(pred < 0.5) & (label < 0.5)])
#     FP = len(pred[(pred > 0.5) & (label < 0.5)])
#
#     AUC = metrics.roc_auc_score(label, score)
#     result = {'AUC': AUC, 'acc': (TP + TN) / (TP + TN + FP + FN), 'sen': (TP) / (TP + FN + 0.0001),
#               'spe': (TN) / (TN + FP + 0.0001),'pred': pred}
#     #     print('acc',(TP+TN),(TP+TN+FP+FN),'spe',(TN),(TN+FP),'sen',(TP),(TP+FN))
#     return result

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


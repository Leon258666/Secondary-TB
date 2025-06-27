import os
import sys
import json
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import pandas as pd
import collections
import SimpleITK as sitk
from PIL import Image
from torchvision import models
import torch.nn.functional as F
from modelres import resnet34



def main():
    # patient_list = pd.read_excel(io=r'D:\breastcanaerdata\patient_list.xlsx',sheet_name = 0)
    # patient_fenxing_list = np.array(patient_list.iloc[:, 24:25].T)[0]
    # patient_number_list = np.array(patient_list.iloc[:, 0:1].T)[0]
    # patient_patient_number_fenxing_list = dict(zip(patient_number_list, patient_fenxing_list))


    path_56 = r'D:\exp\Temporary\56select3-after Trans'
    path_63 = r'D:\exp\Temporary\63select3-after Trans'
    path_80 = r'D:\exp\Temporary\80select3-after Trans'
    path_91 = r'D:\exp\Temporary\91select3-after Trans'
    path_69 = r'D:\exp\Temporary\69select3-after Trans'
    device = torch.device("cpu")
    print("using {} device.".format(device))

    # net = torch.load(r'D:\pycharmproject\2st study\Train_Custom_Dataset-main\pre-mammogram\checkpoints\best-0.402.pth')
    # net = models.resnet50(pretrained=True) # 载入预训练模型
    # net = timm.create_model('resnet50', num_classes=0# num_classes=0表示没有分类层
    # ,pretrained_cfg_overlay=dict(file=r'C:\Users\69559\Desktop\deepfeatures\model.safetensors')
    #                         )
    net = resnet34(num_classes=0)
    net.fc = torch.nn.Identity()
    model_weight_path = r'C:\Users\69559\Desktop\bjxktuberdeep34\t1_sag-fold0.pth'
    pretrained_model = torch.load(model_weight_path)
    model_state_dict = {k: v for k, v in pretrained_model.items() if 'fc' not in k}
    net.load_state_dict(model_state_dict, strict=False)

    # 'resnet50' 'vit_base_patch16_224_in21k' 'tf_efficientnetv2_s_in21ft1k' 'visformer_small'
    print(net)
    net.to(device)

    def readNII(path):
        # 读取医疗文件，转换成numpy
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data

    def getImg(data, seg):
        img = np.argwhere(seg[0])
        xmin, ymin = np.min(img, axis=0)
        xmax, ymax = np.max(img, axis=0)
        if Patient_names[i] == '510':
            img = data[0][xmin:xmax-30, ymin+450:ymax-50].astype(np.float32) #510MLO
        elif Patient_names[i] == '271':
            img = data[0][xmin:xmax, ymin+100:ymax].astype(np.float32)  # 271MLO
        elif Patient_names[i] == '195':
            img = data[0][xmin:xmax, ymin+100:ymax].astype(np.float32) #195
        elif Patient_names[i] == '568':
            img = data[0][xmin+80:xmax, ymin:ymax].astype(np.float32) #568
        elif Patient_names[i] == '579':
            img = data[0][xmin:xmax, ymin:ymax-110].astype(np.float32) #579
        elif Patient_names[i] == '472':
            img = data[0][xmin:xmax, ymin:ymax-40].astype(np.float32) #472
        else:
            img = data[0][xmin:xmax, ymin:ymax].astype(np.float32)
        return img

    def porcessInput(img, size=(224, 224)):
        # 输入网络前，对图片进行预处理

        img = Image.fromarray(img)
        img = img.resize(size=size, resample=Image.NEAREST)

        img = np.array(img)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        # print(np.shape(img))
        img = np.expand_dims(img, 0)
        # print(np.shape(img))
        # print(img)
        return img

    def getFeatures(model, data):
        # 提取模型的最后一层卷积，生成features
        featuresDict = collections.OrderedDict()
        # data = np.concatenate((data, data, data), axis=0) #通道数  1→3
        data = np.repeat(data, 3, axis=0)  #通道数  1→3
        print(np.shape(data))
        data = torch.from_numpy(data)
        mean = data.mean(dim=(1, 2))
        std = data.std(dim=(1, 2))
        data = transforms.Normalize(mean, std)(data)
        # data = transforms.Normalize([0.5382666, 0.5382666, 0.5382666], [0.21533948, 0.21533948, 0.21533948])(data)
        data = data[0].unsqueeze(0)
        model.eval()
        # new_model = torch.nn.Sequential(*(list(model.children())[:-1]))  # 去掉模型最后一层
        # new_model = torch.nn.Sequential(*(list(model.children())[:-1]),nn.Flatten())  # 去掉模型最后一层
        # new_model = torch.nn.Sequential(net.features, net.avgpool, net.classifier[:-1])
        # new_model = torch.nn.Sequential(net.features, net.avgpool)
        # new_model = net.features
        # features = new_model(data.to(device))[0][:,0][:,0].detach().numpy()
        with torch.no_grad():
            features = model(data.to(device))[0].detach().numpy()
        # features = model(data.to(device))[0].detach().numpy()



        # features = F.softmax(features,dim=1)[0].detach().numpy()
        # features = new_model(data.to(device))
        # features = np.max(features, -1)
        # print(features)
        # features = np.max(features, -1)
        # print(features)
        for i, value in enumerate(features):
            featuresDict[ 'CC_deepfeature_' + str(i)] = value
        return featuresDict


    results56 = pd.DataFrame()
    Patient_names = os.listdir(path_56)
    for i in range(len(Patient_names)):
        print(Patient_names[i])
        Hua_directory = os.path.join(path_56,Patient_names[i],'CTHUA')
        New_directory = os.path.join(path_56,Patient_names[i],'CTNEW')
        image_paths = os.listdir(New_directory)
        for img_name in image_paths:
            Patient_path = os.path.join(New_directory, img_name)
            mask_path = os.path.join(Hua_directory, img_name)
            featuresDict = collections.OrderedDict()
            # featuresDict['id'] = Patient_names+img_name
            # featuresDict['label'] = patient_patient_number_fenxing_list[int(Dic_names[i])]
            # featuresDict['label'] = 1
            origin = readNII(Patient_path)
            mask = readNII(mask_path)
            # img = getImg(origin, mask)
            # 有getmig就不用这两句
            img = origin.astype(np.float32)
            mask = mask.astype(np.float32)

            # resnet注释这两句，因为vis要224
            # img = np.squeeze(img)
            # img = porcessInput(img)
            features = getFeatures(net, img)
            aFeature = pd.Series(features)  # Series 是一维数组，基于Numpy的ndarray 结构,给值添加索引
            aFeature = aFeature.to_frame()  # 它返回Series的DataFrame表示形式,最上边空了一行索引
            aFeature.columns = [img_name]
            results56 = results56.append(aFeature.T)

    results63 = pd.DataFrame()
    Patient_names = os.listdir(path_63)
    for i in range(len(Patient_names)):
        print(Patient_names[i])
        # Hua_directory = os.path.join(path,Patient_names[i],'CTHUA')
        New_directory = os.path.join(path_63,Patient_names[i],'CTNEW')
        image_paths = os.listdir(New_directory)
        for img_name in image_paths:
            Patient_path = os.path.join(New_directory, img_name)
            # mask_path = os.path.join(Hua_directory, img_name)
            featuresDict = collections.OrderedDict()
            # featuresDict['id'] = Patient_names+img_name
            # featuresDict['label'] = patient_patient_number_fenxing_list[int(Dic_names[i])]
            # featuresDict['label'] = 1
            origin = readNII(Patient_path)
            # mask = readNII(mask_path)
            # img = getImg(origin, mask)
            img = origin.astype(np.float32)
            # resnet注释这两句，因为vis要224
            # img = np.squeeze(img)
            # img = porcessInput(img)
            features = getFeatures(net, img)
            aFeature = pd.Series(features)  # Series 是一维数组，基于Numpy的ndarray 结构,给值添加索引
            aFeature = aFeature.to_frame()  # 它返回Series的DataFrame表示形式,最上边空了一行索引
            aFeature.columns = [img_name]
            results63 = results63.append(aFeature.T)

    results80 = pd.DataFrame()
    Patient_names = os.listdir(path_80)
    for i in range(len(Patient_names)):
        print(Patient_names[i])
        # Hua_directory = os.path.join(path,Patient_names[i],'CTHUA')
        New_directory = os.path.join(path_80,Patient_names[i],'CTNEW')
        image_paths = os.listdir(New_directory)
        for img_name in image_paths:
            Patient_path = os.path.join(New_directory, img_name)
            # mask_path = os.path.join(Hua_directory, img_name)
            featuresDict = collections.OrderedDict()
            # featuresDict['id'] = Patient_names+img_name
            # featuresDict['label'] = patient_patient_number_fenxing_list[int(Dic_names[i])]
            # featuresDict['label'] = 1
            origin = readNII(Patient_path)
            # mask = readNII(mask_path)
            # img = getImg(origin, mask)
            img = origin.astype(np.float32)
            # resnet注释这两句，因为vis要224
            # img = np.squeeze(img)
            # img = porcessInput(img)
            features = getFeatures(net, img)
            aFeature = pd.Series(features)  # Series 是一维数组，基于Numpy的ndarray 结构,给值添加索引
            aFeature = aFeature.to_frame()  # 它返回Series的DataFrame表示形式,最上边空了一行索引
            aFeature.columns = [img_name]
            results80 = results80.append(aFeature.T)

    results91 = pd.DataFrame()
    Patient_names = os.listdir(path_91)
    for i in range(len(Patient_names)):
        print(Patient_names[i])
        # Hua_directory = os.path.join(path,Patient_names[i],'CTHUA')
        New_directory = os.path.join(path_91,Patient_names[i],'CTNEW')
        image_paths = os.listdir(New_directory)
        for img_name in image_paths:
            Patient_path = os.path.join(New_directory, img_name)
            # mask_path = os.path.join(Hua_directory, img_name)
            featuresDict = collections.OrderedDict()
            # featuresDict['id'] = Patient_names+img_name
            # featuresDict['label'] = patient_patient_number_fenxing_list[int(Dic_names[i])]
            # featuresDict['label'] = 1
            origin = readNII(Patient_path)
            # mask = readNII(mask_path)
            # img = getImg(origin, mask)
            img = origin.astype(np.float32)
            # resnet注释这两句，因为vis要224
            # img = np.squeeze(img)
            # img = porcessInput(img)
            features = getFeatures(net, img)
            aFeature = pd.Series(features)  # Series 是一维数组，基于Numpy的ndarray 结构,给值添加索引
            aFeature = aFeature.to_frame()  # 它返回Series的DataFrame表示形式,最上边空了一行索引
            aFeature.columns = [img_name]
            results91 = results91.append(aFeature.T)

    results69 = pd.DataFrame()
    Patient_names = os.listdir(path_69)
    for i in range(len(Patient_names)):
        print(Patient_names[i])
        # Hua_directory = os.path.join(path,Patient_names[i],'CTHUA')
        New_directory = os.path.join(path_69,Patient_names[i],'CTNEW')
        image_paths = os.listdir(New_directory)
        for img_name in image_paths:
            Patient_path = os.path.join(New_directory, img_name)
            # mask_path = os.path.join(Hua_directory, img_name)
            featuresDict = collections.OrderedDict()
            # featuresDict['id'] = Patient_names+img_name
            # featuresDict['label'] = patient_patient_number_fenxing_list[int(Dic_names[i])]
            # featuresDict['label'] = 1
            origin = readNII(Patient_path)
            # mask = readNII(mask_path)
            # img = getImg(origin, mask)
            img = origin.astype(np.float32)
            # resnet注释这两句，因为vis要224
            # img = np.squeeze(img)
            # img = porcessInput(img)
            features = getFeatures(net, img)
            aFeature = pd.Series(features)  # Series 是一维数组，基于Numpy的ndarray 结构,给值添加索引
            aFeature = aFeature.to_frame()  # 它返回Series的DataFrame表示形式,最上边空了一行索引
            aFeature.columns = [img_name]
            results69 = results69.append(aFeature.T)

    results56.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\Nomask56.csv')  # 提取的特征写入excel中，提取的位置是什么
    results63.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\Nomask63.csv')  # 提取的特征写入excel中，提取的位置是什么
    results80.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\Nomask80.csv')  # 提取的特征写入excel中，提取的位置是什么
    results91.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\Nomask91.csv')  # 提取的特征写入excel中，提取的位置是什么
    results69.to_csv(r'C:\Users\69559\Desktop\deepfeatureNonormalize\Nomask69.csv')  # 提取的特征写入excel中，提取的位置是什么

if __name__ == '__main__':
    main()

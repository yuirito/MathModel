import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from torchvision import transforms
from parser_my import args

#
def getData(pretrainFile,sequence_length,batchSize):
    data=pd.read_excel(pretrainFile)
    data.drop('time', axis=1, inplace=True)

    #data.drop('pos', axis=1, inplace=True)
    #data.drop('AQI', axis=1, inplace=True)
    #data.drop('MainPStr', axis=1, inplace=True)
    #data.drop('MainP', axis=1, inplace=True)
    #data.drop('DAQI', axis=1, inplace=True)
    #data = data.loc[7:19424]

    label_max=[]
    label_min=[]

    label_max.append(data['SO2'].max())
    label_min.append(data['SO2'].min())
    label_max.append(data['NO2'].max())
    label_min.append(data['NO2'].min())
    label_max.append(data['PM10'].max())
    label_min.append(data['PM10'].min())
    label_max.append(data['PM2.5'].max())
    label_min.append(data['PM2.5'].min())
    label_max.append(data['O3'].max())
    label_min.append(data['O3'].min())
    label_max.append(data['CO'].max())
    label_min.append(data['CO'].min())


    df=data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    #print(df)

    # 构造X和Y

    seq_len = sequence_length
    X = []
    Y = []
    for i in range(0,df.shape[0] - seq_len-24*2,24):
        X.append(np.array(df.iloc[i:(i + seq_len), ].values, dtype=np.float32))
        d1 = np.array(df.iloc[i+seq_len-8:i+seq_len-8+24, 0:6], dtype=np.float32)
        d2 = np.array(df.iloc[16 + i+seq_len:(16 + i + seq_len+24), 0:6], dtype=np.float32)
        d3 = np.array(df.iloc[16 + i + seq_len+24:(16 + i + seq_len+48), 0:6], dtype=np.float32)
        d1=np.mean(d1,axis=0)
        d2=np.mean(d2,axis=0)
        d3=np.mean(d3, axis=0)
        Y.append(np.concatenate((d1,d2,d3)))


    # 构建batch
    total_len = len(Y)
    # print(total_len)

    trainx, trainy = X[:int(0.95 * total_len)], Y[:int(0.95 * total_len)]
    testx, testy = X[int(0.95 * total_len):], Y[int(0.95 * total_len):]
    train_loader = DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=True)
    return label_max,label_min,train_loader,test_loader

def getData_2(data_pre_file,data_act_file,sequence_length,batchSize):
    data_pre=pd.read_csv(data_pre_file)
    data=pd.read_excel(data_act_file)

    data.drop('time', axis=1, inplace=True)
    data_pre.drop('模型运行日期', axis=1, inplace=True)

    data = data.loc[args.first_r:]

    label_max=[]
    label_min=[]

    label_max.append(data['SO2'].max())
    label_min.append(data['SO2'].min())
    label_max.append(data['NO2'].max())
    label_min.append(data['NO2'].min())
    label_max.append(data['PM10'].max())
    label_min.append(data['PM10'].min())
    label_max.append(data['PM2.5'].max())
    label_min.append(data['PM2.5'].min())
    label_max.append(data['O3'].max())
    label_min.append(data['O3'].min())
    label_max.append(data['CO'].max())
    label_min.append(data['CO'].min())


    df=data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    data_pre=data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    seq_len = sequence_length
    X_pre = []
    X_act = []
    Y = []
    for i in range(0, df.shape[0] - seq_len - 24*2, 24):
        j=i//24
        X_act.append(np.array(df.iloc[i:(i + seq_len), ].values, dtype=np.float32))
        X_pre.append(np.concatenate(np.array(data_pre.iloc[j:(j + 3), ].values, dtype=np.float32)))
        d1 = np.array(df.iloc[i + seq_len - 8:i + seq_len - 8 + 24, 0:6], dtype=np.float32)
        d2 = np.array(df.iloc[16 + i + seq_len:(16 + i + seq_len + 24), 0:6], dtype=np.float32)
        d3 = np.array(df.iloc[16 + i + seq_len + 24:(16 + i + seq_len + 48), 0:6], dtype=np.float32)
        d1 = np.mean(d1, axis=0)
        d2 = np.mean(d2, axis=0)
        d3 = np.mean(d3, axis=0)
        Y.append(np.concatenate((d1, d2, d3)))

    # 构建batch
    total_len = len(Y)
    print("len:")
    print(total_len)

    trainx_act,trainx_pre, trainy = X_act[:int(0.80 * total_len)],X_pre[:int(0.80 * total_len)], Y[:int(0.80 * total_len)]
    testx_act,textx_pre, testy = X_act[int(0.80 * total_len):],X_pre[int(0.80 * total_len):], Y[int(0.80 * total_len):]

    train_loader = DataLoader(dataset=Mydataset_2(trainx_act,trainx_pre, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True)
    test_loader = DataLoader(dataset=Mydataset_2(testx_act,textx_pre, testy), batch_size=batchSize, shuffle=True)
    print("len test loader:")
    print(len(test_loader))
    return label_max,label_min,train_loader,test_loader


def getCovData(preFile_A,preFile_A1,preFile_A2,preFile_A3,sequence_length,batchSize):
    data=pd.read_excel(preFile_A)
    data.drop('time', axis=1, inplace=True)

    label_max=[]
    label_min=[]

    label_max.append(data['SO2'].max())
    label_min.append(data['SO2'].min())
    label_max.append(data['NO2'].max())
    label_min.append(data['NO2'].min())
    label_max.append(data['PM10'].max())
    label_min.append(data['PM10'].min())
    label_max.append(data['PM2.5'].max())
    label_min.append(data['PM2.5'].min())
    label_max.append(data['O3'].max())
    label_min.append(data['O3'].min())
    label_max.append(data['CO'].max())
    label_min.append(data['CO'].min())


    df=data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    data1 = pd.read_excel(preFile_A1)
    data1.drop('time', axis=1, inplace=True)
    data2 = pd.read_excel(preFile_A2)
    data2.drop('time', axis=1, inplace=True)
    data3 = pd.read_excel(preFile_A3)
    data3.drop('time', axis=1, inplace=True)

    label1_max = [data1['SO2'].max(),data1['NO2'].max(),data1['PM10'].max(),data1['PM2.5'].max(),data1['O3'].max(),data1['CO'].max()]
    label1_min = [data1['SO2'].min(),data1['NO2'].min(),data1['PM10'].min(),data1['PM2.5'].min(),data1['O3'].min(),data1['CO'].min()]

    label2_max = [data2['SO2'].max(), data2['NO2'].max(), data2['PM10'].max(), data2['PM2.5'].max(), data2['O3'].max(),
                  data2['CO'].max()]
    label2_min = [data2['SO2'].min(), data2['NO2'].min(), data2['PM10'].min(), data2['PM2.5'].min(), data2['O3'].min(),
                  data2['CO'].min()]


    label3_max = [data3['SO2'].max(), data3['NO2'].max(), data3['PM10'].max(), data3['PM2.5'].max(), data3['O3'].max(),
                  data3['CO'].max()]
    label3_min = [data3['SO2'].min(), data3['NO2'].min(), data3['PM10'].min(), data3['PM2.5'].min(), data3['O3'].min(),
                  data3['CO'].min()]

    label_max_list = [label_max,label1_max,label2_max,label3_max]
    label_min_list = [label_min,label1_min,label2_min,label3_min]
    df = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df1 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df2 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df3 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    #print(df)

    # 构造X和Y

    seq_len = sequence_length
    X = []
    Y = [[],[],[],[]]
    for i in range(0,df.shape[0] - seq_len-24*2,24):
        x0=np.array(df.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x1=np.array(df1.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x2 = np.array(df2.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x3 = np.array(df3.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x_list=[x0,x1,x2,x3]
        x_list=np.array(x_list)
        X.append(x_list.transpose(0,2,1))
        j=0
        for d in [df,df1,df2,df3]:
            d1 = np.array(d.iloc[i + seq_len - 8:i + seq_len - 8 + 24, 0:6], dtype=np.float32)
            d2 = np.array(d.iloc[16 + i + seq_len:(16 + i + seq_len + 24), 0:6], dtype=np.float32)
            d3 = np.array(d.iloc[16 + i + seq_len + 24:(16 + i + seq_len + 48), 0:6], dtype=np.float32)
            d1 = np.mean(d1, axis=0)
            d2 = np.mean(d2, axis=0)
            d3 = np.mean(d3, axis=0)
            Y[j].append(np.concatenate((d1, d2, d3)))
            j=j+1


    # 构建batch
    total_len = len(Y[0])
    print(total_len)
    train_loader_list=[]
    test_loader_list=[]
    for i in range(4):
        trainx, trainy = X[:int(0.95 * total_len)], Y[i][ :int(0.95 * total_len)]
        testx, testy = X[int(0.95 * total_len):], Y[i][ :int(0.95 * total_len):]
        train_loader_list.append(DataLoader(dataset=Mydataset(trainx, trainy, transform=transforms.ToTensor()),
                                  batch_size=batchSize,
                                  shuffle=True))
        test_loader_list.append(DataLoader(dataset=Mydataset(testx, testy), batch_size=batchSize, shuffle=True))

    return label_max_list,label_min_list,train_loader_list,test_loader_list

def getCovData_2(preFile_A_1,preFile_A1_1,preFile_A2_1,preFile_A3_1,
                 preFile_A,preFile_A1,preFile_A2,preFile_A3,sequence_length,batchSize):
    data = pd.read_excel(preFile_A)
    data.drop('time', axis=1, inplace=True)
    data = data.loc[6432:]

    label_max = []
    label_min = []

    label_max.append(data['SO2'].max())
    label_min.append(data['SO2'].min())
    label_max.append(data['NO2'].max())
    label_min.append(data['NO2'].min())
    label_max.append(data['PM10'].max())
    label_min.append(data['PM10'].min())
    label_max.append(data['PM2.5'].max())
    label_min.append(data['PM2.5'].min())
    label_max.append(data['O3'].max())
    label_min.append(data['O3'].min())
    label_max.append(data['CO'].max())
    label_min.append(data['CO'].min())

    df = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    data1 = pd.read_excel(preFile_A1)
    data1.drop('time', axis=1, inplace=True)
    data1 = data1.loc[6432:]
    data2 = pd.read_excel(preFile_A2)
    data2.drop('time', axis=1, inplace=True)
    data2 = data2.loc[6432:]
    data3 = pd.read_excel(preFile_A3)
    data3.drop('time', axis=1, inplace=True)
    data3 = data3.loc[6432:]

    label1_max = [data1['SO2'].max(), data1['NO2'].max(), data1['PM10'].max(), data1['PM2.5'].max(), data1['O3'].max(),
                  data1['CO'].max()]
    label1_min = [data1['SO2'].min(), data1['NO2'].min(), data1['PM10'].min(), data1['PM2.5'].min(), data1['O3'].min(),
                  data1['CO'].min()]

    label2_max = [data2['SO2'].max(), data2['NO2'].max(), data2['PM10'].max(), data2['PM2.5'].max(), data2['O3'].max(),
                  data2['CO'].max()]
    label2_min = [data2['SO2'].min(), data2['NO2'].min(), data2['PM10'].min(), data2['PM2.5'].min(), data2['O3'].min(),
                  data2['CO'].min()]

    label3_max = [data3['SO2'].max(), data3['NO2'].max(), data3['PM10'].max(), data3['PM2.5'].max(), data3['O3'].max(),
                  data3['CO'].max()]
    label3_min = [data3['SO2'].min(), data3['NO2'].min(), data3['PM10'].min(), data3['PM2.5'].min(), data3['O3'].min(),
                  data3['CO'].min()]

    label_max_list = [label_max, label1_max, label2_max, label3_max]
    label_min_list = [label_min, label1_min, label2_min, label3_min]
    df = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df1 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df2 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df3 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    # print(df)

    # 构造X和Y


    data_pre=pd.read_csv(preFile_A_1)
    data_pre.drop('模型运行日期', axis=1, inplace=True)

    data1_pre = pd.read_csv(preFile_A1_1)
    data1_pre.drop('模型运行日期', axis=1, inplace=True)

    data2_pre = pd.read_csv(preFile_A2_1)
    data2_pre.drop('模型运行日期', axis=1, inplace=True)

    data3_pre = pd.read_csv(preFile_A3_1)
    data3_pre.drop('模型运行日期', axis=1, inplace=True)




    data_pre=data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    data1_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    data2_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    data3_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))



    Y = [[], [], [], []]
    seq_len = sequence_length
    X_pre = []
    X_act = []


    for i in range(0,df.shape[0] - seq_len-24*2,24):
        x0=np.array(df.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x1=np.array(df1.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x2 = np.array(df2.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x3 = np.array(df3.iloc[i:(i + seq_len), ].values, dtype=np.float32)
        x_list=[x0,x1,x2,x3]
        x_list=np.array(x_list)
        X_act.append(x_list.transpose(0,2,1))
        k = i // 24
        x0 = np.concatenate(np.array(data_pre.iloc[k:(k + 3), ].values, dtype=np.float32))
        x1 = np.concatenate(np.array(data1_pre.iloc[k:(k + 3), ].values, dtype=np.float32))
        x2 = np.concatenate(np.array(data2_pre.iloc[k:(k + 3), ].values, dtype=np.float32))
        x3 = np.concatenate(np.array(data3_pre.iloc[k:(k + 3), ].values, dtype=np.float32))
        X_pre.append([x0,x1,x2,x3])
        j=0
        for d in [df,df1,df2,df3]:
            d1 = np.array(d.iloc[i + seq_len - 8:i + seq_len - 8 + 24, 0:6], dtype=np.float32)
            d2 = np.array(d.iloc[16 + i + seq_len:(16 + i + seq_len + 24), 0:6], dtype=np.float32)
            d3 = np.array(d.iloc[16 + i + seq_len + 24:(16 + i + seq_len + 48), 0:6], dtype=np.float32)
            d1 = np.mean(d1, axis=0)
            d2 = np.mean(d2, axis=0)
            d3 = np.mean(d3, axis=0)
            Y[j].append(np.concatenate((d1, d2, d3)))
            j=j+1


    # 构建batch
    X_pre=np.array(X_pre)
    total_len = len(Y[0])
    train_loader_list=[]
    test_loader_list=[]
    for i in range(4):
        trainx_act, trainx_pre, trainy = X_act[:int(0.80 * total_len)], X_pre[:int(0.80 * total_len)], Y[i][:int(
            0.80 * total_len)]
        testx_act, textx_pre, testy = X_act[int(0.80 * total_len):], X_pre[int(0.80 * total_len):], Y[i][
                                                                                                    int(0.80 * total_len):]
        train_loader_list.append(DataLoader(dataset=Mydataset_2(trainx_act,trainx_pre, trainy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True))
        test_loader_list.append(DataLoader(dataset=Mydataset_2(testx_act,textx_pre, testy, transform=transforms.ToTensor()), batch_size=batchSize,
                              shuffle=True))
    return label_max_list,label_min_list,train_loader_list,test_loader_list





def getData_predict(data_pre_file,data_act_file,sequence_length,batchSize):
    data_pre = pd.read_csv(data_pre_file)
    data = pd.read_excel(data_act_file)

    data.drop('time', axis=1, inplace=True)
    data_pre.drop('模型运行日期', axis=1, inplace=True)
    data = data.loc[10776:]

    label_max = []
    label_min = []

    label_max.append(data['SO2'].max())
    label_min.append(data['SO2'].min())
    label_max.append(data['NO2'].max())
    label_min.append(data['NO2'].min())
    label_max.append(data['PM10'].max())
    label_min.append(data['PM10'].min())
    label_max.append(data['PM2.5'].max())
    label_min.append(data['PM2.5'].min())
    label_max.append(data['O3'].max())
    label_min.append(data['O3'].min())
    label_max.append(data['CO'].max())
    label_min.append(data['CO'].min())

    df = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    data_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    seq_len = sequence_length
    X_pre = []
    X_act = []
    X_act.append(np.array(df.iloc[len(df)-seq_len:, ].values, dtype=np.float32))
    print(X_act)
    X_pre.append(np.concatenate(np.array(data_pre.iloc[len(data_pre)-seq_len//24:, ].values, dtype=np.float32)))
    print(X_pre)
    predict_loader = DataLoader(dataset=MyPredictdataset(X_act, X_pre), batch_size=batchSize, shuffle=True)


    return label_max, label_min, predict_loader

def getCovData_predict(preFile_A_1,preFile_A1_1,preFile_A2_1,preFile_A3_1,
                 preFile_A,preFile_A1,preFile_A2,preFile_A3,sequence_length,batchSize):
    data = pd.read_excel(preFile_A)
    data.drop('time', axis=1, inplace=True)
    data = data.loc[6432:]

    label_max = []
    label_min = []

    label_max.append(data['SO2'].max())
    label_min.append(data['SO2'].min())
    label_max.append(data['NO2'].max())
    label_min.append(data['NO2'].min())
    label_max.append(data['PM10'].max())
    label_min.append(data['PM10'].min())
    label_max.append(data['PM2.5'].max())
    label_min.append(data['PM2.5'].min())
    label_max.append(data['O3'].max())
    label_min.append(data['O3'].min())
    label_max.append(data['CO'].max())
    label_min.append(data['CO'].min())

    df = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    data1 = pd.read_excel(preFile_A1)
    data1.drop('time', axis=1, inplace=True)
    data1 = data1.loc[6432:]
    data2 = pd.read_excel(preFile_A2)
    data2.drop('time', axis=1, inplace=True)
    data2 = data2.loc[6432:]
    data3 = pd.read_excel(preFile_A3)
    data3.drop('time', axis=1, inplace=True)
    data3 = data3.loc[6432:]

    label1_max = [data1['SO2'].max(), data1['NO2'].max(), data1['PM10'].max(), data1['PM2.5'].max(), data1['O3'].max(),
                  data1['CO'].max()]
    label1_min = [data1['SO2'].min(), data1['NO2'].min(), data1['PM10'].min(), data1['PM2.5'].min(), data1['O3'].min(),
                  data1['CO'].min()]

    label2_max = [data2['SO2'].max(), data2['NO2'].max(), data2['PM10'].max(), data2['PM2.5'].max(), data2['O3'].max(),
                  data2['CO'].max()]
    label2_min = [data2['SO2'].min(), data2['NO2'].min(), data2['PM10'].min(), data2['PM2.5'].min(), data2['O3'].min(),
                  data2['CO'].min()]

    label3_max = [data3['SO2'].max(), data3['NO2'].max(), data3['PM10'].max(), data3['PM2.5'].max(), data3['O3'].max(),
                  data3['CO'].max()]
    label3_min = [data3['SO2'].min(), data3['NO2'].min(), data3['PM10'].min(), data3['PM2.5'].min(), data3['O3'].min(),
                  data3['CO'].min()]

    label_max_list = [label_max, label1_max, label2_max, label3_max]
    label_min_list = [label_min, label1_min, label2_min, label3_min]
    df = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df1 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df2 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    df3 = data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    # print(df)

    # 构造X和Y

    data_pre = pd.read_csv(preFile_A_1)
    data_pre.drop('模型运行日期', axis=1, inplace=True)

    data1_pre = pd.read_csv(preFile_A1_1)
    data1_pre.drop('模型运行日期', axis=1, inplace=True)

    data2_pre = pd.read_csv(preFile_A2_1)
    data2_pre.drop('模型运行日期', axis=1, inplace=True)

    data3_pre = pd.read_csv(preFile_A3_1)
    data3_pre.drop('模型运行日期', axis=1, inplace=True)

    data_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    data1_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    data2_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    data3_pre = data_pre.apply(lambda x: (x - min(x)) / (max(x) - min(x)))


    seq_len = sequence_length
    X_pre = []
    X_act = []

    x0 = np.array(df.iloc[len(df)-seq_len:, ].values, dtype=np.float32)
    x1 = np.array(df1.iloc[len(df)-seq_len:, ].values, dtype=np.float32)
    x2 = np.array(df2.iloc[len(df)-seq_len:, ].values, dtype=np.float32)
    x3 = np.array(df3.iloc[len(df)-seq_len:, ].values, dtype=np.float32)
    x_list = [x0, x1, x2, x3]
    x_list = np.array(x_list)
    X_act.append(x_list.transpose(0, 2, 1))

    x0 = np.concatenate(np.array(data_pre.iloc[len(data_pre)-seq_len//24:, ].values, dtype=np.float32))
    x1 = np.concatenate(np.array(data1_pre.iloc[len(data_pre)-seq_len//24:, ].values, dtype=np.float32))
    x2 = np.concatenate(np.array(data2_pre.iloc[len(data_pre)-seq_len//24:, ].values, dtype=np.float32))
    x3 = np.concatenate(np.array(data3_pre.iloc[len(data_pre)-seq_len//24:, ].values, dtype=np.float32))
    X_pre.append([x0, x1, x2, x3])
    X_pre=np.array(X_pre)

    predict_loader = DataLoader(dataset=MyPredictdataset(X_act, X_pre,transform=transforms.ToTensor()), batch_size=batchSize, shuffle=True)


    return label_max_list, label_min_list, predict_loader

class Mydataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)

class Mydataset_2(Dataset):
    def __init__(self, xx,xp, yy, transform=None):
        self.x = xx
        self.xp = xp
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        xp1 = self.xp[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1),self.tranform(xp1) ,y1
        return x1,xp1,y1

    def __len__(self):
        return len(self.x)

class MyPredictdataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.tranform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.tranform != None:
            return self.tranform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)

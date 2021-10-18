from LSTMModel import cnn_lstm
from LSTMModel import cnn_lstm_2
from dataset import getData
from dataset import getCovData_2
from parser_my import args
import torch
import math

def max_index(lst_int):
    index = []
    max_n = max(lst_int)
    for i in range(len(lst_int)):
        if lst_int[i] == max_n:
            index.append(i)
    return index
def getAQI(list):
    IAQI_temp=[]
    IAQI=[
    [0,50,100,150,200,300,400,500],
    [0,50,150,475,800,1600,2100,2620],
    [0,40,80,180,280,565,750,940],
    [0,50,150,250,350,420,500,600],
    [0,35,75,115,150,250,350,500],
    [0,100,160,215,265,800],
    [0,2,4,14,24,36,48,60]]
    for i in range(len(list)):
        Cp=list[i]
        IAQI_poll=IAQI[i+1]
        BP_Hi,BP_Lo=0,0
        IAQI_Hi,IAQI_Lo=0,0
        for j in range(len(IAQI_poll)):
            if  Cp<IAQI_poll[j]:
                BP_Hi=IAQI_poll[j]
                IAQI_Hi=IAQI[0][j]
                if j==0:
                    BP_Lo=0
                    IAQI_Lo=0
                else:
                    BP_Lo=IAQI_poll[j-1]
                    IAQI_Lo=IAQI[0][j-1]
                break
        if BP_Hi==BP_Lo:
            IAQI_t=0
        else:
            IAQI_t=math.ceil(((IAQI_Hi-IAQI_Lo)/(BP_Hi-BP_Lo))*(Cp-BP_Lo)+IAQI_Lo)
        IAQI_temp.append(IAQI_t)
    return max(IAQI_temp),max_index(IAQI_temp)

def eval():
    # model = torch.load(args.save_file)
    preModel = cnn_lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers,
                    output_size=args.output_size)
    preModel.to(args.device)
    checkpoint = torch.load(args.save_file_preModelCov)
    preModel.load_state_dict(checkpoint['state_dict'])

    model = cnn_lstm_2(input_size=args.input_size, hidden_size=args.hidden_size,
                   num_layers=args.layers, output_size=args.output_size, first_size=args.first_size,
                   dropout=args.dropout, batch_first=args.batch_first)
    model.to(args.device)
    checkpoint = torch.load(args.save_file_Cov)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    first_pre=[]
    print("preModel:")
    print(preModel)
    print("Model:")
    print(model)
    label_max_list, label_min_list, train_loader_list, test_loader_list = getCovData_2(args.preFileA_1,args.preFileA1_1,args.preFileA2_1,args.preFileA3_1,
                                                                   args.preFileA0,args.preFileA1,args.preFileA2,args.preFileA3,
                                                                   args.sequence_length,args.batch_size )
    test_loader = test_loader_list[args.cov]
    for idx, (x_act,x_pre,label) in enumerate(test_loader):
        if args.useGPU:
            x_act = x_act.squeeze(1).cuda()
        else:
            x_act = x_act.squeeze(1)
            print(x_act.shape)

        pred = model(x_act, x_pre, preModel)

        list = pred.data.squeeze(1).tolist()

        preds=preds+list


        labels.extend(label.tolist())
    loss=[0,0,0,0,0,0]
    APstr=['SO2','NO2','PM10','PM2.5','O3','CO']
    AQI_Loss=0
    mainP=0
    label_max=label_max_list[args.cov]
    label_min=label_min_list[args.cov]
    for i in range(len(preds)):
        for j in range(3):
            list_pre = []
            list_label = []
            for k in range(6):
                pre_val=preds[i][j*6+k] * (label_max[k] - label_min[k]) + label_min[k]
                tru_val=labels[i][j*6+k] * (label_max[k] - label_min[k]) + label_min[k]
                loss[k]= loss[k]+(pre_val-tru_val)**2
                list_pre.append(pre_val)
                list_label.append(tru_val)
                print('第%d天%s的预测值是%.2f,真实值是%.2f' % (j+1,APstr[k],
                    pre_val,tru_val ))
            AQI_pre,mainP_pre=getAQI(list_pre)
            AQI_label,mainP_label=getAQI(list_label)
            AQI_Loss=AQI_Loss+(AQI_pre-AQI_label)**2
            if set(mainP_label)<=set(mainP_pre):
                mainP=mainP+1
            else:
                print("mainP_pre:",mainP_pre)
                print("mainP_label:",mainP_label)

    AQI_Loss=AQI_Loss/(len(preds)*3)
    mainP=mainP/(len(preds)*3)
    for i in range(len(loss)):
        loss[i]=loss[i]/(len(preds)*3)

    print("CP MSELoss(SO2,NO2,PM10,PM2.5,O3,CO):",loss)
    print(AQI_Loss,mainP)



eval()
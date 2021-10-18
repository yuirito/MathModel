from LSTMModel import lstm
from dataset import getData
from parser_my import args
import torch


def eval(type):
    # model = torch.load(args.save_file)
    model = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=args.output_size)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    preds = []
    labels = []
    label_max, label_min, train_loader, test_loader = getData(args.corpusFile, args.sequence_length, args.batch_size)
    for idx, (x, label) in enumerate(test_loader):
        if args.useGPU:
            x = x.squeeze(1).cuda()  # batch_size,seq_len,input_size
        else:
            x = x.squeeze(1)
        pred,_ = model(x)
        list = pred.data.squeeze(1).tolist()
        print(list)
        preds.extend(list[-1])
        labels.extend(label.tolist())

    print(len(preds))
    print(preds)
    APstr=['SO2','NO2','PM10','PM2.5','O3','CO']
    for i in range(len(preds)):
        for j in range(3):
            for k in range(type):
                print('第%d天%s的预测值是%.2f,真实值是%.2f' % (j+1,APstr[k],
                    preds[i][j*6+k] * (label_max[k] - label_min[k]) + label_min[k], labels[i][j*6+k] * (label_max[k] - label_min[k]) + label_min[k]))


eval(6)
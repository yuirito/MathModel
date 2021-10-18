from torch.autograd import Variable
import torch.nn as nn
import torch
from LSTMModel import lstm
from LSTMModel import lstm_2
from parser_my import args
from dataset import getData
from dataset import getData_2
import matplotlib.pyplot as plt

def train():
    preModel = lstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers,
                 output_size=args.output_size)
    preModel.to(args.device)
    checkpoint = torch.load(args.save_file_preModel)
    preModel.load_state_dict(checkpoint['state_dict'])

    model = lstm_2(input_size=args.input_size, hidden_size=args.hidden_size,
                   num_layers=args.layers , output_size=args.output_size,first_size=args.first_size,
                   dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # Adam梯度下降  学习率=0.001

    label_max, label_min, train_loader, test_loader = getData_2(args.preFile,args.actFile,args.sequence_length,args.batch_size )
    min_loss=0x3f3f3f
    loss_list=[]
    for i in range(args.epochs):
        total_loss = 0

        for idx, (x_act,x_pre,label) in enumerate(train_loader):
            if args.useGPU:
                x_act = x_act.squeeze(1).cuda()
                pred,_ = model(Variable(x_act),Variable(x_pre),preModel).cuda()
                # print(pred.shape)
                pred = pred[-1,:,:]
                label = label.unsqueeze(1).cuda()
                # print(label.shape)
            else:
                x_act = x_act.squeeze(1)
                pred = model(Variable(x_act),Variable(x_pre),preModel)
                #print(pred.shape)
                #pred = pred[-1, :, :]
                label = label.unsqueeze(1)


            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)
        loss_list.append(total_loss)
        if min_loss > total_loss and i>80:
            min_loss = total_loss
            torch.save({'state_dict': model.state_dict()}, args.save_file)
            print('第%d epoch，保存模型' % i)
        if i % 10 == 0:
            # torch.save(model, args.save_file)
            print('第%d epoch' % i)
    # torch.save(model, args.save_file)
    #torch.save({'state_dict': model.state_dict()}, args.save_file)
    x = range(1, args.epochs + 1)
    plt.plot(x, loss_list, label="MSELoss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

print(torch.cuda.is_available())
train()
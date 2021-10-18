import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--pretrainFile', default='data/AcDataC_pre.xlsx')
parser.add_argument('--preFile', default='./data/DataA_1pre.csv')
parser.add_argument('--preFileA_1', default='./data/DataA_1pre.csv')
parser.add_argument('--preFileA1_1', default='./data/DataA1_1pre.csv')
parser.add_argument('--preFileA2_1', default='./data/DataA2_1pre.csv')
parser.add_argument('--preFileA3_1', default='./data/DataA3_1pre.csv')

parser.add_argument('--actFile', default='data/AcDataC_pre.xlsx')
parser.add_argument('--preFileA', default='./data/AcDataA_pre.xlsx')
parser.add_argument('--preFileA0', default='./data/AcDataA0_pre.xlsx')
parser.add_argument('--preFileA1', default='./data/AcDataA1_pre.xlsx')
parser.add_argument('--preFileA2', default='./data/AcDataA2_pre.xlsx')
parser.add_argument('--preFileA3', default='./data/AcDataA3_pre.xlsx')

# TODO 常改动参数
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--epochs', default=100, type=int) # 训练轮数
parser.add_argument('--layers', default=3, type=int) # LSTM层数
parser.add_argument('--input_size', default=11, type=int) #输入特征的维度
parser.add_argument('--output_size', default=18, type=int)
parser.add_argument('--first_size', default=63, type=int)
parser.add_argument('--hidden_size', default=33, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
parser.add_argument('--sequence_length', default=72, type=int) # sequence的长度，默认是用前五天的数据来预测下一天的收盘价
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--save_file', default='model/AP.pkl') # 模型保存位置
parser.add_argument('--save_file_preModel', default='model/AP_preModel.pkl') # 模型保存位置
parser.add_argument('--save_file_preModelCov', default='model/AP_preModelCov.pkl') # 模型保存位置
parser.add_argument('--save_file_Cov', default='model/AP_Cov.pkl') # 模型保存位置
parser.add_argument('--first_r', default=10776, type=int) #C对应第一行是4896
parser.add_argument('--cov', default=0, type=int) #0:A,1:A1,2:a2,3:A3

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device
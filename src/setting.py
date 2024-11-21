#setting.py


import torch

PHRASES='get_phrases.py'
VECTOR = '../.vector_cache'
UNK = 0
PAD = 1
BATCH_SIZE = 16  #64
TRAIN_FILE = './data/trainp.txt'  # 训练集
DEV_FILE = './data/devp.txt'  # 验证集
TEST_FILE = './data/testp.txt'  # 测试文件
SAVE_FILE = 'save/modelp.pt'  # 模型保存路径(注意如当前目录无save文件夹需要自己创建)
LAYERS =  1 # encoder和decoder层数
D_MODEL = 512  # embedding 维度
D_FF = 1024  # feed forward第一个全连接层维数
H_NUM = 8  # multi head attention hidden个数
DROPOUT = 0.1  # dropout比例
EPOCHS = 60   #60
MAX_LENGTH = 512
SRC_VOCAB = 25489  
TGT_VOCAB = 11800  

# 这里针对的是DEV文件
BLEU_REFERENCES = "data/bleu/reference.txt" # BLEU评价参考译文
BLEU_CANDIDATE = "data/bleu/candidate.txt"  # 模型翻译译文
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


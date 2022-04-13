import torch
from torch.utils.data import TensorDataset
from model.breast_model import *
from data_reader.data_reader import *
import torch.nn.functional as F
from utils.evaluation import *
device = "cpu"
torch.manual_seed(777)
if device =="cuda:0":
    torch.cuda.manual_seed_all(777)



if __name__ == '__main__':
    # load origin data
    data = Data('breast')
    X_train = np.asarray(data.X_train)
    Y_train = np.asarray(data.Y_train)
    Y_train = Y_train.reshape(-1,1)
    X_test = np.asarray(data.X_test)
    Y_test = np.asarray(data.Y_test)
    Y_test = Y_test.reshape(-1, 1)

    X_train = torch.from_numpy(X_train)
    Y_train = torch.from_numpy(Y_train)
    X_test = torch.from_numpy(X_test)
    Y_test = torch.from_numpy(Y_test)
    X_train = X_train.float()
    X_test = X_test.float()
    Y_train = Y_train.long()
    Y_test = Y_test.long()


    # 创建模型
    central_model = breast_client().to(device)
    epoch = 5000 # default
    lr = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(central_model.weights, lr=lr, momentum=0.9)


    # 训练模型
    for i in range(epoch):
        out = central_model(X_train)
        loss = criterion(out, Y_train.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i %1000 ==0:
            # 评估模型
            out = getBinaryTensor_2(central_model(X_test)).long()
            print(evaluate(out, Y_test))

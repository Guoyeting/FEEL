import torch
from torch.utils.data import TensorDataset
from model.breast_model import *
from data_reader.data_reader import *
import torch.nn.functional as F
from utils.evaluation import *
import time
from multiprocessing import Pool
device = "cpu"
torch.manual_seed(777)
if device =="cuda:0":
    torch.cuda.manual_seed_all(777)

# load data
data = Data('breast')
X_train = np.asarray(data.X_train)
Y_train = np.asarray(data.Y_train)
X_test = np.asarray(data.X_test)
Y_test = np.asarray(data.Y_test)
Y_test = Y_test.reshape(-1, 1)
X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test)
X_test = X_test.float()
Y_test = Y_test.long()

# create clients' dataset
client_number = 100
clients = Clients(X_train, Y_train, client_number)
X_train = np.asarray(clients.client_x)
Y_train = np.asarray(clients.client_y)
client_train_dataset = []
for i in range(client_number):
    X_train[i] = torch.from_numpy(X_train[i])
    Y_train[i] = Y_train[i].reshape(-1, 1)
    Y_train[i] = torch.from_numpy(Y_train[i])
    X_train[i] = X_train[i].float()
    Y_train[i] = Y_train[i].float()

F_measure = []
Accuracy = []
Precision = []
Recall = []

def call_client(i):

    # create client model
    client_model = breast_client().to(device)
    epoch_num = 1000
    lr = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(client_model.weights, lr=lr)

    for epoch in range(epoch_num):
        out = client_model(X_train[i])
        loss = criterion(out, Y_train[i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    out = getBinaryTensor_2(client_model(X_test)).long()
    accuracy, precision, recall, F_score= evaluate(out, Y_test)

    print('Client: ', i, '; Epoch: ', epoch_num, '; Accuracy: ', accuracy, "; Precision:", precision, "; Recall:", recall, '; F_measure:',
          F_score, '; ', len(Accuracy))
    return accuracy, precision, recall, F_score


def thread(n):
    pool = Pool(processes=n)
    result = []
    for i in range(0, n):
        result.append(pool.apply_async(call_client, args=(i,)))
    pool.close()
    pool.join()

    for i in result:
        accuracy, precision, recall, F_score = i.get()
        Accuracy.append(accuracy)
        F_measure.append(F_score)
        Precision.append(precision)
        Recall.append(recall)

    print('The Average Accuracy:', sum(Accuracy) / client_number)
    print('The Average Precision', sum(Precision) / client_number)
    print('The Average Recall:', sum(Recall) / client_number)
    print('The Average F_measure', sum(F_measure) / client_number)
    print('The Min of Accuracy, Precision, Recall, F_measure: ', min(Accuracy), \
          min(Precision), min(Recall), min(F_measure))

def main():
    thread(client_number)

if __name__ == '__main__':
    main()

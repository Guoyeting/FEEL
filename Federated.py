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


def main():
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
    test_dataset = TensorDataset(X_test, Y_test)

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

    # create global model
    global_model = breast_client().to(device)


    # create client model
    client_model = breast_client().to(device)
    lr = 0.01
    epoch_num = 1
    batch_size = 10
    round_num = 500
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(client_model.weights, lr=lr)

    # federated learning training
    for round in range(round_num):
        client_params = []
        for item in global_model.weights:
            client_params.append(torch.zeros_like(item))
        # local training
        for i in range(client_number):
            for j in range(len(global_model.weights)):
                client_model.weights[j].data = global_model.weights[j].data.clone().detach()
            for epoch in range(epoch_num):
                out = client_model(X_train[i])
                loss = criterion(out, Y_train[i])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            for j in range(len(client_model.weights)):
                client_params[j] += client_model.weights[j]
        # model aggregation
        for j in range(len(client_params)):
            global_model.weights[j].data = client_params[j]/client_number

        if round % 10 == 0:
            # evaluation
            out = getBinaryTensor_2(global_model(X_test)).long()
            print(evaluate(out, Y_test))


if __name__ == '__main__':
    main()

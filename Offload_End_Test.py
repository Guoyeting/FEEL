import torch
from torch.utils.data import TensorDataset
from model.breast_model import *
from data_reader.data_reader import *
import torch.nn.functional as F
from utils.evaluation import *
from multiprocessing import Process
from config import *
device = "cpu"
torch.manual_seed(777)
if device =="cuda:0":
    torch.cuda.manual_seed_all(777)

epoch_num=500

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

def main():
    sock_edge = socket.socket()
    sock_edge.connect((EDGE_SERVER_ADDR, EDGE_SERVER_PORT))

    # create client model
    client_model = breast_client(type='client', split_point=split_point).to(device)
    optimizer_c = torch.optim.SGD(client_model.weights, lr=lr)
    criterion = nn.BCELoss()
    if 1:
        msg = recv_msg(sock_edge)
        if msg[0] == 'MSG_INVITE_TO_TRAIN':
            client_index = 1

            for epoch in range(epoch_num):
                # train in the end
                out = client_model(X_train[client_index])
                client_output = out.clone().detach()
                sf_out = torch.max(client_output, axis=0).values - torch.min(client_output, axis=0).values
                for k in range(len(X_train[client_index])):
                    if sigma_edge != 0:
                        noise = torch.normal(mean=0, std=sf_out * sigma_edge)
                        client_output[k] += noise

                send_msg(sock_edge, ['MSG_MID_OUTPUT_TO_EDGE', client_output, Y_train[client_index]])
                msg = recv_msg(sock_edge, 'MIG_GRADIENT_TO_END')
                client_gradient = msg[1]
                # update in the end
                optimizer_c.zero_grad()
                out.backward(client_gradient)
                optimizer_c.step()


if __name__ == '__main__':
    main()

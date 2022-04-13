import torch
from torch.utils.data import TensorDataset
from model.breast_model import *
from data_reader.data_reader import *
import torch.nn.functional as F
from utils.evaluation import *
from config import *
device = "cpu"
torch.manual_seed(777)
if device =="cuda:0":
    torch.cuda.manual_seed_all(777)

epoch_num=1000

def main():

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((EDGE_SERVER_ADDR, EDGE_SERVER_PORT))

    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip, port))
    print(client_sock)

    # create edge model
    edge_model = breast_client(type='server', split_point=split_point).to(device)
    optimizer_e = torch.optim.SGD(edge_model.weights, lr=lr)
    criterion = nn.BCELoss()

    send_msg(client_sock, ['MSG_INVITE_TO_TRAIN'])

    for epoch in range(epoch_num):
        msg = recv_msg(client_sock, 'MSG_MID_OUTPUT_TO_EDGE')
        client_output = msg[1]
        Y_train = msg[2]
        # send client_output to the edge
        client_output = client_output.requires_grad_(True)
        # train in the edge
        optimizer_e.zero_grad()
        edge_output = edge_model(client_output)
        loss_edge = criterion(edge_output, Y_train)

        loss_edge.backward()
        optimizer_e.step()
        send_msg(client_sock, ['MIG_GRADIENT_TO_END', client_output.grad.clone().detach()])

if __name__ == '__main__':
    main()
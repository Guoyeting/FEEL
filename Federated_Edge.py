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

def main():
    sock_cloud = socket.socket()
    sock_cloud.connect((CLOUD_SERVER_ADDR, CLOUD_SERVER_PORT))

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((EDGE_SERVER_ADDR, EDGE_SERVER_PORT))


    client_sock_all = []
    client_number = 100
    # Establish connections to each client, up to n_nodes clients
    while len(client_sock_all) < client_number:
        listening_sock.listen(5)
        print("Waiting for incoming connections...")
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip, port))
        print(client_sock)
        client_sock_all.append(client_sock)

    # create edge model
    edge_model = breast_client(type='server', split_point=split_point).to(device)
    optimizer_e = torch.optim.SGD(edge_model.weights, lr=lr)
    criterion = nn.BCELoss()

    #for round in range(round_num):
    while True:
        msg = recv_msg(sock_cloud)
        if msg[0] == 'MSG_INIT_CLOUD_TO_EDGE':
            send_model = msg[1]

            for j in range(len(send_model)):
                edge_model.weights[j].data = send_model[j].clone().detach()

            client_params = []

            # random select client
            client_select = np.random.permutation(client_number)[:int(client_number * client_ratio)]
            for i in client_select:
                send_msg(client_sock_all[i], ['MSG_INVITE_TO_TRAIN', i])

                for epoch in range(epoch_num):
                    msg = recv_msg(client_sock_all[i], 'MSG_MID_OUTPUT_TO_EDGE')
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
                    send_msg(client_sock_all[i],['MIG_GRADIENT_TO_END',  client_output.grad.clone().detach()])

                # send parameters to the cloud
                tmp_params = []
                for j in range(len(edge_model.weights)):
                    tmp_params.append(edge_model.weights[j].data - send_model[j])
                client_params.append(tmp_params)

            send_msg(sock_cloud, ['MIG_PARA_TO_CLOUD', client_params])
        elif msg[0] == 'MSG_TEST':
            send_model = msg[1]
            for j in range(len(send_model)):
                edge_model.weights[j].data = send_model[j].clone().detach()
            accuracy_sum,precision_sum,recall_sum,F_score_sum, Loss = 0, 0, 0, 0, 0
            client_select = np.arange(len(client_sock_all))
            for i in client_select:
                send_msg(client_sock_all[i], ['MSG_INVITE_TO_TEST', i])
                msg = recv_msg(client_sock_all[i], 'MSG_MID_OUTPUT_FOR_TEST')
                out = getBinaryTensor_2(edge_model(msg[1]))
                send_msg(client_sock_all[i], ['MSG_OUT_FOR_TEST', out])
                msg = recv_msg(client_sock_all[i], 'MSG_TEST_RESULT')
                accuracy, precision, recall, F_score, loss_test = msg[1:]
                accuracy_sum += accuracy
                precision_sum += precision
                recall_sum += recall
                F_score_sum += F_score
                Loss += loss_test
            send_msg(sock_cloud, ['MIG_TEST_RESULT', accuracy_sum/(len(client_select)), \
                                  precision_sum/(len(client_select)), recall_sum/(len(client_select)), F_score_sum/(len(client_select)), Loss/(len(client_select))])
        else:
            for i in client_sock_all:
                send_msg(i, ['MSG_FOR_STOP'])
            break

if __name__ == '__main__':
    main()

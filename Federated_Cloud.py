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

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_sock.bind((CLOUD_SERVER_ADDR, CLOUD_SERVER_PORT))

    listening_sock.listen(5)
    print("Waiting for incoming connections...")
    (client_sock, (ip, port)) = listening_sock.accept()
    print('Got connection from ', (ip, port))
    print(client_sock)


    # create global model
    global_model = breast_client(type='server', split_point=split_point).to(device)

    # federated learning training
    for round in range(round_num):
        send_model = []
        for j in range(len(global_model.weights)):
            send_model.append(global_model.weights[j].data.clone().detach())
        msg = ['MSG_INIT_CLOUD_TO_EDGE', send_model]
        send_msg(client_sock, msg)

        msg = recv_msg(client_sock, 'MIG_PARA_TO_CLOUD')
        client_params = msg[1]
        # calculate sf
        client_norms = []
        for i in range(len(client_params)):
            client_norms.append(calculate_norm(client_params[i]))
        sf = np.median(client_norms, axis=0)
        for i in range(len(client_params)):
            for j in range(len(client_norms[0])):
                client_params[i][j] = client_params[i][j] / max(1, client_norms[i][j] / sf[j])

        # model aggregation in the cloud
        client_params_sum = []
        for item in global_model.weights:
            client_params_sum.append(torch.zeros_like(item))

        for j in range(len(client_params_sum)):
            for i in range(len(client_params)):
                client_params_sum[j] += client_params[i][j]
            if sf[j] * sigma != 0:
                client_params_sum[j] += torch.normal(mean=0., std=sf[j] * sigma, size=global_model.weights[j].shape)
            client_params_sum[j] = client_params_sum[j] / client_number
            global_model.weights[j].data += client_params_sum[j]

        #if round%10 == 0 or round == round_num-1:
        if 1:
            send_model = []
            for j in range(len(global_model.weights)):
                send_model.append(global_model.weights[j].data.clone().detach())
            send_msg(client_sock, ['MSG_TEST', send_model])
            msg = recv_msg(client_sock, 'MIG_TEST_RESULT')
            Accuracy, Precision, Recall, F_measure, loss = msg[1:]
            print('Round: ', round, '; Accuracy: ', Accuracy, "; Precision:", Precision, "; Recall:", Recall, '; F_measure:', F_measure, "; Loss:", loss)


    send_msg(client_sock, ['MSG_FOR_STOP'])




if __name__ == '__main__':
    main()

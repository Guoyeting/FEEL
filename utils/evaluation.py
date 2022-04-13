import torch
import pickle, struct, socket, math

def evaluate(prediction, ground):

    num_correct = torch.eq(prediction, ground).sum().float().item()
    accuracy = num_correct / ground.size()[0]

    TP = torch.multiply(prediction, ground).sum().float().item()
    positive = prediction.sum().float().item()
    if positive != 0:
        precision = TP / positive
    else:
        precision = 0

    FN = (ground > prediction).sum().float().item()
    recall = TP / (TP + FN)

    if precision != 0:
        F_score = (2 * precision * recall) / (precision + recall)
    else:
        F_score = 0
    return accuracy, precision, recall, F_score


def calculate_norm(weights):
    norm = []
    for i in range(len(weights)):
        norm.append(torch.norm(weights[i]))
    return norm


def recv_msg(sock, expect_msg_type=None):
    msg_len = struct.unpack(">I", sock.recv(4))[0]
    msg = sock.recv(msg_len, socket.MSG_WAITALL)
    msg = pickle.loads(msg)

    if (expect_msg_type is not None) and (msg[0] != expect_msg_type):
        raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
    return msg


def send_msg(sock, msg):
    msg_pickle = pickle.dumps(msg)
    sock.sendall(struct.pack(">I", len(msg_pickle)))
    sock.sendall(msg_pickle)

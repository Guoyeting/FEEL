import os
import sys
import numpy as np
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def random_data(X_train, Y_train):
    train_sample = len(X_train)
    # 随机扰乱数据
    perm = np.random.permutation(train_sample)
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # 根据第一个属性的类别大小进行排序
    X_train_one = X_train[:, 0]
    X_train_index = np.argsort(X_train_one)

    X_train = X_train[X_train_index, :]
    Y_train = Y_train[X_train_index]

    # 记录第一个属性每个类别数据的断点
    break_point = [0] * 10
    X_train_one = X_train[:, 0]
    index_num = 1
    for i in range(train_sample):
        if X_train_one[i] == index_num:
            index_num += 1
        break_point[index_num - 2] += 1

    # 基于第一属性排序后扰动数据集
    random_index = []
    for i in range(10):
        perm = np.random.permutation(break_point[i]) + np.ones(break_point[i]) * sum(break_point[:i])
        perm = perm.astype(np.int32)
        perm = list(perm)
        random_index += perm
    X_train = X_train[random_index, :]
    Y_train = Y_train[random_index]

    return X_train, Y_train, break_point


def create_sets(client_num, scope, contain_num):
    shards = client_num * contain_num
    buck = []
    for i in range(scope):
        temp = []
        for j in range(shards):
            temp = np.hstack((temp, i + 1))
        buck = np.hstack((buck, temp))
    total_num = dict()
    num_set = list()
    perm = np.random.permutation(shards)
    ind_list = np.split(buck, shards)
    for i in range(0, shards, contain_num):
        temp = []
        for j in range(contain_num):
            temp = np.hstack((temp, ind_list[int(perm[i + j])]))
        b = set(temp)
        while (len(b) < contain_num):
            b.add(random.randint(1, scope))
        num_set.append(b)
        for v in b:
            if v in total_num.keys():
                total_num[v] += 1
            else:
                total_num[v] = 1

    check = 0
    for v in total_num.keys():
        check += total_num[v]

    for b in num_set:
        if len(b) > contain_num:
            remove_list = []
            for v in b:
                if total_num[v] > 1:
                    remove_list.append([v, total_num[v]])
            remove_list.sort(key=lambda t: t[1], reverse=True)
            for v in remove_list:
                if len(b) != contain_num:
                    b.remove(v[0])
                    total_num[v[0]] -= 1
                else:
                    break
    return num_set


def create_clients(client_num, break_point):
    scope = 10
    contain_num = 1
    z = create_sets(client_num, scope, contain_num)

    exceed = dict()
    for i in range(scope):
        if break_point[i] > 30:
            exceed[i+1] = 0

    for i in range(client_num):
        for j in z[i]:
            if j in exceed.keys():
                exceed[j] += 1

    indexs = dict()
    for j in exceed.keys():
        indexs[j] = create_sets(exceed[j], break_point[j-1], 30)

    clients_index = []
    for i in range(client_num):
        client_index = []
        for j in z[i]:
            if j in exceed.keys():
                select_item = indexs[j]
                select_item = list(select_item[exceed[j]-1])
                for v in range(30):
                    select_item[v] += sum(break_point[:int(j-1)])
                client_index += select_item
                exceed[j] -= 1
            else:
                select_item = [v+sum(break_point[:int(j-1)]) \
                               for v in range(break_point[int(j-1)])]
                client_index += select_item
        for v in range(len(client_index)):
            client_index[v] -= 1
        clients_index.append(client_index)
    return clients_index


def read_data(dataset_name):
# 读取数据集
    if dataset_name == 'breast':
        # load train dataset
        X_train = np.array([[float(j) for j in i.rstrip().split(",")] for i in open("datasets//breast//train.csv").readlines()])
        Y_train = X_train[:, -1]
        X_train = X_train[:, 0:-1]

        # load test dataset
        X_test = np.array([[float(j) for j in i.rstrip().split(",")] for i in open("datasets//breast//test.csv").readlines()])
        Y_test = X_test[:, -1]
        X_test = X_test[:, 0:-1]

    return X_train, Y_train, X_test, Y_test


class Data:
    def __init__(self, dataset_name):
        self.X_train, self.Y_train, self.X_test, self.Y_test = read_data(dataset_name)

class Clients:
    def __init__(self, X_train, Y_train, client_num):
        # 扰动数据
        self.X_train, self.Y_train, self.bp = random_data(X_train, Y_train)
        # 将数据分散到各个客户端，每个客户端有两种属性的数据
        Z = create_clients(client_num, self.bp)
        self.client_x = []
        self.client_y = []
        for i in range(len(Z)):
            Z[i] = np.array(Z[i])
            Z[i] = Z[i].astype(np.int32)
            self.client_x.append(self.X_train[Z[i], :])
            self.client_y.append(self.Y_train[Z[i]])

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
from Users_sets_Encoders import Users_sets_Encoder
from Users_sets_Aggregators import Users_sets_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity

class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)

        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        # [batch_size, embed_dim]
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)
        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)

        # 更新所有参数
        optimizer.step()
        running_loss += loss.item()

        print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
            epoch, i, running_loss, best_rmse, best_mae))
        running_loss = 0.0
    return 0


def valid(model, device, valid_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in valid_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    ta_res = defaultdict(list)
    ta_complete = defaultdict(list)
    ta_target = defaultdict(list)
    # 保存执行任务的用户组合，key为任务，value为用户组合
    group_res = defaultdict(list)
    # 读取社交邻接表数据
    social_adjlist_path = r'../ReprocessData/datasets/gowalla/NewYork_social_adjlist_modified.pickle'
    openSocial = open(social_adjlist_path, 'rb')
    social_adj_list = pickle.load(openSocial)
    # 记录每个用户执行了被同时分配给几个任务的字典
    user_allocation = defaultdict(list)
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            print(tmp_target.mean())
            print(tmp_target.std())
            print(val_output.mean())
            print(val_output.std())
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
            test_u_lst = test_u.tolist()
            test_v_lst = test_v.tolist()
            val_output_lst = val_output.tolist()
            target_lst = tmp_target.tolist()

            for i in range(0, len(test_u_lst)):
                ta_res[test_v_lst[i]].append(test_u_lst[i])
                ta_complete[test_v_lst[i]].append(val_output_lst[i])
                ta_target[test_v_lst[i]].append(target_lst[i])
    task_num = len(ta_complete)
    sum_comp = 0
    sum_comp1 = 0

    for key, value in ta_complete.items():
        # 找最优候选人
        max_comp = max(value)
        max_comp_index = value.index(max(value))
        for user in ta_res[key][max_comp_index]:
            if user != 0:
                user_allocation[user].append(key)
        group_res[key] = ta_res[key][max_comp_index]
        sum_comp = sum_comp + max_comp
        sum_comp1 = sum_comp1 + ta_target[key][max_comp_index]

    print('任务的估计平均完成度为%.4f' % (sum_comp / task_num))
    print('任务的估计对应平均完成度为%.4f' % (sum_comp1 / task_num))

    sum_comp2 = 0
    for key, value in ta_target.items():
        max_comp = max(value)
        max_comp_index = value.index(max(value))
        sum_comp2 = sum_comp2 + ta_target[key][max_comp_index]
    print('任务的实际平均完成度为%.4f' % (sum_comp2 / task_num))

    # 指标2-均衡性
    mutil_task_user_num = 0
    for key, value in user_allocation.items():
        if len(value) > 1:
            mutil_task_user_num = mutil_task_user_num + 1
    print('用户分配的均衡性为%.4f' % (1 - mutil_task_user_num / len(user_allocation.keys())))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    # 指标3-联系紧密度
    total_coef = 0
    for key, value in group_res.items():
        group_coef = 0
        link = 0
        N = len(value)
        if i in value:
            if i==0:
                N -= 1
        if N <= 1:
            max_link = 1
        else:
            max_link = N * (N - 1)
        # 实际上组合内部的边数
        for worker in value:
            # worker的邻居
            if worker != 0:
                N_i = social_adj_list[worker]
                # 查找邻居中的团队成员
                for worker1 in value:
                    if worker1 in N_i:
                        link += 1
        # link的值=团队中的边数*2
        # 团队的联系紧密度
        group_coef = link / 2 / max_link
        total_coef += group_coef
        # print(value, group_coef)
    print("团队联系紧密程度")
    print(total_coef / len(group_res))
    # 指标--平均移动距离
    dist_path = '../ReprocessData/datasets/gowalla/NewYork_dist_table.pickle'
    open_dist = open(dist_path, 'rb')
    dist_matrix = pickle.load(open_dist)
    total_value = 0
    for key, value in group_res.items():
        tmp_dist = 0
        count = 0
        for user in value:
            if user!=0:
                tmp_dist += dist_matrix[key - 1][user - 1]
                count += 1
        tmp_dist = tmp_dist / count
        total_value += tmp_dist
    print('用户的平均移动距离：', total_value / len(group_res))
    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    # 设置任务最大需求人数,默认设置为6
    parser.add_argument('--require_users', type=int, default=6, metavar='N', help='max number of users a task need')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--path', required=True, type=str, help='validR_dist_need_intimate_sample_train_0.6/0.8')
    parser.add_argument('--data',required=True, type=str)
    args = parser.parse_args()
    # 设置读取的数据集
    argument_train_path = args.path
    data_name = args.data

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = r'../ReprocessData/datasets/gowalla/NewYork_'
    dir_data = dir_data + argument_train_path
    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    social = r'../ReprocessData/datasets/gowalla/NewYork_social_adjlist_modified.pickle'
    social_file = open(social, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, valid_u, valid_v, valid_r, social_adj_lists, ratings_list = pickle.load(
        data_file)

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
                                              torch.FloatTensor(valid_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, num_workers=0, shuffle=True)

    u2e = nn.Embedding(4435, embed_dim, padding_idx=0).to(device)
    v2e = nn.Embedding(1001, embed_dim, padding_idx=0).to(device)
    r2e = nn.Embedding(101, embed_dim, padding_idx=0).to(device)

    # user feature
    # features: task * score
    agg_u_history = Users_sets_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = Users_sets_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, social_adj_lists,
                                       agg_u_history, cuda=device, uv=True)

    # task feature: user * score
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)
    # model
    graphrec = GraphRec(enc_u_history, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.8)

    if args.test:
        print('Load checkpoint and testing...')
        path = 'no_social_best_'+argument_train_path+'.pth.tar'
        ckpt = torch.load(path)
        graphrec.load_state_dict(ckpt['state_dict'])
        mae, rmse = test(graphrec, device, test_loader)
        print("Test: RMSE: {:.4f}, MAE: {:.4f}".format(mae, rmse))
        return

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = valid(graphrec, device, valid_loader)

        # please add the validation set to tune the hyper-parameters based on your datasets.
        # early stopping (no validation set in toy dataset)
        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch,
            'state_dict': graphrec.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        best_path = 'no_social_best_' + argument_train_path + '.pth.tar'
        latest_path ='no_social_latest_' + argument_train_path + '.pth.tar'
        torch.save(ckpt_dict, latest_path)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            torch.save(ckpt_dict, best_path)
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))
        #
        if endure_count > 5:
            break
        # 每一轮epoch结束之后都输出test的结果
        test_mae, test_rmse = test(graphrec, device, test_loader)
        print("Test: RMSE: {:.4f}, MAE: {:.4f}".format(test_mae, test_rmse))

if __name__ == "__main__":
    main()
import argparse
import pickle
import scipy
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import io
from Net import *
from smiles2vector import load_drug_smile
from math import *
import random
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as data
from sklearn.metrics import precision_score, recall_score, accuracy_score
from utils import *

raw_file = 'data/raw_frequency_750.mat'
SMILES_file = 'data/drug_SMILES_750.csv'
mask_mat_file = 'data/mask_mat_750.mat'
side_effect_label = 'data/side_effect_label_750.mat'
input_dim = 109
gii = open('data/drug_side.pkl', 'rb')
drug_side = pickle.load(gii)
gii.close()


def Extract_positive_negative_samples(DAL, addition_negative_number=''):
    k = 0
    interaction_target = np.zeros((DAL.shape[0] * DAL.shape[1], 3)).astype(int)
    for i in range(DAL.shape[0]):
        for j in range(DAL.shape[1]):
            interaction_target[k, 0] = i
            interaction_target[k, 1] = j
            interaction_target[k, 2] = DAL[i, j]
            k = k + 1
    data_shuffle = interaction_target[interaction_target[:, 2].argsort()]  # 按照最后一列对行排序
    number_positive = len(np.nonzero(data_shuffle[:, 2])[0])
    final_positive_sample = data_shuffle[interaction_target.shape[0] - number_positive::]
    negative_sample = data_shuffle[0:interaction_target.shape[0] - number_positive]
    a = np.arange(interaction_target.shape[0] - number_positive)
    a = list(a)
    if addition_negative_number == 'all':
        b = random.sample(a, (interaction_target.shape[0] - number_positive))
    else:
        b = random.sample(a, (1 + addition_negative_number) * number_positive)
    final_negtive_sample = negative_sample[b[0:number_positive], :]
    addition_negative_sample = negative_sample[b[number_positive::], :]
    final_positive_sample = np.concatenate((final_positive_sample, final_negtive_sample), axis=0)
    return addition_negative_sample, final_positive_sample, final_negtive_sample


# 损失函数
def loss_fun(output, label):
    output = output.to('cuda')
    label = label.to('cuda')
    loss = torch.sum((output - label) ** 2)
    return loss


# 数据预处理：提取有效子结构
def identify_sub():
    raw_frequency = scipy.io.loadmat(raw_file)
    raw = raw_frequency['R']
    mask_mat = scipy.io.loadmat(mask_mat_file)
    drug_dict, drug_smile = load_drug_smile(SMILES_file)

    # 获得SMILE-sub序号
    sub_dict = {}
    for i in range(750):
        drug_sub, mask = drug2emb_encoder(drug_smile[i])
        drug_sub = drug_sub.tolist()
        sub_dict[i] = drug_sub

    # 矩阵，i-副作用，j-子结构，对于每一个副作用，遍历所有药物，如果有fre，sij+=fre
    rawT = raw.T
    SE_sub = np.zeros((994, 2686))
    for j in range(994):
        nonzero_columns = np.where(rawT[j] != 0)
        print(j)
        for i in nonzero_columns[0]:
            fre = rawT[j][i]
            # i是指药物的编号
            for k in sub_dict[i]:
                if k == 0:
                    continue
                SE_sub[j][k] += 1

    # np.save("SE_sub_1.npy", SE_sub)

    SE_sub = np.load("SE_sub_1.npy")

    # 总和
    n = np.sum(SE_sub)
    # 计算行和
    SE_sum = np.sum(SE_sub, axis=1)
    SE_p = SE_sum / n
    # 计算列和
    Sub_sum = np.sum(SE_sub, axis=0)
    Sub_p = Sub_sum / n

    SE_sub_p = SE_sub / n

    freq = np.zeros((994, 2686))
    for i in range(994):
        for j in range(2686):
            freq[i][j] = (SE_sub_p[i][j] - SE_p[i] * Sub_p[j]) / (sqrt((SE_p[i] * Sub_p[j] / n)
                                                                       * (1 - SE_p[i]) *
                                                                       (1 - Sub_p[j])))
    # 存储结果的列表
    result = []

    # 遍历每一行
    for row in freq:
        # 存储大于1.2的值的列表
        values_gt_1_2 = []
        # 检查每个元素是否大于1.2
        for value in row:
            if value > 1.96:
                # 如果大于1.2，则添加到结果列表中
                values_gt_1_2.append(value)
        # 将当前行的结果添加到总结果列表中
        result.append(values_gt_1_2)

    print("Result:", result)


    l = []
    SE_sub_index = np.zeros((994, 50))
    for i in range(994):
        if i == 665:
            j = 1
        if i == 979:
            j = 3
        k = 0
        sorted_indices = np.argsort(freq[i])[::-1]
        filtered_indices = sorted_indices[freq[i][sorted_indices] > 1.96]
        for j in filtered_indices:
            if k < 50:
                SE_sub_index[i][k] = j
                k = k + 1
            else:
                continue

    np.save("data/SE_sub_index_50.npy", SE_sub_index)

    SE_sub_index = np.load("data/SE_sub_index_50.npy")

    SE_sub_mask = SE_sub_index
    SE_sub_mask[SE_sub_mask > 0] = 1
    np.save("data/SE_sub_mask_50.npy", SE_sub_mask)
    i = 1


def trainfun(model, device, train_loader, optimizer, epoch, log_interval, test_loader):
    # 确定训练集的数量
    print('Training on {} samples...'.format(len(train_loader.dataset)))

    # 开启训练模式
    model.train()
    avg_loss = []

    for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(train_loader):

        DrugMask = DrugMask.to(device)
        SEMsak = SEMsak.to(device)
        Label = torch.FloatTensor([int(item) for item in Label])

        optimizer.zero_grad()
        out, _, _ = model(Drug, SE, DrugMask, SEMsak)

        pred = out.to(device)

        loss = loss_fun(pred.flatten(), Label).to('cpu')

        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())

        if batch_idx % 100 == 0:
            print('Train epoch: {} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, len(train_loader.dataset),
                                                                    100. * (batch_idx + 1) / len(
                                                                        train_loader),
                                                                    loss.item()))
            print(loss)

    return sum(avg_loss) / len(avg_loss)


def predict(model, device, test_loader):
    # 声明为张量
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()

    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label])
            out, _, _ = model(Drug, SE, DrugMask, SEMsak)

            location = torch.where(Label != 0)
            pred = out[location]
            label = Label[location]

            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_labels = torch.cat((total_labels, label.cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate(model, device, test_loader):
    total_preds = torch.Tensor()
    total_label = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)

    with torch.no_grad():
        for batch_idx, (Drug, SE, DrugMask, SEMsak, Label) in enumerate(test_loader):
            DrugMask = DrugMask.to(device)
            SEMsak = SEMsak.to(device)
            Label = torch.FloatTensor([int(item) for item in Label])
            output, _, _ = model(Drug, SE, DrugMask, SEMsak)
            pred = output.cpu()
            pred = torch.Tensor(pred)

            total_preds = torch.cat((total_preds, pred), 0)
            total_label = torch.cat((total_label, Label), 0)

            pred = pred.numpy().flatten()
            pred = np.where(pred > 0.5, 1, 0)
            label = (Label.numpy().flatten() != 0).astype(int)
            label = np.where(label != 0, 1, label)

            singleDrug_auc.append(roc_auc_score(label, pred))
            singleDrug_aupr.append(average_precision_score(label, pred))

        drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
        drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
        total_preds = total_preds.numpy()
        total_label = total_label.numpy()

        total_pre_binary = np.where(total_preds > 0.5, 1, 0)
        label01 = np.where(total_label != 0, 1, total_label)

        pre_list = total_pre_binary.tolist()
        label_list = label01.tolist()

        precision = precision_score(pre_list, label_list)

        # 计算召回率
        recall = recall_score(pre_list, label_list)

        # 计算准确率
        accuracy = accuracy_score(pre_list, label_list)

        total_preds = np.where(total_preds > 0.5, 1, 0)
        total_label = np.where(total_label != 0, 1, total_label)

        pos = np.squeeze(total_preds[np.where(total_label)])
        pos_label = np.ones(len(pos))

        neg = np.squeeze(total_preds[np.where(total_label == 0)])
        neg_label = np.zeros(len(neg))

        y = np.hstack((pos, neg))
        y_true = np.hstack((pos_label, neg_label))
        auc_all = roc_auc_score(y_true, y)
        aupr_all = average_precision_score(y_true, y)

    return auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy


def main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay,log_interval, cuda_name,
         save_model,k):
    print('\n=======================================================================================')
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('weight_decay: ', weight_decay)



    model_st = modeling.__name__
    train_losses = []

    # 确定设备
    print('CPU/GPU: ', torch.cuda.is_available())
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    # 模型初始化
    model = modeling().to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        train_loss = trainfun(model=model, device=device,
                              train_loader=training_generator,
                              optimizer=optimizer, epoch=epoch + 1, log_interval=log_interval, test_loader=testing_generator)
        train_losses.append(train_loss)

        checkpointsFolder = 'checkpoints/'
        torch.save(model.state_dict(), checkpointsFolder + str(epoch))

    print("正在预测")
    test_labels, test_preds = predict(model=model, device=device, test_loader=testing_generator)

    ret_test = [rmse(test_labels, test_preds),MAE(test_labels, test_preds)]

    test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]

    print("正在评估")
    auc_all, aupr_all, drugAUC, drugAUPR, precision, recall, accuracy = evaluate(model=model, device=device,
                                                                                 test_loader=testing_generator)

    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, precision,
              recall, accuracy]

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print('\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}\tdrug Precise: {:.5f}\tRecall: {:.5f}\tdrug ACC: {:.5f}'.format(
            result[4], result[5],
            result[6], result[7], result[8], result[9], result[10]))




class Data_Encoder(data.Dataset):
    def __init__(self, list_IDs, labels, df_dti):

        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti

    def __len__(self):

        return len(self.list_IDs)

    def __getitem__(self, index):

        index = self.list_IDs[index]
        d = self.df.iloc[index]['Drug_smile']
        s = int(self.df.iloc[index]['SE_id'])

        # d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d)

        # 副作用的子结构是读取出来的
        SE_index = np.load("data/SE_sub_index_50.npy").astype(int)
        SE_mask = np.load("data/SE_sub_mask_50.npy")
        s_v = SE_index[s, :]
        input_mask_s = SE_mask[s, :]
        y = self.labels[index]
        return d_v, s_v, input_mask_d, input_mask_s, y


if __name__ == '__main__':
    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0)
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.01, help='weight_decay')
    parser.add_argument('--epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=40, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cpu', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200, help='features dimensions of drugs and side effects')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model and features')

    args = parser.parse_args()

    modeling = [Trans][args.model]
    lr = args.lr
    num_epoch = args.epoch
    weight_decay = args.wd
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    save_model = args.save_model


    #  获取正负样本
    addition_negative_sample, final_positive_sample, final_negative_sample = Extract_positive_negative_samples(
        drug_side, addition_negative_number='all')

    addition_negative_sample = np.vstack((addition_negative_sample, final_negative_sample))

    final_sample = final_positive_sample

    X = final_sample[:, 0::]

    final_target = final_sample[:, final_sample.shape[1] - 1]

    y = final_target
    data = []
    data_x = []
    data_y = []
    data_neg_x = []
    data_neg_y = []
    data_neg = []
    drug_dict, drug_smile = load_drug_smile(SMILES_file)
    for i in range(addition_negative_sample.shape[0]):
        data_neg_x.append((addition_negative_sample[i, 1], addition_negative_sample[i, 0]))
        data_neg_y.append((int(float(addition_negative_sample[i, 2]))))
        data_neg.append(
            (addition_negative_sample[i, 1], addition_negative_sample[i, 0], addition_negative_sample[i, 2]))
    for i in range(X.shape[0]):
        data_x.append((X[i, 1], X[i, 0]))
        data_y.append((int(float(X[i, 2]))))
        data.append((X[i, 1], drug_smile[X[i, 0]], X[i, 2]))

    fold = 1
    kfold = StratifiedKFold(10, random_state=1, shuffle=True)


    params = {'batch_size': 128,
              'shuffle': True}

    for k, (train, test) in enumerate(kfold.split(data_x, data_y)):
        data_train = np.array(data)[train]
        data_test = np.array(data)[test]

        # 将数据转为DataFrame
        df_train = pd.DataFrame(data=data_train.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])
        df_test = pd.DataFrame(data=data_test.tolist(), columns=['SE_id', 'Drug_smile', 'Label'])

        # 创建数据集和数据加载器
        training_set = Data_Encoder(df_train.index.values, df_train.Label.values, df_train)
        testing_set = Data_Encoder(df_test.index.values, df_test.Label.values, df_test)

        training_generator = torch.utils.data.DataLoader(training_set, **params)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)

        main(training_generator, testing_generator, modeling, lr, num_epoch, weight_decay, log_interval,
             cuda_name, save_model, k)

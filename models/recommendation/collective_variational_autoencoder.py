import argparse
from collections import defaultdict
from pprint import pprint
import pickle

import torch
from torch import nn
from torch import optim
from torch.nn import functional
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import pytrec_eval
from prettytable import PrettyTable


def regularization(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def Guassian_loss(recon_x, x):
    weights = x * args.alpha + (1 - x)
    loss = x - recon_x
    loss = torch.sum(weights * loss * loss)
    return loss


def BCE_loss(recon_x, x):
    eps = 1e-8
    loss = -torch.sum(args.alpha * torch.log(recon_x + eps) * x + torch.log(1 - recon_x + eps) * (1 - x))
    return loss


def train(epoch):
    model.train()
    loss_value = 0
    for batch_idx, data in enumerate(train_loader):

        data = data.to(args.device)
        # print(data.shape)
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data) + regularization(mu, logvar) * args.beta
        loss.backward()
        loss_value += loss.item()
        optimizer.step()
        if args.log != 0 and batch_idx % args.log == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, loss_value / len(train_loader.dataset)))
    return loss_value / len(train_loader.dataset)


def sort2query(run):
    m, n = run.shape
    return {str(i): {str(int(run[i, j])): float(1.0 / (j + 1)) for j in range(n)} for i in range(m)}


def test2dict(test):
  # test = test.numpy()

  dict_ = defaultdict(dict)
  for pair in test:
    # pair -> (disease i, drug j)
    dict_[str(pair[0])].update(
      {str(pair[1]): 1}
    )
  
  return dict_


class Evaluator:
    def __init__(self, metrics):
        self.result = {}
        self.metrics = metrics

    def evaluate(self, predict, test):
        evaluator = pytrec_eval.RelevanceEvaluator(test, self.metrics)
        self.result = evaluator.evaluate(predict)

    def show(self, metrics):
        result = {}
        for metric in metrics:
            res = pytrec_eval.compute_aggregated_measure(metric, [user[metric] for user in self.result.values()])
            result[metric] = res
            # print('{}={}'.format(metric, res))
        return result

    def show_all(self):
        key = next(iter(self.result.keys()))
        keys = self.result[key].keys()
        return self.show(keys)



# Implementation of Variaitonal Autoencoder
class VAE(nn.Module):
    # Define the initialization function，which defines the basic structure of the neural network
    def __init__(self, args):
        super(VAE, self).__init__()
        self.l = len(args.layer)
        self.L = args.L
        self.device = args.device
        self.inet = nn.ModuleList()
        darray = [args.d] + args.layer
        for i in range(self.l - 1):
            self.inet.append(nn.Linear(darray[i], darray[i + 1]))
        self.mu = nn.Linear(darray[self.l - 1], darray[self.l])
        self.sigma = nn.Linear(darray[self.l - 1], darray[self.l])
        self.gnet = nn.ModuleList()
        for i in range(self.l):
            self.gnet.append(nn.Linear(darray[self.l - i], darray[self.l - i - 1]))

    def encode(self, x):
        h = x
        for i in range(self.l - 1):
            h = functional.relu(self.inet[i](h))
            # h = functional.relu(functional.dropout(self.inet[i](h), p=0.5, training=True))
        return self.mu(h), self.sigma(h)

    def decode(self, z):
        h = z
        for i in range(self.l - 1):
            h = functional.relu(self.gnet[i](h))
            # h = functional.relu(functional.dropout(self.gnet[i](h), p=0.5, training=True))
        return functional.sigmoid(self.gnet[self.l - 1](h))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn([self.L] + list(std.shape)).to(self.device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # Define the forward propagation function for the neural network.
    # Once defined, the backward propagation function will be autogeneration（autograd）
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='collective Variational Autoencoder')
    parser.add_argument('--batch', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('-m', '--maxiter', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--gpu', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dir', help='dataset directory', default='/Users/deepDR/dataset')
    parser.add_argument('--layer', nargs='+', help='number of neurons in each layer', type=int, default=[20])
    parser.add_argument('-L', type=int, default=1, help='number of samples')
    parser.add_argument('-N', help='number of recommended items', type=int, default=200)
    parser.add_argument('--learn_rate', help='learning rate', type=float, default=0.001)
    parser.add_argument('-a', '--alpha', help='parameter alpha', type=float, default=1)
    parser.add_argument('-b', '--beta', help='parameter beta', type=float, default=1)
    parser.add_argument('--rating', help='feed input as rating', action='store_true')
    parser.add_argument('--save', help='save model', action='store_true')
    parser.add_argument('--load', help='load model, 1 for cVAE_side_information_model and 2 for cVAE_rating_model', type=int, default=0)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # whether to ran with cuda
    args.device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print('dataset directory: ' + args.dir)
    directory = args.dir

    path = '{}/drug_disease.txt'.format(directory)
    print('train data path: ' + path)
    R = np.loadtxt(path)
    Rtensor = R.transpose()
    if args.rating:  # feed in with rating
        whole_positive_index = []
        whole_negative_index = []
        for i in range(np.shape(Rtensor)[0]):
            for j in range(np.shape(Rtensor)[1]):
                if int(Rtensor[i][j]) == 1:
                    whole_positive_index.append([i, j])
                elif int(Rtensor[i][j]) == 0:
                    whole_negative_index.append([i, j])
        negative_sample_index = np.random.choice(np.arange(len(whole_negative_index)),
                                                 size=1 * len(whole_positive_index), replace=False)
        # whole_negative_index=np.array(whole_negative_index)
        data_set = np.zeros((len(negative_sample_index) + len(whole_positive_index), 3), dtype=int)
        count = 0
        for i in whole_positive_index:
            data_set[count][0] = i[0]
            data_set[count][1] = i[1]
            data_set[count][2] = 1
            count += 1
        for i in negative_sample_index:
            data_set[count][0] = whole_negative_index[i][0]
            data_set[count][1] = whole_negative_index[i][1]
            data_set[count][2] = 0
            count += 1
        test_auc_fold = []
        test_aupr_fold = []
        rs = np.random.randint(0, 1000, 1)[0]
        kf = KFold(n_splits=5, shuffle=True, random_state=rs)

        results = {}
        for i, (train_index, test_index) in enumerate(kf.split(data_set[:, 2])):
            DTItrain, DTItest = data_set[train_index], data_set[test_index]
            Xtrain = np.zeros((np.shape(Rtensor)[0], np.shape(Rtensor)[1]))
            for ele in DTItrain:
                Xtrain[ele[0], ele[1]] = ele[2]
            Rtensor = torch.from_numpy(Xtrain.astype('float32')).to(args.device)
            args.d = Rtensor.shape[1]
            train_loader = DataLoader(Rtensor, args.batch, shuffle=True)
            loss_function = BCE_loss

            model = VAE(args).to(args.device)
            print(model)
            if args.load > 0:
                name = 'cVAE_rating_model' if args.load == 2 else 'cVAE_side_information_model'
                path = 'test_models/' + name
                for l in args.layer:
                    path += '_' + str(l)
                print('load model from path: ' + path)
                model.load_state_dict(torch.load(path))

            optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

            loss_list = []
            for epoch in range(1, args.maxiter + 1):
                loss = train(epoch)
                loss_list.append(loss)

            model.eval()
            score, _, _ = model(Rtensor)
            # print(score)
            print(score.detach().numpy().shape)
            Zscore = score.detach().numpy()
            # np.savetxt(f'Zscore{i}.txt', Zscore, delimiter='\t', fmt='%1.4f', newline='\n')

            # evaluation metrics:
            fold_result = {
              'auc': 0.,
              'fpr': None,
              'tpr': None,
              'aupr': 0.,
              'p': None,
              'r': None,
              'recall': []
            }

            # Calculating AUROC and AUPRC:
            pred_list = []
            ground_truth = []
            for ele in DTItrain:
                pred_list.append(Zscore[ele[0], ele[1]])
                ground_truth.append(ele[2])
            train_auc = roc_auc_score(ground_truth, pred_list)
            train_aupr = average_precision_score(ground_truth, pred_list)
            print('train auc aupr,', train_auc, train_aupr)
            pred_list = []
            ground_truth = []
            for ele in DTItest:
                pred_list.append(Zscore[ele[0], ele[1]])
                ground_truth.append(ele[2])
            test_auc = roc_auc_score(ground_truth, pred_list)
            fpr, tpr, _ = roc_curve(ground_truth, pred_list)

            test_aupr = average_precision_score(ground_truth, pred_list)
            p, r, _ = precision_recall_curve(ground_truth, pred_list)

            print('test auc aupr', test_auc, test_aupr)
            test_auc_fold.append(test_auc)
            fold_result['auc'] = test_auc
            fold_result['fpr'] = fpr
            fold_result['tpr'] = tpr

            test_aupr_fold.append(test_aupr)
            fold_result['aupr'] = test_aupr
            fold_result['p'] = p
            fold_result['r'] = r

            # model.train()

            # Calculating recall@K on score matrix (Disease-Drug):
            # score -> [1229 x 1519]
            score.detach_()
            score = score.squeeze(0)
            # print(score.shape)

            pos_DTItrain = DTItrain[DTItrain[:, -1] == 1]
            pos_DTItest = DTItest[DTItest[:, -1] == 1]

            score[pos_DTItrain[:, 0], pos_DTItrain[:, 1]] = 0
            _, rec_items = torch.topk(score, args.N, dim=1)
            run = sort2query(rec_items[:, 0:args.N])
            test = test2dict(pos_DTItest[:, 0:2])
            evaluator = Evaluator({'recall'})
            evaluator.evaluate(run, test)
            fold_recall_result = evaluator.show(
              ['recall_5', 'recall_10', 'recall_15', 'recall_20', 'recall_30', 'recall_100', 'recall_200']
            )

            for re in fold_recall_result:
              fold_result['recall'].append(float("{:.6f}".format(fold_recall_result[re])))
            
            res = PrettyTable()
            res.field_names = ['recall']
            res.add_row([fold_result['recall']])
            print(res)

            results[i + 1] = fold_result

        avg_auc = np.mean(test_auc_fold)
        avg_pr = np.mean(test_aupr_fold)
        print('\nmean auc aupr', avg_auc, avg_pr)
        
        # pprint(results)
        with open('results.pickle', 'wb') as handle:
          pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:  # feed in with side information
        path = 'drug_features.txt'
        print('feature data path: ' + path)
        fea = np.loadtxt(path)
        X = fea.transpose()
        X[X > 0] = 1
        args.d = X.shape[1]
        # X = normalize(X, axis=1)
        X = torch.from_numpy(X.astype('float32')).to(args.device)
        train_loader = DataLoader(X, args.batch, shuffle=True)
        loss_function = Guassian_loss

        model = VAE(args).to(args.device)
        if args.load > 0:
            name = 'cvae' if args.load == 2 else 'fvae'
            path = 'test_models/' + name
            for l in args.layer:
                path += '_' + str(l)
            print('load model from path: ' + path)
            model.load_state_dict(torch.load(path))

        optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)

        for epoch in range(1, args.maxiter + 1):
            train(epoch)

    if args.save:
        name = 'cVAE_rating_model' if args.rating else 'cVAE_side_information_model'
        path = 'test_models/' + name
        for l in args.layer:
            path += '_' + str(l)
        model.cpu()
        torch.save(model.state_dict(), path)

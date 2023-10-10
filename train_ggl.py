import argparse
import os.path as osp
import os
os.environ['TL_BACKEND'] = 'torch'
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np
import pickle
import tensorlayerx as tlx
from tensorlayerx.model import TrainOneStep, WithLoss
import numpy

from gammagl.utils import set_device
from gammagl.layers.conv import GCNConv
from my_utils_ggl import seed_everything,dropout_adj,get_dataset,generate_split,get_batch, get_A_bounds, lighten_color
from model_ggl import Encoder, Model, drop_feature
from eval_ggl import log_regression, MulticlassEvaluator


# def train_loss(model: Model, x, edge_index, epoch):
#     # model.set_train()
#     # optimizer.zero_grad()
#     #optimizer.apply_gradients(0)#?????不知道这么写对不对
#     edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
#     edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
#     # x_1 = drop_feature(x, drop_feature_rate_1)
#     # x_2 = drop_feature(x, drop_feature_rate_2)
#     x_1, x_2 = x, x
#     z1 = model(x_1, edge_index_1)
#     z2 = model(x_2, edge_index_2)
#     node_list = np.arange(z1.shape[0])
#     np.random.shuffle(node_list)
#     if args.dataset in ["PubMed", "Computers", "WikiCS"]:
#         batch_size = 4096
#     else:
#         batch_size = None

#     if batch_size is not None:
#         node_list_batch = get_batch(node_list, batch_size, epoch)

#     # nce loss
#     if batch_size is not None:
#         z11 = z1[node_list_batch]
#         z22 = z2[node_list_batch]
#         nce_loss = model.loss(z11, z22)
#     else:
#         nce_loss = model.loss(z1, z2)
#     # pot loss
#     if use_pot:
#         # get node_list_tmp, the nodes to calculate pot_loss
#         if pot_batch != -1:
#             if batch_size is None:
#                 node_list_tmp = get_batch(node_list, pot_batch, epoch)
#             else:
#                 node_list_tmp = get_batch(node_list_batch, pot_batch, epoch)
#         else:
#             # full pot batch
#             if batch_size == None:
#                 node_list_tmp = node_list
#             else:
#                 node_list_tmp = node_list_batch
#         z11 = z1[node_list_tmp]
#         z22 = z2[node_list_tmp]
#         global A_upper_1, A_upper_2, A_lower_1, A_lower_2
#         if A_upper_1 is None or A_upper_2 is None:
#             A_upper_1, A_lower_1 = get_A_bounds(args.dataset, drop_edge_rate_1)
#             A_upper_2, A_lower_2 = get_A_bounds(args.dataset, drop_edge_rate_2)
#         pot_loss_1 = model.pot_loss(z11, z22, data.x, data.edge_index, edge_index_1, local_changes=drop_edge_rate_1, 
#                                   node_list=node_list_tmp, A_upper=A_upper_1, A_lower=A_lower_1)
#         pot_loss_2 = model.pot_loss(z22, z11, data.x, data.edge_index, edge_index_2, local_changes=drop_edge_rate_2, 
#                                   node_list=node_list_tmp, A_upper=A_upper_2, A_lower=A_lower_2)
#         pot_loss = (pot_loss_1 + pot_loss_2) / 2
#         loss = (1 - kappa) * nce_loss + kappa * pot_loss
#     else:
#         loss = nce_loss
#     loss.backward()
#     optimizer.step()

#     return loss.item()

class train_loss(WithLoss):
    def __init__(self, model, drop_edge_rate_1, drop_edge_rate_2, use_pot=False, pot_batch=-1, kappa=0.5):
        super(train_loss, self).__init__(backbone=model, loss_fn=None)
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.use_pot = use_pot
        self.pot_batch = pot_batch
        self.kappa = kappa

    def forward(self, model, x, edge_index, epoch, data=None):
        edge_index_1 = dropout_adj(edge_index, p=self.drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(edge_index, p=self.drop_edge_rate_2)[0]
        x_1, x_2 = x, x
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        node_list = np.arange(z1.shape[0])
        np.random.shuffle(node_list)
        
        batch_size = 4096 if args.dataset in ["PubMed", "Computers", "WikiCS"] else None

        if batch_size is not None:
            node_list_batch = get_batch(node_list, batch_size, epoch)

        # nce loss
        if batch_size is not None:
            z11 = z1[node_list_batch]
            z22 = z2[node_list_batch]
            nce_loss = model.loss(z11, z22)
        else:
            nce_loss = model.loss(z1, z2)

        # pot loss
        if self.use_pot:
            # get node_list_tmp, the nodes to calculate pot_loss
            print(self.pot_batch)
            if self.pot_batch != -1:
                if batch_size is None:
                    node_list_tmp = get_batch(node_list, self.pot_batch, epoch)
                else:
                    node_list_tmp = get_batch(node_list_batch, self.pot_batch, epoch)
            else:
                # full pot batch
                if batch_size is None:
                    node_list_tmp = node_list
                else:
                    node_list_tmp = node_list_batch
                    
            z11 = tlx.gather(z1, tlx.convert_to_tensor(node_list_tmp))
            z22 = tlx.gather(z2, tlx.convert_to_tensor(node_list_tmp))
            # z11 = z1[tlx.convert_to_tensor(node_list_tmp)]
            # z22 = z2[tlx.convert_to_tensor(node_list_tmp)]

            global A_upper_1, A_upper_2, A_lower_1, A_lower_2
            if A_upper_1 is None or A_upper_2 is None:
                A_upper_1, A_lower_1 = get_A_bounds(args.dataset, self.drop_edge_rate_1)
                A_upper_2, A_lower_2 = get_A_bounds(args.dataset, self.drop_edge_rate_2)
            ###x index???
            pot_loss_1 = model.pot_loss(z11, z22, data.x, data.edge_index, edge_index_1, local_changes=self.drop_edge_rate_1, 
                                          node_list=node_list_tmp, A_upper=A_upper_1, A_lower=A_lower_1)
            pot_loss_2 = model.pot_loss(z22, z11, data.x, data.edge_index, edge_index_2, local_changes=self.drop_edge_rate_2, 
                                          node_list=node_list_tmp, A_upper=A_upper_2, A_lower=A_lower_2)
            pot_loss = (pot_loss_1 + pot_loss_2) / 2
            loss = (1 - self.kappa) * nce_loss + self.kappa * pot_loss
        else:
            loss = nce_loss

        return loss


def test(final=False):
    model.set_eval()
    z = model(data.x, data.edge_index)
    print(z,tlx.get_tensor_shape(z))
    if args.dataset == 'ogbn-arxiv':
        y_pred = z.argmax(dim=-1, keepdim=True)

        train_acc = evaluator.eval({
            'y_true': data.y[split['train']],
            'y_pred': y_pred[split['train']],
        })['acc']
        valid_acc = evaluator.eval({
            'y_true': data.y[split['valid']],
            'y_pred': y_pred[split['valid']],
        })['acc']
        test_acc = evaluator.eval({
            'y_true': data.y[split['test']],
            'y_pred': y_pred[split['test']],
        })['acc']
        return {
            "train_acc": train_acc,
            "valid_acc": valid_acc,
            "test_acc": test_acc
        }
    else:
        evaluator = MulticlassEvaluator()
        res = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=split)
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=5)
    parser.add_argument('--config', type=str, default='/home/lyq/gf/GRACE/config.yaml')
    parser.add_argument('--use_pot', default=False, action="store_true") # whether to use pot in loss
    parser.add_argument('--kappa', type=float, default=0.5)
    parser.add_argument('--pot_batch', type=int, default=-1)
    parser.add_argument('--drop_1', type=float, default=0.4)
    parser.add_argument('--drop_2', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.9) # temperature of nce loss
    parser.add_argument('--num_epochs',type=int,default=-1)
    parser.add_argument('--save_file', type=str, default=".")
    parser.add_argument('--seed', type=int, default=12345)

    args = parser.parse_args()

    # assert args.gpu_id in range(0, 9)
    # tlx.set_device(device='GPU',id=args.gpu_id)
    tlx.set_device('CPU')
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    # for hyperparameter tuning
    if args.drop_1 != -1:
        config['drop_edge_rate_1'] = args.drop_1
    if args.drop_2 != -1:
        config['drop_edge_rate_2'] = args.drop_2
    if args.tau != -1:
        config['tau'] = args.tau
    if args.num_epochs != -1:
        config['num_epochs'] = args.num_epochs
    print(args)
    print(config)

    # # set seed   先不解决
    seed_everything(args.seed)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': tlx.nn.ReLU, 'prelu': tlx.nn.PRelu()})[config['activation']]
    base_model = ({'GCNConv': GCNConv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    use_pot = args.use_pot
    kappa = args.kappa
    pot_batch = args.pot_batch

    path = osp.join(osp.expanduser('~'), 'datasets')
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    # data = data.to(device)
    
    # generate split
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        split = data.train_mask, data.val_mask, data.test_mask
        print("Public Split")
    else:
        split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
        print("Random Split")

    encoder = Encoder(dataset.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, dataset=args.dataset)
    train_weights = model.trainable_weights
    #Adam少参数
    optimizer = tlx.optimizers.Adam(lr=learning_rate, weight_decay=weight_decay)
    loss_func = train_loss(model, drop_edge_rate_1, drop_edge_rate_2, use_pot, pot_batch, kappa)
    train_one_step = TrainOneStep(loss_func, optimizer, train_weights)

    if use_pot:
        fp = osp.join(osp.expanduser('~/datasets'),f"bounds/{args.dataset}_{drop_edge_rate_1}_upper_lower.pkl")
        if os.path.exists(fp):
            pickle.load(fp)
            # A_upper_1_, A_lower_1_ = tlx.model.load_weights(fp)
        else:
            A_upper_1, A_lower_1 = get_A_bounds(args.dataset, drop_edge_rate_1)
            A_upper_2, A_lower_2 = get_A_bounds(args.dataset, drop_edge_rate_2)
    #timing        
    start = t()
    prev = start
    for epoch in range(1, num_epochs + 1):
        model.set_train()
        loss=train_one_step(model, data.x, data.edge_index, epoch ,data)
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        res = test()
        print(res)
        # if epoch % 100 == 0:
        prev = now

    print("=== Final ===")
    res = test(final=True)
    print(res)

    res_file = f"res/{args.dataset}_pot_temp.csv" if use_pot else f"res/{args.dataset}_base_temp.csv"
    if args.save_file == '.':
        f = open(res_file,"a+")
    else:
        f = open(args.save_file, "a+")
    res_str = f'{res["F1Mi"]:.4f}, {res["F1Ma"]:.4f}' 
    if use_pot:
        f.write(f'{config["drop_edge_rate_1"]}, {config["drop_edge_rate_2"]}, {config["tau"]}, {kappa}, '
                f'{res_str}\n')
    else:
        f.write(f'{config["drop_edge_rate_1"]}, {config["drop_edge_rate_2"]}, {config["tau"]}, '
                f'{res_str}\n')
    f.close()
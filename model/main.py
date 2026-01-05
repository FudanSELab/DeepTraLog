import torch
import time
import utils
import random
import os
import logging
import numpy as np
from model import GGNN

from torch_geometric.data import DataLoader
from torch.utils.data import Subset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from dataset import GraphDataset
from sklearn.manifold import TSNE


def save_model(model: 'nn.Module', path):
    logger.info(f"save model to {path}")
    torch.save(model.state_dict(), path)

def load_model_GGNN(path) -> GGNN:
    print(f"load GAE model form {path}")
    model = GGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

if __name__ == '__main__':
    epochs = 100
    batch_size = 32
    hidden_dim = 300

    num_layers = 3
    lr = 0.0001
    normal_classes = [0]
    abnormal_classes = [1]
    ratio = [6, 4]
    lr_milestones = [60]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # svdd param
    R = 0.0
    c = None
    R = torch.tensor(R, device=device)
    nu = 0.05
    objective = 'soft-boundary'
    # R updata after warm up epochs
    warm_up_n_epochs = 10

    # config dataset
    dataset = GraphDataset(root='../gae/data/')

    normal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), normal_classes)
    abnormal_idx = utils.get_target_label_idx(dataset.data.y.clone().data.cpu().numpy(), abnormal_classes)

    random.shuffle(abnormal_idx)
    random.shuffle(normal_idx)

    # train_normal = normal_idx[:65490]
    # test_normal = normal_idx[70090:]
    # val_normal = normal_idx[65490:70090]

    train_normal = normal_idx[:int(len(normal_idx) * 0.6)]
    test_normal = normal_idx[int(len(normal_idx) * 0.7):]
    val_normal = normal_idx[int(len(normal_idx) * 0.6):int(len(normal_idx) * 0.7)]
    train_abnormal = abnormal_idx[:int(len(abnormal_idx) * 0)]
    test_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):]
    val_abnormal = abnormal_idx[int(len(abnormal_idx) * 0):int(len(abnormal_idx) * 0)]

    train_dataset = Subset(dataset, train_normal + train_abnormal)
    test_dataset = Subset(dataset, test_abnormal + test_normal)
    val_dataset = Subset(dataset, val_normal + val_abnormal)

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up logging
    xp_path = f"./output/newR" + time.strftime("%Y-%m-%d_%Hh-%Mm-%Ss", time.localtime())
    if not os.path.exists(xp_path):
        os.makedirs(xp_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    load_model_flag = False
    save_model_flag = True
    save_model_path = xp_path + "/save_" + str(time.time()).split(".")[0] + ".model"
    load_model_path = "./output/2021-08-20_16h-52m-04s/save_1629449524.model"

    # train
    model = GGNN(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    c = utils.init_center_c(train_loader, model, device)

    logger.info(
        "************************************** Start Train ****************************************************")
    logger.info(
        "batch_size=%s, epochs=%s, ratio=%s, device=%s" % (
            batch_size, epochs, ratio,  device))
    logger.info(
        "objective=%s, lr=%s, num_layers=%s, nu=%s, c=%s" % (
            objective, lr, num_layers, nu, c))
    logger.info(
        "train_normal=%s, test_normal=%s, val_normal=%s, train_abnormal=%s, test_abnormal=%s, val_abnormal=%s" % (
            len(train_normal), len(test_normal), len(val_normal), len(train_abnormal), len(test_abnormal), len(val_abnormal)))
    logger.info(
        "new r version")

    train_start_time = time.time()
    if load_model_flag:
        print(type(model))
        if isinstance(model, GGNN):
            model = load_model_GGNN(load_model_path)
        else:
            raise Exception(f"Load Model Failed, No Such model {type(model)}")

    else:

        for epoch in range(epochs):
            start_time = time.time()

            model.train()

            total_loss = 0.0
            total = 0

            # train
            dist_set = []
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output, _ = model(batch)
                dist = torch.sum((output - c) ** 2, dim=1)
                if objective == 'soft-boundary':
                    scores = dist - R ** 2
                    loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                total += 1

                dist_set.append(dist)

            if (objective == 'soft-boundary') and (epoch >= warm_up_n_epochs):
                R.data = torch.tensor(utils.get_radius(torch.cat(tuple(dist_set), 0), nu), device=device)

            scheduler.step()
            total_loss = total_loss / total
            end_time = time.time()
            logger.info('Epoch: %3d/%3d, Train Loss: %.10f, Time cost: %.5f, Learning Rate: %.5f' % (
                epoch + 1, epochs, total_loss, end_time - start_time, float(scheduler.get_last_lr()[0])))

    train_time = time.time() - train_start_time
    logger.info('Training time: %.3f' % train_time)
    logger.info("last r is %s" % R.data)

    # eval
    if save_model_flag:
        save_model(model, save_model_path)


    # calculate threshold
    start_time = time.time()
    threshold_score = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output, _ = model(batch)
            dist = torch.sum((output - c) ** 2, dim=1)
            if objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            # Save triples of (idx, label, score) in a list
            threshold_score += list(zip(scores.cpu().data.numpy().tolist(),
                                        batch.trace_id))

    test_time = time.time() - start_time
    logger.info('Calculate threshold time: %.3f' % test_time)

    train_scores, _ = zip(*threshold_score)
    train_scores = np.array(train_scores)

    # test
    start_time = time.time()
    idx_label_score = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch.y
            batch = batch.to(device)
            output, attention_scores = model(batch)
            dist = torch.sum((output - c) ** 2, dim=1)
            if objective == 'soft-boundary':
                scores = dist - R ** 2
            else:
                scores = dist

            nodes_scores = []
            for ng in range(batch.batch.max() + 1):
                nodes_attention_scores = attention_scores[batch.batch==ng]
                nodes_attention_scores = nodes_attention_scores.cpu().data.numpy()
                nodes_attention_scores = np.array2string(nodes_attention_scores, formatter={'float_kind': lambda x: "%.10f" % x})
                nodes_scores.append(nodes_attention_scores)

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                        scores.cpu().data.numpy().tolist(),
                                        batch.trace_id,
                                        batch.error_trace_type,
                                        nodes_scores,
                                        output.cpu().data.numpy().tolist()))

    test_time = time.time() - start_time
    logger.info('Testing time: %.3f' % test_time)

    test_scores = idx_label_score

    # Compute AUC
    labels, scores, trace_ids, error_types, _, graph_embedding = zip(*idx_label_score)
    labels = np.array(labels)
    # for label in np.nditer(labels, op_flags=['readwrite']):
    #     if label == 0:
    #         label[...] = 1
    #     elif label == 1:
    #         label[...] = 0
    scores = np.array(scores)
    trace_ids = np.array(trace_ids)

    test_auc = roc_auc_score(labels, scores)
    logger.info('Test set AUC: {:.2f}%'.format(100. * test_auc))


    # 0
    pred_labels = scores.copy()
    pred_labels[pred_labels > 0] = 1
    pred_labels[pred_labels < 0] = 0

    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)

    logger.info('Test 0 Precision: {:.2f}%'.format(100. * precision))
    logger.info('Test 0 Recall: {:.2f}%'.format(100. * recall))
    logger.info('Test 0 F1: {:.2f}%'.format(100. * f1))


    utils.write_csv_file(xp_path, 'result.csv',
                         ("label", "score", "trace_id", "error_type", "nodes_score", "graph_embedding"),
                         idx_label_score)

    # tsne = TSNE()
    # x = tsne.fit_transform(graph_embedding)
    # plt.scatter(x[:, 0], x[:, 1], c=labels, marker='o', s=40, cmap=plt.cm.Spectral)
    # plt.savefig(xp_path+'/t-sne-test.jpg')
    #
    #
    # low_dim = list(zip(x, labels))
    # utils.write_csv_file(xp_path, 'low_dim.csv', ("graph", "labels"), low_dim)

    logger.info('Finished testing.')









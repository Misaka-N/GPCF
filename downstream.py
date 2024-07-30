import dgl
import torch
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm
from dgl.dataloading import MultiLayerNeighborSampler
import torch.optim as optim
import torch.nn.functional as F
from models.encoder import GCN
from models.prompt import PromptComponent, AnswerFunc
from utils.args import get_downstream_args
from utils.tools import set_random
from utils.dataloader import pretrain_dataloader
from utils.tools import EarlyStopping, label_smoothing
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, average_precision_score
import warnings

def divide_dataset(g, num_classes, shot):
    labels = g.ndata['label'].numpy() 
    train_idx, val_idx, test_idx = [], [], []

    for class_id in range(num_classes):
        class_nodes = np.where(labels == class_id)[0]

        np.random.shuffle(class_nodes)

        if len(class_nodes) < shot:
            warnings.warn(f'the node num of class{class_id} ({len(class_nodes)}) is less than {shot}')

        train_nodes = class_nodes[:shot]
        remaining_nodes = class_nodes[shot:]
        split_point = len(remaining_nodes) // 2
        val_nodes = remaining_nodes[:split_point]
        test_nodes = remaining_nodes[split_point:]

        train_idx.extend(train_nodes)
        val_idx.extend(val_nodes)
        test_idx.extend(test_nodes)

    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)
    test_idx = torch.tensor(test_idx, dtype=torch.long)

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    args = get_downstream_args()

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda")
        set_random(args.seed, True)
    else:
        print("CUDA is not available")
        device = torch.device("cpu")
        set_random(args.seed, False)

    
    # Get downstream dataset
    print("---Downloading dataset: " + args.dataset + "---")
    g, dataname, num_classes = pretrain_dataloader(input_dim=args.input_dim, dataset=args.dataset)
    print("---Divide dataset: " + args.dataset + "---")
    train_idx, val_idx, test_idx = divide_dataset(g, num_classes, args.shot)
    neighbor_sampler = MultiLayerNeighborSampler([args.neighbor_num] * args.neighbor_layer)
    train_dataloader = dgl.dataloading.DataLoader(g, train_idx, neighbor_sampler, device=device, use_ddp=False, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_dataloader = dgl.dataloading.DataLoader(g, val_idx, neighbor_sampler, device=device, use_ddp=False, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_dataloader = dgl.dataloading.DataLoader(g, test_idx, neighbor_sampler, device=device, use_ddp=False, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    # Get prompt
    print("---Creating new prompt component---")
    prompt = PromptComponent(prompt_num=args.prompt_num, prompt_dim=args.prompt_dim, input_dim=args.input_dim).to(device)

    # Answering head
    answering = AnswerFunc(input_dim=2 * args.prompt_dim, hidden_dim=args.prompt_dim, output_dim=num_classes).to(device)

    # Downstream tasks
    print("---Dealing with downstream task---")
    gnn = GCN(input_dim=args.input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.output_dim,
                gnn_layer=args.gnn_layer,
                ).to(device)
    gnn.conv_layers.load_state_dict(torch.load(args.pretrained_model))

    optimizer = optim.Adam(list(prompt.parameters()) + list(answering.parameters()), lr=args.lr, weight_decay=args.decay)
    early_stopper = EarlyStopping(path=args.prompt_path, patience=args.patience, min_delta=0)
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    ce_loss = nn.CrossEntropyLoss()

    # Train
    for epoch in range(args.max_epoches):
        gnn.eval()
        prompt.train()
        answering.train()

        tot_loss, tot_acc = [], []
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            blocks = [block.to(device) for block in blocks]
            final_block = gnn(g, blocks)

            prompt_emb = prompt(g, final_block)

            predict_ans = answering(prompt_emb).cpu()
            true_labels = g.ndata['label'][seeds].cpu()

            if num_classes == 2: 
                train_loss = ce_loss(F.softmax(predict_ans, dim=-1), true_labels) / predict_ans.shape[0]
            else:
                soft_label = label_smoothing(true_labels, args.label_smoothing, num_classes)
                train_loss = kl_loss(F.log_softmax(predict_ans, dim=-1), soft_label) / predict_ans.shape[0]

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            _, predict = torch.max(predict_ans, dim=1)
            accuracy = accuracy_score(true_labels.cpu().numpy(), predict.cpu().numpy())

            tot_loss.append(train_loss)
            tot_acc.append(accuracy)

        avg_loss = sum(tot_loss) / len(tot_loss)
        avg_acc = sum(tot_acc) / len(tot_acc)

        print("Epoch: {} | Loss: {:.4f} | ACC: {:.4f}".format(epoch, avg_loss, avg_acc))

        # Evaluation
        if (epoch + 1) % 10 == 0:
            # Set the models to evaluation mode
            gnn.eval()
            prompt.eval()
            answering.eval()

            predict, pred, label = [], [], []
            for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                blocks = [block.to(device) for block in blocks]
                final_block = gnn(g, blocks)

                prompt_emb = prompt(g, final_block)

                predict_ans = answering(prompt_emb).cpu()
                
                predict.extend(predict_ans.argmax(dim=1).tolist())
                predict_ans = F.softmax(predict_ans, dim=-1)
                pred.extend(predict_ans.detach().numpy().tolist())
                label.extend(g.ndata['label'][seeds].cpu().tolist())

            pred = torch.tensor(pred)
            predict = torch.tensor(predict)
            label = torch.tensor(label)

            accuracy = accuracy_score(label, predict)
            recall = recall_score(label, predict, average='macro')
            f1 = f1_score(label, predict, average='macro')

            if num_classes == 2:
                auc = roc_auc_score(label, pred[ :,1]) # for binary classification
                ap = average_precision_score(label, pred[ :,1]) # for binary classification
            else:
                auc = roc_auc_score(label, pred, multi_class='ovr')
                ap = average_precision_score(label, pred)

            early_stopper((prompt, answering), -(accuracy + auc + f1))
            if early_stopper.early_stop:
                print("Stopping training...")
                break

            print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch + 1, accuracy, auc, f1, recall, ap))

    # test on the best model
    print("Evaluating on the best model...")
    gnn.eval()
    prompt, answering = torch.load(args.prompt_path)
    torch.save(prompt, args.component_path) # save the component for Plus Version
    prompt.eval()
    answering.eval()

    predict, pred, label = [], [], []
    for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
        blocks = [block.to(device) for block in blocks]
        final_block = gnn(g, blocks)

        prompt_emb = prompt(g, final_block)

        predict_ans = answering(prompt_emb).cpu()
        
        predict.extend(predict_ans.argmax(dim=1).tolist())
        predict_ans = F.softmax(predict_ans, dim=-1)
        pred.extend(predict_ans.detach().numpy().tolist())
        label.extend(g.ndata['label'][seeds].cpu().tolist())

    pred = torch.tensor(pred)
    predict = torch.tensor(predict)
    label = torch.tensor(label)

    accuracy = accuracy_score(label, predict)
    recall = recall_score(label, predict, average='macro')
    f1 = f1_score(label, predict, average='macro')

    if num_classes == 2:
        auc = roc_auc_score(label, pred[ :,1]) # for binary classification
        ap = average_precision_score(label, pred[ :,1]) # for binary classification
    else:
        auc = roc_auc_score(label, pred, multi_class='ovr')
        ap = average_precision_score(label, pred)

    print("Epoch: {} | ACC: {:.4f} | AUC: {:.4f} | F1: {:.4f} | Recall : {:.4f} | AP: {:.4f}".format(epoch + 1, accuracy, auc, f1, recall, ap))
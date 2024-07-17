import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets
    parser.add_argument("--dataset", type=str, nargs='+', default=['Amazon_Photo','Amazon_Computer','Amazon_Fraud'],
                        help="Datasets used for pretrain")
    parser.add_argument("--sample_shots", type=int, default=200, help="node shots for pretrain")

    # Pretrained model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--projector_layer", type=int, default=2, help="layer num for projector")
    parser.add_argument("--input_dim", type=int, default=100, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=100, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--path", type=str, default="pretrained_gnn/Amazon.pth", help="model saving path")

    # Pretrain Process
    parser.add_argument("--neighbor_num", type=int, default=100, help="neighbor node num for neighbor sampler")
    parser.add_argument("--neighbor_layer", type=int, default=2, help="layer num for neighbor sampler")
    parser.add_argument("--batch_size", type=int, default=20, help="node num for each batch")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for pretraining")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for pretraining")
    parser.add_argument("--max_epoches", type=int, default=500, help="max epoches for pretraining")
    parser.add_argument("--moco_bias", type=float, default=0.95, help="bias for MoCo update")
    parser.add_argument("--adapt_step", type=int, default=2, help="model adapt steps for meta-learning")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for similarity calculation")
    parser.add_argument("--weight", type=float, default=0.5, help="weight bias for local task contrastive loss")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=40, help="random seed")
    parser.add_argument("--patience", type=int, default=20, help="early stop steps")
    
    args = parser.parse_args()
    return args



def get_downstream_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets
    parser.add_argument("--task", type=str, default='node', help="if node level tasks")
    parser.add_argument("--dataset", type=str, default="Amazon_Photo",help="Datasets used for downstream tasks")
    parser.add_argument("--pretrained_model", type=str, default="pretrained_gnn/Amazon.pth", help="pretrained model path")
    parser.add_argument("--shot", type=int, default=100, help="shot for few-shot learning")
    parser.add_argument("--neighbor_num", type=int, default=100, help="neighbor node num for neighbor sampler")
    parser.add_argument("--neighbor_layer", type=int, default=2, help="neighbor layer num for neighbor sampler")

    # Model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--input_dim", type=int, default=100, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=100, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=100, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--batch_size", type=int, default=10, help="batch size for nodes")

    # Prompt
    parser.add_argument("--prompt_num", type=int, default=5, help="prompt num for each component")
    parser.add_argument("--prompt_dim", type=int, default=100, help="dimension of prompt, should be same as hidden_dim")
    parser.add_argument("--prompt_path", type=str, default='downstream_model/prompt_pools.pth', help="prompt pool and head saving path")
    parser.add_argument("--component_path", type=str, default='downstream_model/Amazon_prompt.pth', help="prompt saving path")

    # Downstream Tasks
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for downstream training")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for downstream training")
    parser.add_argument("--max_epoches", type=int, default=200, help="max epoches for downstream training")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label_smoothing for over-fitting")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=1142, help="random seed")
    parser.add_argument("--patience", type=int, default=10, help="early stop steps")

    args = parser.parse_args()
    return args
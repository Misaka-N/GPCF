import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets
    parser.add_argument("--dataset", type=str, nargs='+', default=['Amazon_Fraud', 'Yelp_Fraud'],
                        help="Datasets used for pretrain")
    parser.add_argument("--sample_shots", type=int, default=500, help="node shots for pretrain")

    # Pretrained model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--projector_layer", type=int, default=2, help="layer num for projector")
    parser.add_argument("--input_dim", type=int, default=25, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=128, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--path", type=str, default="pretrained_gnn/NewFraud300.pth", help="model saving path")

    # Pretrain Process
    parser.add_argument("--neighbor_num", type=int, default=500, help="neighbor node num for neighbor sampler")
    parser.add_argument("--neighbor_layer", type=int, default=2, help="layer num for neighbor sampler")
    parser.add_argument("--batch_size", type=int, default=500, help="node num for each batch")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for pretraining")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for pretraining")
    parser.add_argument("--max_epoches", type=int, default=500, help="max epoches for pretraining")
    parser.add_argument("--moco_bias", type=float, default=0.95, help="bias for MoCo update")
    parser.add_argument("--adapt_step", type=int, default=2, help="model adapt steps for meta-learning")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for similarity calculation")
    parser.add_argument("--weight", type=float, default=0.5, help="weight bias for restoration loss")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=114514, help="random seed")
    parser.add_argument("--patience", type=int, default=10, help="early stop steps")
    
    args = parser.parse_args()
    return args



def get_downstream_args():
    parser = argparse.ArgumentParser(description='Args for Pretrain')

    # Datasets
    parser.add_argument("--dataset", type=str, default="Amazon_Fraud",help="Datasets used for downstream tasks")
    parser.add_argument("--pretrained_model", type=str, default="pretrained_gnn/NewFraud300.pth", help="pretrained model path")
    parser.add_argument("--shot", type=int, default=100, help="shot for few-shot learning")
    parser.add_argument("--neighbor_num", type=int, default=500, help="neighbor node num for neighbor sampler")
    parser.add_argument("--neighbor_layer", type=int, default=2, help="neighbor layer num for neighbor sampler")

    # Model
    parser.add_argument("--gnn_layer", type=int, default=2, help="layer num for gnn")
    parser.add_argument("--input_dim", type=int, default=25, help="input dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden dimension")
    parser.add_argument("--output_dim", type=int, default=128, help="output dimension(also dimension of projector and answering fuction)")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for nodes")

    # Prompt
    parser.add_argument("--prompt_num", type=int, default=8, help="prompt num for each component")
    parser.add_argument("--prompt_dim", type=int, default=128, help="dimension of prompt, should be same as hidden_dim")
    parser.add_argument("--prompt_path", type=str, default='downstream_model/prompt.pth', help="prompt pool and head saving path")
    parser.add_argument("--component_path", type=str, default='downstream_model/Photo_prompt.pth', help="prompt saving path")

    # Downstream Tasks
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate for downstream training")
    parser.add_argument("--decay", type=float, default=0.0001, help="weight decay for downstream training")
    parser.add_argument("--max_epoches", type=int, default=500, help="max epoches for downstream training")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label_smoothing for over-fitting")

    # Trainging enviorment
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use, -1 for CPU")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--patience", type=int, default=5, help="early stop steps")

    args = parser.parse_args()
    return args
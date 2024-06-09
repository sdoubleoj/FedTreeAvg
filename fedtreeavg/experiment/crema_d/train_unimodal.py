import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'dataloader'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'trainers'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'constants'))

import constants
from server_trainer import Server
from unimodal_models import RNNClassifier, ConvRNNClassifier
from dataload_manager import DataloadManager

# trainer
from fed_rs_trainer import ClientFedRS
from fed_avg_trainer import ClientFedAvg
from scaffold_trainer import ClientScaffold

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():

    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")

    parser = argparse.ArgumentParser(description='FedMultimoda experiments')
    parser.add_argument(
        '--data_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )
    
    parser.add_argument(
        '--audio_feat', 
        default='mfcc',
        type=str,
        help="audio feature name",
    )
    
    parser.add_argument(
        '--video_feat', 
        default='mobilenet_v2',
        type=str,
        help="video feature name",
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        '--mu',
        type=float, 
        default=0.001,
        help='Fed prox term'
    )


    parser.add_argument(
        '--global_learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--att', 
        type=bool, 
        default=False,
        help='self attention applied or not'
    )
    
    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )
    
    parser.add_argument(
        '--att_name',
        type=str, 
        default='multihead',
        help='attention name'
    )
    
    parser.add_argument(
        '--hid_size',
        type=int, 
        default=64,
        help='RNN hidden size dim'
    )

    parser.add_argument(
        '--sample_rate', 
        default=0.1,
        type=float,
        help="client sample rate",
    )
    
    parser.add_argument(
        '--num_epochs', 
        default=300,
        type=int,
        help="total training rounds",
    )
    
    parser.add_argument(
        '--local_epochs', 
        default=1,
        type=int,
        help="local epochs",
    )
    
    parser.add_argument(
        '--optimizer', 
        default='sgd',
        type=str,
        help="optimizer",
    )
    
    parser.add_argument(
        '--fed_alg', 
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )
    
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help="training batch size",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label'
    )
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong'
    )
    
    parser.add_argument(
        '--metric', 
        type=str, 
        default='uar',
        help='evaluation metric: uar/acc/f1'
    )

    parser.add_argument(
        '--modality', 
        type=str, 
        default='audio',
        help='modality type'
    )
    
    parser.add_argument(
        "--dataset", 
        default="crema_d"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # argument parser
    args = parse_args()

    # data manager
    dm = DataloadManager(args)
    dm.get_simulation_setting()
    
    # find device
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    save_result_dict = dict()

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS

    # We perform 5 fold experiments
    for fold_idx in range(1, 6):
        # load simulation feature
        dm.load_sim_dict(
            fold_idx=fold_idx
        )
        # load all data
        dm.get_client_ids(
            fold_idx=fold_idx
        )

        # set dataloaders
        dataloader_dict = dict()
        logging.info('Loading Data')
        for client_id in tqdm(dm.client_ids):

            if args.modality == 'audio':
                data_dict = dm.load_audio_feat(
                    client_id=client_id, 
                    fold_idx=fold_idx
                )
            elif args.modality == 'video':
                data_dict = dm.load_video_feat(
                    client_id=client_id, 
                    fold_idx=fold_idx
                )
            dm.get_label_dist(
                data_dict, 
                client_id
            )
            shuffle = False if client_id in ['dev', 'test'] else True
            client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)
            dataloader_dict[client_id] = dm.set_unimodal_dataloader(
                data_dict,
                shuffle=shuffle
            )
        
        # number of clients
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)
        # set seeds
        set_seed(8)
        # loss function
        criterion = nn.NLLLoss().to(device)
        # Define the model
        if args.modality == 'audio':
            global_model = ConvRNNClassifier(
                num_classes=constants.num_class_dict[args.dataset],
                input_dim=constants.feature_len_dict["mfcc"],
                d_hid=args.hid_size,
                en_att=args.att,
                att_name=args.att_name
            )
        elif args.modality == 'video':
            global_model = RNNClassifier(
                num_classes=constants.num_class_dict[args.dataset],
                input_dim=constants.feature_len_dict["mobilenet_v2"],
                d_hid=args.hid_size,
                en_att=args.att,
                att_name=args.att_name
            )
        global_model = global_model.to(device)

        # initialize server
        server = Server(
            args, 
            global_model, 
            device=device, 
            criterion=criterion,
            client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients, 
            sample_rate=args.sample_rate
        )

        # save json path
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result',
            args.fed_alg,
            args.dataset, 
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)
        
        server.save_json_file(
            dm.label_dist_dict, 
            save_json_path.joinpath('label.json')
        )

        # set seeds again
        set_seed(8)

        # Training steps
        for epoch in range(int(args.num_epochs)):
            # define list varibles that saves the weights, loss, num_sample, etc.
            server.initialize_epoch_updates(epoch)
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                # Local training
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue
                # initialize client object
                client = Client(
                    args, 
                    device, 
                    criterion, 
                    dataloader, 
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset]
                )

                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control), 
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()

                    # server append updates
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result,
                        delta_control=copy.deepcopy(client.delta_control)
                    )
                else:
                    client.update_weights()
                    # server append updates
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result
                    )
                del client
            
            # logging skip client
            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            
            # 2. aggregate, load new global weights
            if len(server.num_samples_list) == 0: continue
            server.average_weights()

            logging.info('---------------------------------------------------------')
            server.log_classification_result(
                data_split='train', 
                metric=args.metric
            )
            with torch.no_grad():
                # 3. Perform the validation on dev set
                server.inference(dataloader_dict['dev'])
                server.result_dict[epoch]['dev'] = server.result
                server.log_classification_result(
                    data_split='dev', 
                    metric=args.metric
                )
                # 4. Perform the test on holdout set
                server.inference(dataloader_dict['test'])
                server.result_dict[epoch]['test'] = server.result
                server.log_classification_result(
                    data_split='test', 
                    metric=args.metric
                )
            
            logging.info('---------------------------------------------------------')
            server.log_epoch_result(metric=args.metric)
            logging.info('---------------------------------------------------------')

        # Performance save code
        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()
        
        # output to results
        server.save_json_file(
            save_result_dict, 
            save_json_path.joinpath('result.json')
        )

    # Calculate the average of the 5-fold experiments
    save_result_dict['average'] = dict()
    for metric in ['uar', 'acc', 'top5_acc']:
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]: continue
            result_list.append(save_result_dict[key][metric])
        save_result_dict['average'][metric] = np.nanmean(result_list)
    
    # dump the dictionary
    server.save_json_file(
        save_result_dict, 
        save_json_path.joinpath('result.json')
    )


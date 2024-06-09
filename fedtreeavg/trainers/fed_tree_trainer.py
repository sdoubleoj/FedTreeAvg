import collections
import numpy as np
import pandas as pd
import copy, pdb, time, warnings, torch
import random

from torch import nn
from torch.utils import data
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, recall_score

# import optimizer
from .optimizer import FedProxOptimizer

warnings.filterwarnings('ignore')
from .evaluation import EvalMetric

class ClientFedTree(object):
    def __init__(
        self, 
        args, 
        device, 
        criterion, 
        dataloader, 
        model, 
        label_dict=None,
        num_class=None
    ):
        self.args = args
        self.model = model
        self.device = device
        self.criterion = criterion
        self.dataloader = dataloader
        self.multilabel = True if args.dataset == 'ptb-xl' else False
        self.result = None
        self.test_true = None
        self.test_pred = None
        self.train_groundtruth = None
        self.eval = None
        
    def get_parameters(self):
        # Return model parameters
        return self.model.state_dict()
    
    def get_model_result(self):
        # Return model results
        return self.result
    
    def get_test_true(self):
        # Return test labels
        return self.test_true
    
    def get_test_pred(self):
        # Return test predictions
        return self.test_pred
    
    def get_train_groundtruth(self):
        # Return groundtruth used for training
        return self.train_groundtruth

    def update_weights(self):
        # Set model to training mode
        self.model.train()
        
        # Initialize evaluation metric
        self.eval = EvalMetric(self.multilabel)
        
        # Choose optimizer based on federated algorithm
        if self.args.fed_alg in ['fed_avg', 'fed_opt']:
            optimizer = torch.optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5
            )
        else:
            optimizer = FedProxOptimizer(
                self.model.parameters(), 
                lr=self.args.learning_rate,
                momentum=0.9,
                weight_decay=1e-5,
                mu=self.args.mu
            )
        
        # Save a copy of the last global model
        last_global_model = copy.deepcopy(self.model)
        
        # Train model for local epochs
        for iter in range(int(self.args.local_epochs)):
            for batch_idx, batch_data in enumerate(self.dataloader):
                if self.args.dataset == 'extrasensory' and batch_idx > 20: continue
                self.model.zero_grad()
                optimizer.zero_grad()
                if self.args.modality == "multimodal":
                    x_a, x_b, l_a, l_b, y = batch_data
                    x_a, x_b, y = x_a.to(self.device), x_b.to(self.device), y.to(self.device)
                    l_a, l_b = l_a.to(self.device), l_b.to(self.device)
                    
                    # Forward pass
                    outputs, _ = self.model(
                        x_a.float(), x_b.float(), l_a, l_b
                    )
                else:
                    x, l, y = batch_data
                    x, l, y = x.to(self.device), l.to(self.device), y.to(self.device)
                    
                    # Forward pass
                    outputs, _ = self.model(
                        x.float(), l
                    )
                
                if not self.multilabel: 
                    outputs = torch.log_softmax(outputs, dim=1)
                    
                # Compute loss
                loss = self.criterion(outputs, y)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    10.0
                )
                
                # Update model parameters
                optimizer.step()
                
                # Save results
                if not self.multilabel: 
                    self.eval.append_classification_results(
                        y, 
                        outputs, 
                        loss
                    )
                else:
                    self.eval.append_multilabel_results(
                        y, 
                        outputs, 
                        loss
                    )
        
        # Summarize epoch train results
        if not self.multilabel:
            self.result = self.eval.classification_summary()
        else:
            self.result = self.eval.multilabel_summary()


class FederatedLearning:
    def __init__(self, clients):
        self.clients = clients

    def train_clients(self):
        # Train each client
        for client in self.clients:
            client.update_weights()

    def aggregate_parameters(self):
        num_clients = len(self.clients)
        # Create tree nodes for each client
        tree_nodes = [client.get_parameters() for client in self.clients]

        while len(tree_nodes) > 1:
            new_tree_nodes = []
            random.shuffle(tree_nodes)
            
            for i in range(0, len(tree_nodes), 2):
                if i + 1 < len(tree_nodes):
                    client1, client2 = tree_nodes[i], tree_nodes[i + 1]
                    
                    # Evaluate each client's accuracy
                    client1_accuracy = self.evaluate_client_accuracy(self.clients[i])
                    client2_accuracy = self.evaluate_client_accuracy(self.clients[i + 1])
                    
                    total_accuracy = client1_accuracy + client2_accuracy
                    weight1 = client1_accuracy / total_accuracy
                    weight2 = client2_accuracy / total_accuracy

                    # Compute weighted average of parameters
                    aggregated_parameters = {
                        key: weight1 * client1[key] + weight2 * client2[key] 
                        for key in client1.keys()
                    }
                    
                    # Assign weighted average to next node
                    new_tree_nodes.append(aggregated_parameters)
                else:
                    # If odd number of nodes, pass the last one to the next level
                    new_tree_nodes.append(tree_nodes[i])
                    
            tree_nodes = new_tree_nodes
        
        # Return final aggregated parameters
        return tree_nodes[0]

    def evaluate_client_accuracy(self, client):
        # Implement a method to evaluate the client's accuracy
        # This is a placeholder implementation
        true_labels = client.get_test_true()
        predictions = client.get_test_pred()
        accuracy = accuracy_score(true_labels, predictions)
        return accuracy

    def update_global_model(self, global_model):
        # Aggregate parameters and update the global model
        global_parameters = self.aggregate_parameters()
        global_model.load_state_dict(global_parameters)
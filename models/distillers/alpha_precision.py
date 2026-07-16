from __future__ import absolute_import, division, print_function
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import scipy
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# from representations.OneClass import * 
# from metrics.evaluation import *

# Dependency graph
# toy_metric_evaluation.ipynb
#   + representations.OneClass:
#       + representations.networks
#   + metrics.evaluation
#   + from metrics.prdc import compute_prdc

# torch.manual_seed(1) 
# representations.networks #####################################################
ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(),
                   "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(), 
                   "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(), 
                   "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(), 
                   "SELU": torch.nn.SELU(), 
                   "GLU": torch.nn.GLU(), 
                   "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(), 
                   "Softplus": torch.nn.Softplus()}

def build_network(network_name, params):
    if network_name=="feedforward":
        net = feedforward_network(params)
    return net
def feedforward_network(params):
    """Architecture for a Feedforward Neural Network
	
    Args:
        ::params::
        ::params["input_dim"]::
        ::params[""rep_dim""]::
        ::params["num_hidden"]::
        ::params["activation"]::
        ::params["num_layers"]::
        ::params["dropout_prob"]::
        ::params["dropout_active"]:: 
        ::params["LossFn"]::
	
    Returns:
        ::_architecture::
    """
    modules          = []
    if params["dropout_active"]: 
        modules.append(torch.nn.Dropout(p=params["dropout_prob"]))
    # Input layer    
    modules.append(torch.nn.Linear(params["input_dim"], params["num_hidden"],bias=False))
    modules.append(ACTIVATION_DICT[params["activation"]])
    # Intermediate layers
    for u in range(params["num_layers"] - 1):
        if params["dropout_active"]:
            modules.append(torch.nn.Dropout(p=params["dropout_prob"]))
        modules.append(torch.nn.Linear(params["num_hidden"], params["num_hidden"],
                                       bias=False))
        modules.append(ACTIVATION_DICT[params["activation"]])
    # Output layer    
    modules.append(torch.nn.Linear(params["num_hidden"], params["rep_dim"],bias=False))
    _architecture    = nn.Sequential(*modules)
    return _architecture


# representations.OneClass #####################################################
def OneClassLoss(outputs, c): 
    dist   = torch.sum((outputs - c) ** 2, dim=1)
    loss   = torch.mean(dist)
    return loss
def SoftBoundaryLoss(outputs, R, c, nu):
    dist   = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist - R ** 2
    loss   = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    scores = dist 
    loss   = (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss

LossFns    = dict({"OneClass": OneClassLoss, "SoftBoundary": SoftBoundaryLoss})

class BaseNet(nn.Module):
    """Base class for all neural networks."""
    def __init__(self):
        super().__init__()
        self.logger  = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer
    def forward(self, *input):
        """Forward pass logic
        :return: Network output
        """
        raise NotImplementedError
    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params         = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

def get_radius(dist:torch.Tensor, nu:float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.float().numpy()), 1 - nu)

class OneClassLayer(BaseNet):
    def __init__(self, params=None, hyperparams=None):
        super().__init__()
        # set all representation parameters - remove these lines
        self.rep_dim        = params["rep_dim"] 
        self.input_dim      = params["input_dim"]
        self.num_layers     = params["num_layers"]
        self.num_hidden     = params["num_hidden"]
        self.activation     = params["activation"]
        self.dropout_prob   = params["dropout_prob"]
        self.dropout_active = params["dropout_active"]  
        self.loss_type      = params["LossFn"]
        self.train_prop     = params['train_prop']
        self.learningRate   = params['lr']
        self.epochs         = params['epochs']
        self.warm_up_epochs = params['warm_up_epochs']
        self.weight_decay   = params['weight_decay']
        if torch.cuda.is_available():
            self.device     = torch.device('cuda') # Make this an option
        else:
            self.device     = torch.device('cpu')
        # set up the network
        self.model          = build_network(network_name="feedforward", params=params).to(self.device)
        # create the loss function
        self.c              = hyperparams["center"].to(self.device)
        self.R              = hyperparams["Radius"]
        self.nu             = hyperparams["nu"]
        self.loss_fn        = LossFns[self.loss_type]
    def forward(self, x):
        x = x.to(device=self.device)
        x = self.model(x)
        return x
    def fit(self, x_train, verbosity=True):
        self.optimizer      = torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay = self.weight_decay)
        self.X              = torch.tensor(x_train.reshape((-1, self.input_dim))).float()
        if self.train_prop != 1:
            x_train, x_val = x_train[:int(self.train_prop*len(x_train))], x_train[int(self.train_prop*len(x_train)):]
            inputs_val = Variable(torch.tensor(x_val).to(self.device)).float()
        self.losses         = []
        self.loss_vals       = []
        for epoch in range(self.epochs):
            # Converting inputs and labels to Variable
            inputs = Variable(torch.tensor(x_train)).to(self.device).float()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            # get output from the model, given the inputs
            outputs = self.model(inputs)
            # get loss for the predicted output
            if self.loss_type=="SoftBoundary":
                self.loss = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu) 
            elif self.loss_type=="OneClass":
                self.loss = self.loss_fn(outputs=outputs, c=self.c) 
            #self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)
            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().cpu().numpy())
            # update parameters
            self.optimizer.step()
            if (epoch >= self.warm_up_epochs) and (self.loss_type=="SoftBoundary"):
                dist   = torch.sum((outputs - self.c) ** 2, dim=1)
                #self.R = torch.tensor(get_radius(dist, self.nu))
            if self.train_prop != 1.0:
                with torch.no_grad():
                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)
                    # get loss for the predicted output
                    if self.loss_type=="SoftBoundary":
                        loss_val = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu) 
                    elif self.loss_type=="OneClass":
                        loss_val = self.loss_fn(outputs=outputs, c=self.c).detach.cpu().numpy()
                    self.loss_vals.append(loss_val)
            if verbosity:
                if self.train_prop == 1:
                    print('epoch {}, loss {}'.format(epoch, self.loss.item()))
                else:
                    print('epoch {:4}, train loss {:.4e}, val loss {:.4e}'.format(epoch, self.loss.item(),loss_val))


# metrics.evaluation ##########################################################
device = 'cpu' # matrices are too big for gpu

def compute_alpha_precision(real_data, synthetic_data, emb_center, n_steps:int=100):
    emb_center = torch.tensor(emb_center, device=device)
    # n_steps = 30
    nn_size = 2
    alphas  = np.linspace(0, 1, n_steps)
    Radii   = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center) ** 2, dim=1)), alphas)
    synth_center          = torch.tensor(np.mean(synthetic_data, axis=0)).float()
    alpha_precision_curve = []
    beta_coverage_curve   = []
    synth_to_center       = torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - emb_center) ** 2, dim=1))
    nbrs_real = NearestNeighbors(n_neighbors = 2, n_jobs=-1, p=2).fit(real_data)
    real_to_real, _       = nbrs_real.kneighbors(real_data)
    nbrs_synth = NearestNeighbors(n_neighbors = 1, n_jobs=-1, p=2).fit(synthetic_data)
    real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(real_data)
    # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
    real_to_real          = torch.tensor(real_to_real[:,1].squeeze())
    real_to_synth         = torch.tensor(real_to_synth.squeeze())
    real_to_synth_args    = real_to_synth_args.squeeze()
    real_synth_closest    = synthetic_data[real_to_synth_args]
    real_synth_closest_d  = torch.sqrt(torch.sum((torch.tensor(real_synth_closest).float()- synth_center) ** 2, dim=1))
    closest_synth_Radii   = np.quantile(real_synth_closest_d, alphas)
    for k in range(len(Radii)):
        precision_audit_mask = (synth_to_center <= Radii[k]).detach().float().numpy()
        alpha_precision      = np.mean(precision_audit_mask)
        beta_coverage        = np.mean(((real_to_synth <= real_to_real) * (real_synth_closest_d <= closest_synth_Radii[k])).detach().float().numpy())
        alpha_precision_curve.append(alpha_precision)
        beta_coverage_curve.append(beta_coverage)
    # See which one is bigger
    authen = real_to_real[real_to_synth_args] < real_to_synth
    authenticity = np.mean(authen.numpy())
    Delta_precision_alpha = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (alphas[1] - alphas[0])
    Delta_coverage_beta  = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_coverage_curve))) * (alphas[1] - alphas[0])
    return alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authenticity

# from metrics.prdc import compute_prdc ########################################
def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric='euclidean', n_jobs=8)
    return dists

def get_kth_value(unsorted, k, axis=-1):
    """
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values

def compute_nearest_neighbour_distances(input_features, nearest_k):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii

def compute_prdc(real_features, fake_features, nearest_k=5):
    """
    Computes precision, recall, density, and coverage given two manifolds.
    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(
        real_features, fake_features)

    precision = (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).any(axis=0).mean()

    recall = (
            distance_real_fake <
            np.expand_dims(fake_nearest_neighbour_distances, axis=0)
    ).any(axis=1).mean()

    density = (1. / float(nearest_k)) * (
            distance_real_fake <
            np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <
            real_nearest_neighbour_distances
    ).mean()

    return dict(precision=precision, recall=recall,
                density=density, coverage=coverage)


# toy_metric_evaluation.ipynb ##################################################
nearest_k = 5
params  = dict({"rep_dim": None, 
                "num_layers": 2, 
                "num_hidden": 200, 
                "activation": "ReLU",
                "dropout_prob": 0.5, 
                "dropout_active": False,
                "train_prop" : 1,
                "epochs" : 100,
                "warm_up_epochs" : 10,
                "lr" : 1e-3,
                "weight_decay" : 1e-2,
                "LossFn": "SoftBoundary"})   
hyperparams = dict({"Radius": 1, "nu": 1e-2})

def plot_all(x, res, x_axis):
    print(x_axis)
    if type(res) == type([]):
        plot_legend = False
        res = {'0':res}
    else:
        plot_legend = True
    exp_keys = list(res.keys())
    print(res)
    metric_keys = res[exp_keys[0]][0].keys() 
    for m_key in metric_keys:
        for e_key in exp_keys:
          y = [res[e_key][i][m_key] for i in range(len(x))]
          plt.plot(x, y, label=e_key)
        
        plt.ylabel(m_key)
        plt.ylim(bottom=0)
        plt.xlabel(x_axis) 
        if plot_legend:
            plt.legend()
        plt.show()


def compute_metrics(X, Y, nearest_k = 5, model=None):
    results = compute_prdc(X,Y, nearest_k)
    if model is None:
        #these are fairly arbitrarily chosen
        params["input_dim"] = X.shape[1]
        params["rep_dim"] = X.shape[1]        
        hyperparams["center"] = torch.ones(X.shape[1])
        
        model = OneClassLayer(params=params, hyperparams=hyperparams)
         
        model.fit(X,verbosity=False)

    X_out = model(torch.tensor(X).float()).to(device='cpu').float().detach().numpy()
    Y_out = model(torch.tensor(Y).float()).to(device='cpu').float().detach().numpy()
    
    # alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, (thresholds, authen) = compute_alpha_precision(X_out, Y_out, model.c)
    alphas, alpha_precision_curve, beta_coverage_curve, Delta_precision_alpha, Delta_coverage_beta, authen = compute_alpha_precision(X_out, Y_out, model.c)
    # results['Dpa'] = Delta_precision_alpha
    # results['Dcb'] = Delta_coverage_beta
    # results['mean_aut'] = np.mean(authen)
    results.update({
        'alphas':alphas,
        'alpha_precision_curve':alpha_precision_curve,
        'beta_coverage_curve':beta_coverage_curve,
        'Delta_precision_alpha':Delta_precision_alpha,
        'Delta_coverage_beta':Delta_coverage_beta,
        'authen':authen,
    })
    return results, model

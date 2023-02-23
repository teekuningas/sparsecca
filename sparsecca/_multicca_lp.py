import numpy as np
from scipy.linalg import svd
from collections import defaultdict
import pyomo.environ as pyo

from ._utils_pmd import scale


def preprocess_datasets(datasets:list, standardize=True, mimic_R=True):
    # preprocess data
    datasets = datasets.copy()
    # 2 features needed
    for data in datasets:
        if len(data[0]) < 2:
            raise Exception('Need at least 2 features in each dataset')

    # standardize if set TRUE
    if standardize:
        for idx in range(len(datasets)):
            if mimic_R:
                datasets[idx] = scale(datasets[idx], center=True, scale=True)
            else:
                datasets[idx] = scale(datasets[idx], center=True, scale=False)

            datasets[idx] = datasets[idx].tolist()
            
    return datasets


def ObjRule(model):
    """Objective Function (4.3 in witten 2009)"""
    features = len(model.F.data())
    samples = len(model.S.data())
    return sum(
                (np.asarray([model.w_i_f[idx, f] for f in model.F.data()])[np.newaxis]
               @ np.asarray(xi).reshape(samples,features).T 
               @ np.asarray(xj).reshape(samples,features)
               @ np.asarray([model.w_i_f[jdx, f] for f in model.F.data()])[np.newaxis].T)[0,0] 
               for idx, xi in enumerate(model.X) for jdx, xj in enumerate(model.X) if idx<jdx )


def _update_w_lp(datasets, penalties, ws_init):
    """Solves 4.3 of witten 2009 with linear programming approach

    Params
    ------
    datasets: N matrices [samples x features]
    penalties: list of length N for each Xi
    ws_init: initial values for ws, usually provided by svd

    Returns
    -------
        w: defaultdict(list)
            - for each matrix Xn in datasets (n in [1:n]): n-> weights_vector
            - each weights_vector: list of length f (featuresize)
            - f = len(datasets[0][0])
        """
    model = pyo.ConcreteModel()

    # sets 
    model.N = pyo.Set(initialize=range(len(datasets)))
    model.S = pyo.Set(initialize=range(len(datasets[0])))
    model.F = pyo.Set(initialize=range(len(datasets[0][0])))
    model.X = pyo.Set(initialize=datasets)

    # params: ci i in [1:K]
    model.c = pyo.Param(model.N, initialize=penalties)

    # var
    model.w_i_f = pyo.Var(model.N, model.F, initialize=0.5)
    for n in range(len(ws_init)):
        for f in range(len(ws_init[0])):
            model.w_i_f[n,f].value = ws_init[n][f][0]

    # obj
    model.Obj = pyo.Objective(rule=ObjRule, sense=pyo.maximize)
    
    # constraints: lasso 
    model.constraint_lasso = pyo.ConstraintList()
    for i in model.N:
        model.constraint_lasso.add(sum(model.w_i_f[i,f] for f in model.F.data())<= model.c[i])
        
    # constraints: (2-norm)^2 ||wi||22 <=1
    model.constraint_norm2 = pyo.ConstraintList()
    for i in model.N:
        model.constraint_norm2.add(sum(model.w_i_f[i,f] * model.w_i_f[i,f] for f in model.F.data()) <= 1)
        
    # solving
    nonLinearOpt = pyo.SolverFactory('ipopt')
    instance_non_linear = model.create_instance()
    res = nonLinearOpt.solve(instance_non_linear)
    model.solutions.load_from(res)
    
    instance_non_linear.display()
    
    from collections import defaultdict
    w = defaultdict(list)
    for i in model.N:
        for f in model.F.data():
            w[i].append(instance_non_linear.w_i_f[i,f].value) 
            
    return w



def lp_pmd(datasets, penalties, K=1, standardize=True, mimic_R=True):
    """Calculates K weights [1xN]

    Params
    ------
    datasets : list
        List of n matrices of shape (samples x features)
    penalties : list
        List of n (1 x features) vectors. `c` in Witten 2009
    K : int (default: 1)
        Number of latent factors to calculate.
    standardize : bool (default: True)
        Whether to center and scale each dataset before computing sparse
        canonical variates.
    mimic_R : bool (default: True)
        Whether to mimic the R implementation exactly. Note that this flag can
        significantly change the resulting values.

    Returns
    -------
    w_final : list
        - list of length N, arrays of shape feature x K
    w_init: initialized with 0.5
    """
    sample_size = len(datasets[0])
    n_features = len(datasets[0][0])
    
    datasets_next = preprocess_datasets(datasets, standardize=standardize, mimic_R=mimic_R)
    weights = []
    
    k = 0
    for k in range(K):
        ws_init=[]
        for idx in range(len(datasets_next)):
            ws_init.append(svd(datasets_next[idx])[2][0:K].T)
        w = _update_w_lp(datasets_next, penalties, ws_init)
        datasets_current = datasets_next
    
        w_samples = {}
        for w_n in w:
            w_sample = np.repeat(w[w_n],sample_size, axis=0).reshape(sample_size, n_features)
            w_samples[w_n] = w_sample

        datasets_next = []
        for i, X_i in enumerate(datasets_current):
            X_i_next = X_i - w_samples[i]
            datasets_next.append(X_i_next.tolist())
            
        weights.append(w)      

    w_final = np.zeros((len(datasets), n_features, K))
    for k, w_k in enumerate(weights):
        #print(f"k: {k}")
        for n, w_value in enumerate(w_k.values()):
            #print(f"n: {n}")
            for f, w_feature in enumerate(w_value):
                #print(f"f: {f}")
                #print(w_feature)
                w_final[n][f][k] = w_feature
        
    w_init = [np.full((n_features, K), 0.5)]*len(datasets)
    return w_final, w_init
    

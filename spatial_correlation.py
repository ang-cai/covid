''' 
Information matrix

Make information matrix of NYC zones based on COVID covariates and cases.
Distance matrix is made of only neighboring zones, including zones connected by
bridge or tunnel.
'''
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.csgraph import connected_components
from scipy.special import digamma, polygamma

def inla_scale_model(Q, eps=np.sqrt(np.finfo(float).eps)):
    '''Copy of the inla.scale.model function in R. Original implementation 
    online at https://github.com/inbo/INLA/blob/master/R/scale.model.R. Comments 
    are from original R code. Assumes constraint matrix is all 1 and e=0.
    
    This function scales an intrinsic GMRF model so the geometric mean of the
    marginal variances is one.

    Returns scaled matrix Q and the scaled marginal variances in a dict.'''

    # marg.var = rep(0, nrow(Q))
    marg_var = np.zeros(Q.shape[0])

    # Q = inla.as.sparse(Q)
    Q = csr_matrix(Q)

    # inla.read.graph(Q))
    n_components, labels = connected_components(csgraph=Q, directed=False, 
                                                return_labels=True)
    
    # for(k in seq_len(g$cc$n)) {
    for k in range(n_components):
        # i = g$cc$nodes[[k]]
        i = np.where(labels == k)[0]

        # n = length(i)
        n = len(i)

        # QQ = Q[i, i, drop=FALSE]
        QQ = Q[i, :][:, i].toarray()

        # if (n == 1) {
        if n == 1:
            # QQ[1, 1] = 1
            QQ[0, 0] = 1

            # marg.var[i] = 1
            marg_var[i] = 1

        # } else {
        else:
            # cconstr = constr
            # if (!is.null(constr)) {
                ## the GMRFLib will automatically drop duplicated constraints; 
                # how convenient...
                # cconstr$A = constr$A[, i, drop = FALSE]                
                # eeps = eps
            eeps = eps

            # } else {
                # eeps = 0
            
            # res = inla.qinv(QQ + Diagonal(n) * max(diag(QQ)) * eeps, 
            # constr = cconstr)
            QQ = QQ + diags([np.max(np.diag(QQ)) * eeps], 0, 
                            shape=(n, n)).toarray()
            res = sparse_inv(csr_matrix(QQ)).toarray()

            # fac = exp(mean(log(diag(res))))
            fac = np.exp(np.mean(np.log(np.diag(res))))

            # QQ = fac * QQ
            QQ = fac * QQ

            # marg.var[i] = diag(res)/fac
            marg_var[i] = np.diag(res) / fac
        
        # Q[i, i] = QQ
        for idx, node in enumerate(i):
            Q[node, i] = QQ[idx, :]
    
    # return (list(Q=Q, var = marg.var))
    return {"Q": Q, "var": marg_var}

if __name__ == "__main__":
    nyc = gpd.read_file("nyc/nyc.shp")
    cases = nyc["nyc_case"]
    tests = nyc["nyc_test"]
    print(cases)

    neighbor = pd.read_csv("bordering_neighbor.csv")
    neighbor = neighbor.set_index("zcta")
    aproximate_precision = neighbor.to_numpy()
    diagonal = np.diag(np.sum(aproximate_precision, axis=0))
    aproximate_precision = np.subtract(diagonal, aproximate_precision)

    scaled_result = inla_scale_model(aproximate_precision)
    scaled_array = scaled_result["Q"].toarray()

    phi = .5

    precision_zcta = 1 / np.std(scaled_array) ** 2
    # sd_log_posrate = diagama function, trigama (gamma distribution)
    # add diagonal of sd log pos rate to spatial correlation matrix

    # spatial data repository dhs program to use their actual covariate data for blocks of countries for africa. we should only use the covariate and location data
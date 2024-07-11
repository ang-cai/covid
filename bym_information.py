''' 
Information matrix

Make information matrix of NYC zones based on COVID covariates and cases.
Uses the BYM model version of spatial correlation because of a neighborhood
matrix.
'''
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse.csgraph import connected_components
from scipy.special import digamma, polygamma
import matplotlib.pyplot as plt

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
            res = sparse_inv(csc_matrix(QQ)).toarray()

            # fac = exp(mean(log(diag(res))))
            fac = np.exp(np.mean(np.log(np.diag(res))))

            # QQ = fac * QQ
            QQ = fac * QQ

            # marg.var[i] = diag(res)/fac
            marg_var[i] = np.diag(res) / fac
        
        Q_lil = Q.tolil()
        # Q[i, i] = QQ
        for idx, node in enumerate(i):
            Q_lil[node, i] = QQ[idx, :]
        Q = Q_lil.tocsr()
    
    # return (list(Q=Q, var = marg.var))
    return {"Q": Q, "var": marg_var}

def bym_variance_covariance(design: list, spatial_correlation: list):
    '''Creates a variance covariance matrix using BYM model. Forumla: 
    Variance-covariance = (design' * spatial correlation * design)^-1
    
    Returns numpy array.
    '''
    variance_covariance = np.linalg.inv(np.matmul(
        np.matmul(design.T, np.linalg.inv(spatial_correlation)), design))

    return variance_covariance

if __name__ == "__main__":
    nyc = gpd.read_file("nyc/nyc.shp")
    neighbor_type = "Bordering" 
    # neighbor_type = "Bridged" 

    # Variables
    cases = nyc["nyc_case"].to_numpy()
    tests = nyc["nyc_test"].to_numpy()
    phi_values = [0, .5, 1]
    n = len(cases)

    # Design matrix
    pop = nyc["nyc_pop"]
    adr = nyc["nyc_adr"]
    old = nyc["nyc_old"]
    mp1f = nyc["nyc_mp1f"]
    white = nyc["nyc_white"]
    gini = nyc["nyc_gini"]
    income = nyc["nyc_income"]
    unempl = nyc["nyc_unempl"]
    povert = nyc["nyc_povert"]
    uninsu = nyc["nyc_uninsu"]
    hosp_beds = nyc["hosp_beds"]

    covariates = [pop, adr, old, mp1f, white, gini, income, povert, uninsu, 
                  hosp_beds]
    
    for i in range(len(covariates)):
        covariates[i] = pd.to_numeric(covariates[i], errors="coerce").to_numpy()
        np.nan_to_num(covariates[i], copy=False, nan=250000) # "250000+"
        covariates[i] = (covariates[i] - np.mean(covariates[i])) / np.std(
            covariates[i])
        
    design = np.stack(tuple(covariates), axis = -1)

    # Precision matrix
    neighbor = pd.read_csv(f"{neighbor_type}_neighbor.csv")
    neighbor = neighbor.set_index("zcta")
    aproximate_precision = neighbor.to_numpy()
    diagonal = np.diag(np.sum(aproximate_precision, axis=0))
    aproximate_precision = np.subtract(diagonal, aproximate_precision)
    scaled_result = inla_scale_model(aproximate_precision)
    scaled_prec = scaled_result["Q"].toarray()

    log_posrate = digamma(cases + 1) - np.log(tests + 1)
    sd_log_posrate = np.sqrt(polygamma(1, cases + 1))
    prec_zcta = 0.1/max(sd_log_posrate)**2

    # Initialize graphic
    figure, axis = plt.subplots(1, len(phi_values), figsize=(20, 6), sharey=True)

    # Information matrix
    for i in range(len(phi_values)):
        spatial_corre = (np.linalg.pinv(scaled_prec) * phi_values[i] + 
                         np.identity(n) * (1 - phi_values[i])) / prec_zcta
        spatial_corre = spatial_corre + np.diag(sd_log_posrate ** 2)
        variance_covar = bym_variance_covariance(design, spatial_corre)
        sign, with_info = np.linalg.slogdet(variance_covar)

        information_gain = []
        for j in range(n):
            design_minus = np.delete(design, j, 0)
            spatial_corre_minus = np.delete(np.delete(spatial_corre, j, 0), j, 1)
            variance_covar_minus = bym_variance_covariance(
                design_minus, spatial_corre_minus)
            sign, without_info = np.linalg.slogdet(variance_covar_minus)
            information_gain.append(-(with_info - without_info))
            
        information_gain_scaled = np.array(information_gain)
        information_gain_scaled = np.divide(information_gain_scaled, .125)
        information_gain_scaled = np.float_power(information_gain_scaled, .25)

        # Plot
        nyc["info_gain"] = information_gain_scaled.tolist()
        nyc.plot(column="info_gain", ax=axis[i])
        axis[i].get_xaxis().set_visible(False)
        axis[i].get_yaxis().set_visible(False)
        figure.subplots_adjust(wspace=0)
        if phi_values[i] == 0:
            axis[i].set_title("Independent")
        else:
            axis[i].set_title(f"BYM: phi={phi_values[i]}") 
    
    title = f"NYC Information Gain of COVID Covariates with {neighbor_type} Neighbors"
    figure.suptitle(title, fontsize="xx-large")

    file = f"{neighbor_type}_information_gain_nyc.png"
    plt.savefig(file, bbox_inches="tight") 
    plt.show()



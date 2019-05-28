import numpy as np 
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
import theano.tensor as T



def plot_coef(model, X):
        
    """
    Plots the coefficients of a linear model

    Parameters
    ----------
        
    model : pymc3_models linear model object
        
        
    X : X dataframe used to train the model
            shape [num_training_samples, num_pred]

    """
        
    
    coefs = model.summary.reset_index().rename(columns = {'index' : 'coef'})
    ypa_ci = np.array(list(zip(-coefs['hpd_2.5'] + coefs['mean'], 
                                coefs['hpd_97.5'] - coefs['mean']))).T


    # Correct order coefficients are returned
    coef = ['intercept']
    for i in X.columns:
        coef.append(i)
    coef.append('sigma')
    coefs['coef'] = coef
    coefs = coefs.sort_values('mean')
    plt.figure(figsize = (12, 8))
    ax = plt.errorbar('mean', 'coef', xerr=ypa_ci, data=coefs, fmt='ko', 
                 capthick=2, capsize=10, label=None)
    plt.title('Coefficient Effect Size')
    plt.axvline(0)
    return ax


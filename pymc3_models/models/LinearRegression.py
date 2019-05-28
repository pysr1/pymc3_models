import numpy as np
import pymc3 as pm
from sklearn.metrics import r2_score
import theano
import theano.tensor as T

from pymc3_models.exc import PyMC3ModelsError
from pymc3_models.models import BayesianModel


class LinearRegression(BayesianModel):
    """
    Linear Regression built using PyMC3.
    """

    def __init__(self):
        super(LinearRegression, self).__init__()
        
        
    def create_model(self, sd, dof):
        """
        Creates and returns the PyMC3 model.

        Note: The size of the shared variables must match the size of the training data.
        Otherwise, setting the shared variables later will raise an error.
        See http://docs.pymc.io/advanced_theano.html

        Returns
        -------
        the PyMC3 model
        """
        model_input = theano.shared(np.zeros([self.num_training_samples, self.num_pred]))

        model_output = theano.shared(np.zeros(self.num_training_samples))

        self.shared_vars = {
            'model_input': model_input,
            'model_output': model_output,
        }

        model = pm.Model()

        with model:
            alpha = pm.StudentT('intercept', mu=0, sd=sd, nu = 7, shape=(1))
            betas = pm.StudentT('coefficients', mu=0, sd=sd, nu = 7, shape=(1, self.num_pred))

            
            
            s = pm.HalfNormal('s', tau=1)

            mean = alpha + T.sum(betas * model_input, 1)

            y = pm.Normal('y', mu=mean, sd=s, observed=model_output)

        return model

    def fit(
        self,
        X,
        y,
        inference_type='advi',
        num_advi_sample_draws=10000,
        minibatch_size=None,
        inference_args=None,
        scale = 'auto',
        dof = 7
    ):
        """
        Train the Linear Regression model

        Parameters
        ----------
        X : numpy array
            shape [num_training_samples, num_pred]

        y : numpy array
            shape [num_training_samples, ]

        inference_type : str (defaults to 'advi')
            specifies which inference method to call
            Currently, only 'advi' and 'nuts' are supported.

        num_advi_sample_draws : int (defaults to 10000)
            Number of samples to draw from ADVI approximation after it has been fit;
            not used if inference_type != 'advi'

        minibatch_size : int (defaults to None)
            number of samples to include in each minibatch for ADVI
            If None, minibatch is not run.

        inference_args : dict (defaults to None)
            arguments to be passed to the inference methods.
            Check the PyMC3 docs for permissable values.
            If None, default values will be set.
        
        scale : int or 'auto' (defaults to 'auto')
            The scale for the T-distribution. When set to
            default 'auto', scale = 2.5 * sd(y) just like 
            in Rstanarm
        
        dof : int (defaults to 7)
            The degrees of free for the T-distribution.


        """
        self.num_training_samples, self.num_pred = X.shape

        self.inference_type = inference_type

        if y.ndim != 1:
            y = np.squeeze(y)

        if not inference_args:
            inference_args = self._set_default_inference_args()

        if self.cached_model is None:
            if scale == 'auto':
                self.cached_model = self.create_model(sd = 2.5 * np.std(y), dof = dof)
            else:
                self.cached_model = self.create_model(sd = scale, dof = dof)
        if minibatch_size:
            with self.cached_model:
                minibatches = {
                    self.shared_vars['model_input']: pm.Minibatch(X, batch_size=minibatch_size),
                    self.shared_vars['model_output']: pm.Minibatch(y, batch_size=minibatch_size),
                }

                inference_args['more_replacements'] = minibatches
        else:
            self._set_shared_vars({'model_input': X, 'model_output': y})

        self._inference(inference_type, inference_args, num_advi_sample_draws=num_advi_sample_draws)

        return self

    def predict(self, X, return_std=False, num_ppc_samples=2000):
        """
        Predicts values of new data with a trained Linear Regression model

        Parameters
        ----------
        X : numpy array
            shape [num_training_samples, num_pred]

        return_std : bool (defaults to False)
            flag of whether to return standard deviations with mean values

        num_ppc_samples : int (defaults to 2000)
            'samples' parameter passed to pm.sample_ppc
        """

        if self.trace is None:
            raise PyMC3ModelsError('Run fit on the model before predict.')

        num_samples = X.shape[0]

        if self.cached_model is None:
            self.cached_model = self.create_model()

        self._set_shared_vars({'model_input': X, 'model_output': np.zeros(num_samples)})

        ppc = pm.sample_ppc(self.trace, model=self.cached_model, samples=num_ppc_samples)

        if return_std:
            return ppc['y'].mean(axis=0), ppc['y'].std(axis=0)
        else:
            return ppc['y'].mean(axis=0)

    def score(self, X, y, num_ppc_samples=2000):
        """
        Scores new data with a trained model with sklearn's r2_score.

        Parameters
        ----------
        X : numpy array
            shape [num_training_samples, num_pred]

        y : numpy array
            shape [num_training_samples, ]

        num_ppc_samples : int (defaults to 2000)
            'samples' parameter passed to pm.sample_ppc
        """

        return r2_score(y, self.predict(X, num_ppc_samples=num_ppc_samples))

    def save(self, file_prefix):
        params = {
            'inference_type': self.inference_type,
            'num_pred': self.num_pred,
            'num_training_samples': self.num_training_samples
        }

        super(LinearRegression, self).save(file_prefix, params)

    def load(self, file_prefix):
        params = super(LinearRegression, self).load(file_prefix, load_custom_params=True)

        self.inference_type = params['inference_type']
        self.num_pred = params['num_pred']
        self.num_training_samples = params['num_training_samples']

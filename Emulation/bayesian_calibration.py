import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd
import math


class BayesianCalibration:
    def __init__(self, input, emulator_output, filtered_output, which_obs, 
                 epsilon_obs_scale, epsilon_alt=None):
        self.input = input
        self.emulator_output = emulator_output
        self.filtered_output = filtered_output
        self.which_obs = which_obs
        self.epsilon_alt = epsilon_alt 
        


        # Priors
        self.mu_0 = np.array(input.mean().loc[:'T'])
        self.mu_0[-1] = input.iloc[which_obs]['T']  # Assuming 'T' is the last parameter
        self.mu_0 = self.mu_0.reshape(-1, 1)
        self.Sigma_0 = np.diag(input.var().loc[:'T'])
        self.Sigma_0[-1, -1] = 0.0000001
        
        # Parameter names
        self.param_names = input.loc[:, :'T'].columns.to_list()

        # Model error
        self.epsilon_model = np.diag(emulator_output['RSE']**2) 
       
        
        # Observation error
        self.obs_error_scale = epsilon_obs_scale 
        default_epsilon_obs = np.diag(np.std(filtered_output) * self.obs_error_scale)  
        self.epsilon_obs = default_epsilon_obs if epsilon_alt is None else self.epsilon_alt*self.obs_error_scale
        
      
        
        # Compute posterior
        self.compute_posterior()
    
    def compute_posterior(self):
        full_error = self.epsilon_obs + self.epsilon_model
       
        # Construct beta matrix and intercepts
        beta_matrix = []
        intercept = []
        for _, row_entry in self.emulator_output.iterrows():
            model = row_entry['Model']
            beta_matrix.append(model.coef_)
            intercept.append(model.intercept_)
        
        beta_matrix = np.array(beta_matrix)
        intercept = np.array(intercept).reshape(len(intercept), 1)

        # Select observation and scale by intercept
        Y_obs = np.array(self.filtered_output.T[self.which_obs]).reshape(-1, 1)
        Y_scaled = Y_obs - intercept
        

        # Compute posterior covariance
        Sigma_post_inv = (beta_matrix.T @ np.linalg.inv(full_error) @ beta_matrix) + np.linalg.inv(self.Sigma_0)
        self.Sigma_post = np.linalg.inv(Sigma_post_inv)
        
        # Compute posterior mean
        self.Mu_post = self.Sigma_post @ (beta_matrix.T @ np.linalg.inv(full_error) @ Y_scaled + np.linalg.inv(self.Sigma_0) @ self.mu_0)
        
       

    def sample_posterior(self, n_samples):
        rg = np.random.default_rng(1)
        self.samples = rg.multivariate_normal(self.Mu_post.flatten(), self.Sigma_post, size=n_samples)  # Generate 10 samples
        self.samples_df = pd.DataFrame(self.samples)
        self.samples_df.columns = self.param_names

    def plot_distributions(self):
        prior_means = self.mu_0.flatten()
        prior_stds = np.sqrt(np.diag(self.Sigma_0))
        posterior_means = self.Mu_post.flatten()
        posterior_stds = np.sqrt(np.diag(self.Sigma_post))
        true_values = self.input.iloc[self.which_obs].values
        
        
        fig, axes = plt.subplots(2, math.ceil(len(self.param_names)/2), figsize=(18, 8))  
        axes = axes.flatten()  # Flatten to 1D array
        for i, ax in enumerate(axes[:len(self.param_names)]):  # Only iterate over valid axes
        
            # Define x-range based on prior and posterior means
            x_min = min(prior_means[i] - 3 * prior_stds[i], posterior_means[i] - 3 * posterior_stds[i])
            x_max = max(prior_means[i] + 3 * prior_stds[i], posterior_means[i] + 3 * posterior_stds[i])
            x = np.linspace(x_min, x_max, 100)

            # Compute PDFs
            prior_pdf = norm.pdf(x, prior_means[i], prior_stds[i])
            posterior_pdf = norm.pdf(x, posterior_means[i], posterior_stds[i])

            # Plot prior and posterior distributions
            ax.plot(x, prior_pdf, label="Prior", linestyle="dashed", color="blue")
            ax.plot(x, posterior_pdf, label="Posterior", linestyle="solid", color="red")

            # Plot true value as a vertical line
            ax.axvline(true_values[i], color="green", linestyle="dotted", label="True Value")

            # Labels and title
            ax.set_title(self.param_names[i])
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()

        plt.tight_layout()
        plt.show()
            
    def plot_covariances(self):
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(self.Sigma_0, annot=True, fmt=".3f", cmap="RdBu", xticklabels=self.param_names, yticklabels=self.param_names, ax=axes[0])
        axes[0].set_title("Prior Covariance Matrix")
        sns.heatmap(self.Sigma_post, annot=True, fmt=".4f", cmap="PiYG", xticklabels=self.param_names, yticklabels=self.param_names, ax=axes[1])
        axes[1].set_title("Posterior Covariance Matrix")
        plt.tight_layout()
        plt.show()

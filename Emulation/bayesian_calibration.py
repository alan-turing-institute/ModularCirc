import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pandas as pd

class BayesianCalibration:
    def __init__(self, filtered_input, selected_rows, filtered_output, which_obs, 
                 epsilon_obs_scale=0.05, epsilon_alt=None):
        self.filtered_input = filtered_input
        self.selected_rows = selected_rows
        self.filtered_output = filtered_output
        self.which_obs = which_obs
        self.epsilon_alt = epsilon_alt 
        
        
        # Fixed priors
        self.mu_0 = np.array([20.5, 0.31, 3.8, 1.15, filtered_input['T'][which_obs]])[:, np.newaxis]
        self.Sigma_0 = np.diag([3.42**2, 0.05**2, 0.63**2, 0.48**2, 0.0000001])
        
        # Model error
        self.epsilon_model = np.diag(selected_rows['RSE']**2) 
       
        
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
        for _, row_entry in self.selected_rows.iterrows():
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
        self.samples_df.columns = ["svn.c", "pat.r", "pat.c", "rv.E_act", "T"]

    def plot_distributions(self):
        prior_means = self.mu_0.flatten()
        prior_stds = np.sqrt(np.diag(self.Sigma_0))
        posterior_means = self.Mu_post.flatten()
        posterior_stds = np.sqrt(np.diag(self.Sigma_post))
        true_values = self.filtered_input.iloc[self.which_obs].values
        param_names = ["svn.c", "pat.r", "pat.c", "rv.E_act", "T"]
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        for i, ax in enumerate(axes):
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
            ax.set_title(param_names[i])
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_covariances(self):
        param_names = ["svn.c", "pat.r", "pat.c", "rv.E_act", "T"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(self.Sigma_0, annot=True, fmt=".3f", cmap="RdBu", xticklabels=param_names, yticklabels=param_names, ax=axes[0])
        axes[0].set_title("Prior Covariance Matrix")
        sns.heatmap(self.Sigma_post, annot=True, fmt=".4f", cmap="PiYG", xticklabels=param_names, yticklabels=param_names, ax=axes[1])
        axes[1].set_title("Posterior Covariance Matrix")
        plt.tight_layout()
        plt.show()

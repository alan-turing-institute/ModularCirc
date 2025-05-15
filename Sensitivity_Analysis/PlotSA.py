import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PlotSA:
    def __init__(self, directory):
        self.directory = directory
        self.csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    def _load_data(self, selected_files):
        """Load data from selected CSV files."""
        files_to_read = self.csv_files if not selected_files else selected_files
        data = {}
        for file in files_to_read:
            file_path = os.path.join(self.directory, file)
            data[file] = pd.read_csv(file_path, index_col=0)
        return data
    
    def plot_top_sensitivity(self, number, all_files=True, selected_files=[]):
        """Plots pie charts for the top x sensitive parameters."""
        data = self._load_data([] if all_files else selected_files)
        
        rows, cols = (len(data) + 2) // 3, min(3, len(data))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        fig.suptitle(f"Pie Charts of Top {number} Sensitivity Indices", fontsize=16)
        axes = axes.flatten()
        
        for i, (file_name, df) in enumerate(data.items()):
            sorted_df = df["ST"].sort_values(ascending=False).head(x)
            
            axes[i].pie(sorted_df, labels=sorted_df.index, autopct='%1.1f%%', startangle=140)
            axes[i].set_title(file_name.rsplit("_", 1)[-1].replace(".csv", ""))
        
        for j in range(len(data), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def plot_percent_sensitivity(self, percentage, all_files=True, selected_files=[]):
        """Plots pie charts for parameters accounting for x% of variation."""
        data = self._load_data([] if all_files else selected_files)
        
        rows, cols = (len(data) + 2) // 3, min(3, len(data))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        fig.suptitle(f"Pie Charts of Total Sensitivity Indices Accounting for {percentage}% Sensitivity", fontsize=16)
        axes = axes.flatten()
        
        for i, (file_name, df) in enumerate(data.items()):
            sorted_df = df["ST"].sort_values(ascending=False)
            cumulative_sum = sorted_df.cumsum()
            total_sum = sorted_df.sum()
            significant_indices = cumulative_sum <= (percentage / 100) * total_sum
            
            axes[i].pie(sorted_df[significant_indices], labels=sorted_df.index[significant_indices], autopct='%1.1f%%', startangle=140)
            axes[i].set_title(file_name.rsplit("_", 1)[-1].replace(".csv", ""))
        
        for j in range(len(data), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def plot_heatmap(self, all_files=True, selected_files=[]):
        """Plots a heatmap of sensitivity indices for each parameter across all CSV files."""
        data = self._load_data([] if all_files else selected_files)
        
        combined_df = pd.DataFrame()
        
        for file_name, df in data.items():
            combined_df[file_name] = df["ST"]
        
        combined_df = combined_df.fillna(0).T  # Transpose to have CSV files on Y-axis and parameters on X-axis
        combined_df['means'] = combined_df.mean(axis=1)
        cols = 0.5 * len(combined_df.index)
        plt.figure(figsize=(25, cols))
        sns.heatmap(combined_df, annot=True, cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": 0.5})
        plt.title("Sensitivity Heatmap")
        plt.xlabel("Parameters")
        plt.ylabel("CSV Files")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        self.combined_df = combined_df



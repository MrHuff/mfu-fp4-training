import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
from test_mxfp_matmul import ErrorMetrics, plot_metrics
import random
from low_bits_training.quantization.dimensionQuantisationClass import (
    MXLinearDimConfig,
)
from low_bits_training.quantization.fusedMXFPMatmul import swap_linear_with_mx_linear_fused
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4
)
import json 
import seaborn as sns
import itertools
from torchviz import make_dot

torch.autograd.set_detect_anomaly(True)

#TODO, keep playing around and discuss what operations can be included in backprop, try to replicate the total STE and understand why including absmax normalisation etc sucks

class RegressionTrainer:
    def __init__(self, model, num_samples=100000, input_dim=128, noise_std=0.1, lr=0.01,
                 batch_size=4096, epochs=10,dtype = torch.bfloat16):
        self.device  = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.lr = lr
        self.dtype = dtype
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.first_epoch_grads = []

        self.train_data, self.val_data = self._generate_data(num_samples, input_dim, noise_std)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _generate_data(self, num_samples, input_dim, noise_std):
        X = torch.randn(num_samples, input_dim, dtype=self.dtype)
        true_weights = torch.randn(input_dim, 1, dtype=self.dtype)
        y = X @ true_weights + noise_std * torch.randn(num_samples, 1, dtype=self.dtype)

        split = int(0.8 * num_samples)
        train_dataset = TensorDataset(X[:split], y[:split])
        val_dataset = TensorDataset(X[split:], y[split:])
        return train_dataset, val_dataset

    def extract_grad(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        self.model.train()
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            X_ref = torch.tensor(X_batch,requires_grad=True)   
            y_pred = self.model(X_ref.to(self.device))
            loss = self.criterion(y_pred, y_batch.to(self.device))
            loss.backward()
            w_grad = self.model.linear.weight.grad
            X_batch_grad = X_ref.grad
            return X_batch_grad, w_grad
        
    def train(self): #extract gradient trajectory!
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch.to(self.device))
                loss = self.criterion(y_pred, y_batch.to(self.device))
                #make_dot(loss, params=dict(self.model.named_parameters()),show_attrs=True, show_saved=True).render("graph", format="pdf")

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                # print(y_pred)
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name} grad:\n{param.grad}")

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_pred = self.model(X_batch.to(self.device))
                    loss = self.criterion(y_pred, y_batch.to(self.device))
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.epochs}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

class TinyModel(torch.nn.Module):
    def __init__(self,dtype):
        super().__init__()
        self.linear = torch.nn.Linear(128, 1, dtype=dtype)

    def forward(self, x):
        return self.linear(x)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Slower, but reproducible
    torch.backends.cudnn.benchmark = False


BLOCK_SIZES = [32]
SCALE_TYPES = ['E8M0']
MAX_APPROX = ['hardsoftmax']#softsoftmax "hardsoftmax", 'STE','absmax'
STEP_GRADIENTS = ['STE'] #[None, 'baseline', 'spline', 'sigmoid'] #This apparently sucks ass, figure out why and fix it! Cut the bullshit!
SCALING_QUANT_GRADIENTS = ['STE'] #[None, 'baseline', 'spline', 'sigmoid'] #This apparently sucks ass, figure out why and fix it! Cut the bullshit!


#Bruv you've got vanishing gradients, beware of clipping functions and keep using graphviz to inspect the backprop flow
# Also do note that somehow the scaler quantisation function records the entire backpropagation of the scaling quantisation, not sure if this actually is desired!
DTYPE = torch.bfloat16
if __name__ == "__main__":
    df_results_w = pd.DataFrame(columns=[
        "block_size", "exmy","max_type", "step_grad_type","sq_grad",
        "mean_rel_error", "median_rel_error", "max_rel_error", "std_rel_error", "p99_rel_error", "p999_rel_error", "iqr_rel_error",
        "mean_abs_error", "median_abs_error", "max_abs_error", "std_abs_error", "p99_abs_error", "p999_abs_error", "iqr_abs_error",
        "rms_error", "worst_offenders_rel", "worst_offenders_abs"
    ])
    df_results_X = pd.DataFrame(columns=[
        "block_size", "exmy","max_type", "step_grad_type","sq_grad",
        "mean_rel_error", "median_rel_error", "max_rel_error", "std_rel_error", "p99_rel_error", "p999_rel_error", "iqr_rel_error",
        "mean_abs_error", "median_abs_error", "max_abs_error", "std_abs_error", "p99_abs_error", "p999_abs_error", "iqr_abs_error",
        "rms_error", "worst_offenders_rel", "worst_offenders_abs"
    ])


    set_seed(42)
    model = TinyModel(DTYPE)
    base_trainer = RegressionTrainer(model)
    X_grad_ref,w_grad_ref = base_trainer.extract_grad()
    base_trainer.train()
    for a in [1.0]:
        all_results = []
        for block_size, scale_type, ma, step_gradient,sq_gradient in itertools.product(
            BLOCK_SIZES, SCALE_TYPES, MAX_APPROX, STEP_GRADIENTS,SCALING_QUANT_GRADIENTS
        ):
            
            config = MXLinearDimConfig(
                                        block_dim =None, 
                                        block_size=block_size, 
                                        elem_dtype=DTYPE_FP4, 
                                        scale_type=scale_type,
                                        use_approx={
                                                        'smooth': ma,
                                                         'alpha': a , 
                                                         'stepGradient': step_gradient, 
                                                         'temperature': 0.025, 
                                                         'k':5, 
                                                         'dtype':DTYPE,
                                                         'lb':1,
                                                         'ub':1.5,
                                                         'qGradient': sq_gradient
                                                         },
                                        dtype = DTYPE
                                        )
            set_seed(42)
            model = TinyModel(DTYPE)
            swap_linear_with_mx_linear_fused(model, config=config)
            trainer = RegressionTrainer(model)
            X_grad,w_grad = trainer.extract_grad()
            trainer.train()
            err_metrics_w = ErrorMetrics(w_grad_ref, w_grad)
            err_metrics_X = ErrorMetrics(X_grad_ref, X_grad)

            df_results_w.loc[len(df_results_w)] = [block_size, scale_type,ma,step_gradient,sq_gradient] + err_metrics_w.get_stats()
            df_results_X.loc[len(df_results_X)] = [block_size, scale_type,ma,step_gradient,sq_gradient] + err_metrics_X.get_stats()
    print(df_results_w[[ "block_size", "exmy","max_type", "step_grad_type","sq_grad",
        "mean_rel_error", "median_rel_error"]])
    print(df_results_X[[ "block_size", "exmy","max_type", "step_grad_type","sq_grad",
        "mean_rel_error", "median_rel_error"]])
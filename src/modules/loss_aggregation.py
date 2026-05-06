import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNODEsAggregation(nn.Module):
    """ MultiNODEs Scalarization (LS)
    """
    def __init__(self, tasks, weights=None, device='cpu'):
        super().__init__()
        self.device = device
        num_tasks = len(tasks) 
        self.tasks = tasks
        if weights is None:
            # If no weights are provided, use equal weighting.
            self.weights = torch.ones(num_tasks, device=self.device) 
        else:
            for task in weights:
                if task not in tasks:
                    raise ValueError(f"Weight provided for unknown task '{task}'. Known tasks: {tasks}")
            weights = [weights[task] for task in tasks]
            self.weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        self.weights_dict = {task: weight for task, weight in zip(tasks, self.weights)}
            
    def forward(self, dict_losses):
        for task in self.tasks:
            if task not in dict_losses:
                raise ValueError(f"Loss for task '{task}' not found in input dictionary.")
        loss_long = dict_losses['Longi ODE']
        loss_stat = dict_losses['Static']
        long = loss_long / (loss_long  + loss_stat)
        stat = loss_stat / (loss_long + loss_stat)
        long_scaled = stat / (long + stat) * loss_long
        stat_scaled = long / (long + stat) * loss_stat
        aggregated_loss = long_scaled + self.weights_dict['Static'].to(self.device) * stat_scaled
        return aggregated_loss


class LinearScalarization(nn.Module):
    """ Linear Scalarization (LS)
        Simply aggregates losses with fixed (or equal) weights. 
    """
    def __init__(self, tasks, weights=None, device='cpu'):
        super().__init__()
        self.device = device
        num_tasks = len(tasks) 
        self.tasks = tasks
        if weights is None:
            # If no weights are provided, use equal weighting.
            self.weights = torch.ones(num_tasks, device=self.device) / num_tasks
        else:
            for task in weights:
                if task not in tasks:
                    raise ValueError(f"Weight provided for unknown task '{task}'. Known tasks: {tasks}")
            weights = [weights[task] for task in tasks]
            sum_weights = sum(weights)
            weights = [w / sum_weights for w in weights]  # Normalize weights to sum to 1
            self.weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        self.weights_dict = {task: weight for task, weight in zip(tasks, self.weights)}
        # print(self.weights_dict)

    def update_weights(self, multiplicators):
        new_weights = self.weights * multiplicators
        sum_weights = sum(new_weights)
        new_weights = [w / sum_weights for w in new_weights]
        self.weights = torch.tensor(new_weights, dtype=torch.float, device=self.device)
        self.weights_dict = {task: weight for task, weight in zip(self.tasks, self.weights)}
            
    def forward(self, dict_losses):
        for task in self.tasks:
            if task not in dict_losses:
                raise ValueError(f"Loss for task '{task}' not found in input dictionary.")
        losses = torch.stack([dict_losses[task] for task in self.tasks])
        aggregated_loss = (self.weights.to(losses.device) * losses).sum()
        return aggregated_loss
    
    def compute_dict_weighted(self, dict_losses):
        dict_losses_weighted = {}
        for task in self.tasks:
            loss = dict_losses[task]
            weight = self.weights_dict[task]
            dict_losses_weighted[task] = weight * loss
        return dict_losses_weighted
    

class RealTimeLossScalarization(nn.Module):
    """ Li/Li_grad_free 
    """
    def __init__(self, tasks, device='cpu'):
        super().__init__()
        self.device = device
        self.tasks = tasks
            
    def forward(self, dict_losses): 
        for task in self.tasks:
            if task not in dict_losses:
                raise ValueError(f"Loss for task '{task}' not found in input dictionary.")
        losses = torch.stack([dict_losses[task] for task in self.tasks])
        losses_scales = torch.stack([abs(dict_losses[task].detach()) for task in self.tasks])
        aggregated_loss = (losses / losses_scales).sum()
        return aggregated_loss
    
    def compute_dict_weighted(self, dict_losses):
        dict_losses_weighted = {}
        for task in self.tasks:
            loss = dict_losses[task]
            weight = abs(dict_losses[task].detach())
            dict_losses_weighted[task] = loss / weight
        return dict_losses_weighted
    

class UncertaintyWeighting(nn.Module):
    """ Homoscedastic Uncertainty Weighting (UW)
        Learns a log-variance for each task to scale its loss.
    """
    def __init__(self, tasks, device='cpu'):
        super().__init__()
        # Initialize learnable log-variance parameters (one per task)
        self.device = device
        self.tasks = tasks
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.tensor([-0.5], device=self.device)) # torch.zeros(1, device=self.device)
            for task in self.tasks
        })

        
    def forward(self, dict_losses):
        for task in self.tasks:
            if task not in dict_losses:
                raise ValueError(f"Loss for task '{task}' not found in input dictionary.")
        
        weighted_losses = 0
        for i, task in enumerate(self.tasks):
            loss = dict_losses[task]
            log_var = self.log_vars[task]
            # For each task, compute weighted loss: exp(-s_i)*L_i + s_i, where s_i = log_vars[i]
            weighted_loss = torch.exp(-log_var) * loss + 0.5 * log_var
            weighted_losses += weighted_loss
        return weighted_losses
    
    def compute_dict_weighted(self, dict_losses):
        dict_losses_weighted = {}
        for task in self.tasks:
            loss = dict_losses[task]
            log_var = self.log_vars[task]
            # print("Task:", task, "coef:", torch.exp(-log_var).item())
            dict_losses_weighted[task] = torch.exp(-log_var) * loss + 0.5 * log_var
        return dict_losses_weighted

        

class DynamicWeightAverage(nn.Module):
    """ Dynamic Weight Average (DWA)
        Uses the history of losses to compute task weights dynamically.
    """
    def __init__(self, num_tasks, T=2.0, device='cpu'):
        super().__init__()
        self.num_tasks = num_tasks
        self.T = T
        self.device = device
        self.loss_history = [torch.ones(num_tasks, device=device), torch.ones(num_tasks, device=self.device)]
        self.epoch = 1  # Start at epoch 1
        
    def update_loss_history(self, current_losses):
        """
        Update the loss history at the end of an epoch.
        Args:
            current_losses (torch.Tensor): Tensor of losses for the current epoch.
        """
        self.loss_history[0] = self.loss_history[1].clone()
        self.loss_history[1] = current_losses.detach().clone()
        self.epoch += 1
        
    def forward(self, losses):
        """
        Compute the aggregated loss using dynamic weights.
        Returns:
            aggregated_loss (torch.Tensor): The weighted sum of losses.
            weights (torch.Tensor): The computed weights (for logging or analysis).
        """
        if self.epoch == 1:
            weights = torch.ones_like(losses, device=self.device)
        else:
            ratio = self.loss_history[1] / (self.loss_history[0] + 1e-8)
            weights = self.num_tasks * torch.softmax(ratio / self.T, dim=0)

        aggregated_loss = (weights * losses).sum()
        self.update_loss_history(losses)
        return aggregated_loss 
    
    
class DynamicWeightAverageNormalization(nn.Module):
    """ Dynamic Weight Average (DWA)
        Uses the history of losses to compute task weights dynamically.
    """
    def __init__(self, num_tasks, T=2.0, device='cpu'):
        super().__init__()
        self.num_tasks = num_tasks
        self.T = T
        self.device = device
        self.alpha = 0.8
        self.loss_history = [torch.ones(num_tasks, device=device), torch.ones(num_tasks, device=self.device)]
        self.epoch = 1  # Start at epoch 1
        
    def update_loss_history(self, current_losses):
        """
        Update the loss history at the end of an epoch.
        Args:
            current_losses (torch.Tensor): Tensor of losses for the current epoch.
        """
        self.loss_history[0] = self.loss_history[1].clone()
        self.loss_history[1] = current_losses.detach().clone()
        self.epoch += 1

        
    def forward(self, losses):
        """
        Compute the aggregated loss using dynamic weights.
        Returns:
            aggregated_loss (torch.Tensor): The weighted sum of losses.
            weights (torch.Tensor): The computed weights (for logging or analysis).
        """
        if self.epoch > 1:
            losses = self.alpha * self.loss_history[1] + (1 - self.alpha) * losses

        if self.epoch == 1:
            weights = torch.ones_like(losses, device=self.device)
        else:
            ratio = self.loss_history[1] / (self.loss_history[0] + 1e-8)
            weights = self.num_tasks * torch.softmax(ratio / self.T, dim=0)

        # print('weights : ', weights)
        aggregated_loss = (weights * losses).sum()
        self.update_loss_history(losses)
        return aggregated_loss 


class VarianceWeightedLoss(torch.nn.Module):
    def __init__(self, tasks, beta=0.1, eps=1e-8, device="cpu"):
        super().__init__()
        self.tasks = tasks
        self.beta = beta
        self.eps = eps
        self.device = device
        # Initialize moving means and variances
        self.register_buffer("means", torch.zeros(len(tasks), device=device))
        self.register_buffer("vars", torch.ones(len(tasks), device=device))
        self.weights_dict = {task: 1.0 for task in tasks}

    def forward(self, dict_losses):
        task_losses = torch.stack([dict_losses[t] for t in self.tasks])  # each is a scalar tensor with grad
        
        task_losses_detached = task_losses.detach()
        self.means = (1 - self.beta) * self.means + self.beta * task_losses_detached
        diff = task_losses_detached - self.means
        self.vars = (1 - self.beta) * self.vars + self.beta * (diff ** 2)

        # Compute inverse variance weights
        inv_std = 1.0 / (torch.sqrt(self.vars) + self.eps)
        inv_std = inv_std * len(self.tasks) / inv_std.sum()
        self.weights_dict = {t: inv_std[i] for i, t in enumerate(self.tasks)}

        # Weighted total loss
        total_loss = torch.sum(inv_std.detach() * task_losses)
        return total_loss

    def compute_dict_weighted(self, dict_losses):
        dict_losses_weighted = {}
        for task in self.tasks:
            loss = dict_losses[task]
            weight = self.weights_dict[task]
            dict_losses_weighted[task] = weight * loss
        return dict_losses_weighted






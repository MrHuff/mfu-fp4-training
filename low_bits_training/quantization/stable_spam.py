import math # Only used for Python-side setup (create_stable_spam)
import torch
from torch.optim.optimizer import Optimizer
import torch.optim as optim

# ==============================================================================
#  PURE FUNCTIONAL KERNEL (Fully Tensorized)
# ==============================================================================
@torch.compile(fullgraph=True)
def stable_spam_functional_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    max_exp_avg_sq: torch.Tensor,
    step: torch.Tensor,       
    # Scalar States (Inputs)
    m_max_t: torch.Tensor,
    m_norm_t: torch.Tensor,
    v_norm_t: torch.Tensor,
    # Hyperparams (ALL TENSORS to prevent recompile on value change)
    lr: torch.Tensor,         
    weight_decay: float,      # Constant 0.0 usually, safe as float
    beta1: torch.Tensor,      # Scaled beta1
    beta2: torch.Tensor,      
    eps: float,               # Constant 1e-8, safe as float
    gamma1: torch.Tensor,     # Scaled gamma1
    gamma2: torch.Tensor,
    theta: torch.Tensor,
    adaclip: bool,            # Boolean flags are fine (static graph branching)
    amsgrad: bool,
):
    # 1. Weight Decay
    current_param = param
    if weight_decay != 0:
        current_param = param * (1 - lr * weight_decay)

    # 2. AdaClip Logic
    next_m_max_t = m_max_t
    processed_grad = grad 

    if adaclip:
        max_gradient = torch.max(grad.abs())
        
        # Tensor Math: next_m = m * theta + g * (1-theta)
        next_m_max_t = m_max_t * theta + max_gradient * (1 - theta)
        
        # Bias correction using torch.pow
        m_max_hat = next_m_max_t / (1 - torch.pow(theta, step))
        
        grad_abs = grad.abs()
        # use torch.where for conditional logic
        scale_factor = torch.where(
            grad_abs > m_max_hat,
            m_max_hat / (max_gradient + 1e-16),
            torch.tensor(1.0, device=grad.device, dtype=grad.dtype)
        )
        processed_grad = grad * scale_factor

    # 3. Norm Scaling Logic
    grad_norm = torch.norm(processed_grad)
    
    # Tensor Math for Norm EMAs
    next_m_norm_t = m_norm_t * gamma1 + grad_norm * (1 - gamma1)
    next_v_norm_t = v_norm_t * gamma2 + (grad_norm**2) * (1 - gamma2)
    
    m_norm_hat = next_m_norm_t / (1 - torch.pow(gamma1, step))
    v_norm_hat = next_v_norm_t / (1 - torch.pow(gamma2, step))
    
    c_norm_t = m_norm_hat / (torch.sqrt(v_norm_hat) + eps)
    
    norm_scaler = torch.where(
        grad_norm > 0,
        c_norm_t / (grad_norm + 1e-16),
        torch.tensor(1.0, device=grad.device, dtype=grad.dtype)
    )
    processed_grad = processed_grad * norm_scaler

    # 4. AdamW Logic
    # Update Moments
    next_exp_avg = exp_avg * beta1 + processed_grad * (1 - beta1)
    next_exp_avg_sq = exp_avg_sq * beta2 + (processed_grad * processed_grad) * (1 - beta2)
    
    # Bias Correction (Pure Tensor Math)
    bias_corr1 = 1 - torch.pow(beta1, step)
    bias_corr2 = 1 - torch.pow(beta2, step)
    
    denom = next_exp_avg_sq.sqrt() / bias_corr2.sqrt() + eps
    
    next_max_exp_avg_sq = max_exp_avg_sq
    if amsgrad:
        if max_exp_avg_sq.numel() > 0:
            next_max_exp_avg_sq = torch.max(max_exp_avg_sq, next_exp_avg_sq)
            denom = next_max_exp_avg_sq.sqrt() / bias_corr2.sqrt() + eps

    step_size = lr / bias_corr1
    
    # 5. Final Param
    next_param = current_param - (next_exp_avg / denom) * step_size

    return (
        next_param, 
        next_exp_avg, 
        next_exp_avg_sq, 
        next_max_exp_avg_sq,
        next_m_max_t, 
        next_m_norm_t, 
        next_v_norm_t
    )

# ==============================================================================
#  OPTIMIZER CLASS
# ==============================================================================

class CosineDecay(object):
    def __init__(self, death_rate, T_max, eta_min=0.5, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=death_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max+1, eta_min, last_epoch)
        self.T_max=T_max
        self.eta_min=eta_min
    def step(self,current_step):
        self.cosine_stepper.step(current_step)

    def get_dr(self,current_step):
        self.step(current_step)
        return self.sgd.param_groups[0]['lr']

class StableSPAM(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, gamma1=0.7, gamma2=0.9, gamma3=0.999,
                 total_T=None, eta_min=0.5, update_proj_gap=1000, adaclip=True):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super(StableSPAM, self).__init__(params, defaults)
        
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.theta = gamma3
        self.total_T = total_T
        if self.total_T is not None:
            self.warmup = CosineDecay(1.0, total_T, eta_min=eta_min)
        self.total_steps = 0
        if self.gamma1 == -1:
            self.gamma1 = betas[0]
        self.update_proj_gap = update_proj_gap
        self.adaclip = adaclip

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.total_steps += 1
        
        # Calculate scale factor (float)
        scale = 1.0
        if self.total_T is not None:
            scale = self.warmup.get_dr(self.total_steps)

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1_val, beta2_val = group['betas']
            eps = group['eps']
            amsgrad = group['amsgrad']

            # Pre-calculate scaled hyperparams as floats
            beta1_scaled_val = beta1_val * scale
            gamma1_scaled_val = self.gamma1 * scale

            for p in group['params']:
                dev, dt = p.device, p.dtype

                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('StableSPAM does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        state['max_exp_avg_sq'] = torch.tensor([], device=p.device)
                    
                    state['m_max_t'] = torch.tensor(0.0, device=dev, dtype=dt)
                    state['m_norm_t'] = torch.tensor(0.0, device=dev, dtype=dt)
                    state['v_norm_t'] = torch.tensor(0.0, device=dev, dtype=dt)

                state['step'] += 1
                
                if self.update_proj_gap != 0 and (self.total_steps % self.update_proj_gap == 0):
                     state['exp_avg'].zero_()
                     state['exp_avg_sq'].zero_()
                     state['step'] = 1 

                # --- TENSORIZE HYPERPARAMS ---
                # We convert all floats to Tensors on the device.
                # This ensures the kernel sees "Tensor inputs" which have a stable type,
                # preventing recompilation when values change (like scale or lr).
                lr_t = torch.tensor(lr, device=p.device, dtype=dt)
                step_t = torch.tensor(state['step'], device=p.device, dtype=dt)
                beta1_t = torch.tensor(beta1_scaled_val, device=p.device, dtype=dt)
                beta2_t = torch.tensor(beta2_val, device=p.device, dtype=dt)
                gamma1_t = torch.tensor(gamma1_scaled_val, device=p.device, dtype=dt)
                gamma2_t = torch.tensor(self.gamma2, device=p.device, dtype=dt)
                theta_t = torch.tensor(self.theta, device=p.device, dtype=dt)

                # Clone state inputs (Break Graph Cycle)
                m_max_in = state['m_max_t'].clone()
                m_norm_in = state['m_norm_t'].clone()
                v_norm_in = state['v_norm_t'].clone()

                # --- CALL COMPILED KERNEL ---
                (next_param, next_exp, next_exp_sq, next_max_sq, 
                 next_m_max, next_m_norm, next_v_norm) = stable_spam_functional_step(
                    p,
                    grad,
                    state['exp_avg'],
                    state['exp_avg_sq'],
                    state['max_exp_avg_sq'],
                    step_t,
                    m_max_in,
                    m_norm_in,
                    v_norm_in,
                    lr_t,
                    weight_decay,
                    beta1_t,
                    beta2_t,
                    eps,
                    gamma1_t,
                    gamma2_t,
                    theta_t,
                    # scale is removed here because we passed scaled gamma/beta tensors
                    self.adaclip,
                    amsgrad
                )
                
                # --- UPDATE STATE ---
                state['exp_avg'].copy_(next_exp)
                state['exp_avg_sq'].copy_(next_exp_sq)
                if amsgrad:
                    state['max_exp_avg_sq'].copy_(next_max_sq)
                
                state['m_max_t'].copy_(next_m_max)
                state['m_norm_t'].copy_(next_m_norm)
                state['v_norm_t'].copy_(next_v_norm)

                p.copy_(next_param)

        return loss


def create_stable_spam(params,**optimizer_kwargs):
    param_groups = [
        {
            "params": params,
            **optimizer_kwargs,
        }
    ]
    return StableSPAM(param_groups)

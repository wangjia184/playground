# https://github.com/DanyeongLee/sde-diffusion-pytorch/blob/main/src/sde.py

import numpy as np
import torch
from sde_base import SDEBase




class VP_SDE(SDEBase):
    '''
    An SDE version of DDPM.
    '''
    def __init__(self, beta_min=0.1, beta_max=20., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    # f(x,t) = - \frac{1}{2} \beta(t) x(t)
    def drift_coef(self, x, t):
        drift = self.beta_t(t)           # drift.shape = [N]
        drift = self.match_dim(drift, x) # drift.shape = [N, 1, 1, 1]
        drift = - drift * x / 2
        return drift
    
    # g(t) = \sqrt{ \beta(t) }
    def diffusion_coef(self, t):
        return torch.sqrt(self.beta_t(t))
    
    # mean of q(x_t|x_0) 
    # = x(0) e^{ -\frac{1}{2} \int_0^t \beta(s) ds }
    # = x(0) e^{ -\frac{1}{2} \beta_{min} t - \frac{1}{4} (\beta_{max} - \beta_{min}) t^2 }
    def x0_coef(self, t):
        exponent = - t**2 * (self.beta_max - self.beta_min) / 4 - t * self.beta_min / 2
        return torch.exp(exponent)
    
    # stddev of q(x_t|x_0) 
    # = \sqrt{ 1 - e^{- \int_0^t \beta(s) ds } }
    # = \sqrt{ 1 - e^{ - \beta_{min} t - \frac{1}{2} (\beta_{max} - \beta_{min}) t^2 } }
    def sigma_t(self, t):
        exponent = - t**2 * (self.beta_max - self.beta_min) / 2 - t * self.beta_min
        return  torch.sqrt( 1 - torch.exp(exponent) )
    




class VE_SDE(SDEBase):
    '''
    An SDE version of NCSN (Variance Exploding SDE).
    '''
    def __init__(self, sigma_min=0.01, sigma_max=1., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # f(x,t) = 0 (no drift)
    def drift_coef(self, x, t):
        return torch.zeros_like(x)

    # g(t) = \sigma(t) \sqrt{2 \log(\sigma_max / \sigma_min)}
    def diffusion_coef(self, t):
        s_t = self.sigma_t(t)
        return s_t * torch.sqrt(2 * torch.log(self.sigma_max / self.sigma_min))

    # \sigma(t) = \sigma_min (\sigma_max / \sigma_min)^t
    def sigma_t(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    # mean of q(x_t|x_0) = x(0) (no scaling)
    def x0_coef(self, t):
        return torch.ones_like(t)


class SubVP_SDE(SDEBase):
    '''
    An SDE version of DDPM with Sub-Variance Preserving (SubVP) schedule.
    '''
    def __init__(self, beta_min=0.1, beta_max=20., eps=1e-5, rescale=True):
        super().__init__(eps, rescale)
        self.beta_min = beta_min
        self.beta_max = beta_max

    # \beta(t) = \beta_min + t (\beta_max - \beta_min)
    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    # \int_0^t \beta(s) ds = \beta_min t + \frac{1}{2} (\beta_max - \beta_min) t^2
    def beta_t_integrated(self, t):
        return self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2

    # f(x,t) = -\frac{1}{2} \beta(t) x(t)
    def drift_coef(self, x, t):
        return -self.beta_t(t) * x / 2

    # g(t) = \sqrt{ \beta(t) (1 - e^{-2 \int_0^t \beta(s) ds}) }
    def diffusion_coef(self, t):
        coef = self.beta_t(t) * (1 - torch.exp(-2 * self.beta_t_integrated(t)))
        return torch.sqrt(coef)

    # mean of q(x_t|x_0) = x(0) e^{-\frac{1}{2} \int_0^t \beta(s) ds}
    def x0_coef(self, t):
        exponent = -self.beta_t_integrated(t) / 2
        return torch.exp(exponent)

    # stddev of q(x_t|x_0) = \sqrt{1 - e^{-\int_0^t \beta(s) ds}}
    def sigma_t(self, t):
        exponent = -self.beta_t_integrated(t)
        return torch.sqrt(1 - torch.exp(exponent))
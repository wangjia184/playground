# VP SDE

## From Discrete DDPM to Continuous Stochastic Differential Equation (SDE)

In discrete DDPM, the forward process is defined by the following formula:

```math
\begin{align*}
x_t &= \sqrt{1-\beta_t} x_{t-1} + \sqrt{\beta_{t}} \epsilon_{t-1} &\quad t \in [1, 1000], 
\epsilon_{t-1} \sim \mathcal{N}(0, I).
\end{align*}
```

There are $N$ timesteps, and each step is indexed by $i$. Thus, the formula can also be expressed as:

```math
\begin{align*}
x_i &= \sqrt{1-\beta_i} x_{i-1} + \sqrt{\beta_{i}} \epsilon_{i-1} &\quad i \in [1, N],
\epsilon_{i-1} \sim \mathcal{N}(0, I).
\end{align*}
```

We introduce a new variable $\bar{\beta_i}$ and define it as $\bar{\beta_i} = N \beta_i$. Consequently, we have: $$\beta_i = \frac{1}{N} \bar{\beta_i}$$

Substituting this into the previous formula yields:

```math
\begin{align*}
x_i &= \sqrt{1-\frac{1}{N} \bar{\beta_i}} x_{i-1} + \sqrt{\frac{1}{N} \bar{\beta_i}} \epsilon_{i-1} &\quad i \in [1, N], 
\epsilon_{i-1} \sim \mathcal{N}(0, I).
\end{align*}
```

As $N \to \infty$, the discrete process transitions into a continuous process for $x(t)$, where $t \in [0.0, 1.0]$. To formalize this, we introduce the following definitions, with $\beta$ now treated as a continuous function:

```math
\begin{align*}
\frac{1}{N} &= \Delta t \\
x_i &= x(\frac{i}{N}) = x(t + \Delta t) \\
\epsilon_i &= \epsilon(\frac{i}{N}) = \epsilon(t + \Delta t) \\
\bar{\beta_i} &= \beta(\frac{i}{N}) = \beta(t + \Delta t) \\
\end{align*}
```

With the above definitions, we can derive the following relationships :

```math
\begin{align*}
x_i &= \sqrt{1-\frac{1}{N} \bar{\beta_i}} x_{i-1} + \sqrt{\frac{1}{N} \bar{\beta_i}} \epsilon_{i-1} \\
x(t+\Delta t) &= \sqrt{1 - \Delta t \beta(t + \Delta t)} x(t) + \sqrt{\Delta t \beta(t + \Delta t)} \epsilon(t) \\
x(t+\Delta t) &= \left[ 1 - \frac{1}{2} \Delta t \beta(t + \Delta t) \right]  x(t) + \sqrt{\Delta t \beta(t + \Delta t)} \epsilon(t) 
 &\text{1st-order Taylor approximation} \\
x(t+\Delta t) &= \left[ 1 - \frac{1}{2} \Delta t \beta(t) \right]  x(t) + \sqrt{\Delta t \beta(t)} \epsilon(t) 
 &\because \beta(t) \text{ is continous} \\
x(t+\Delta t) &=  x(t) - \frac{1}{2} \Delta t \beta(t)  x(t) + \sqrt{\Delta t \beta(t)} \epsilon(t)  \\
x(t+\Delta t) -  x(t) &=  - \frac{1}{2}  \beta(t)  x(t) \Delta t + \sqrt{\beta(t)} \sqrt{\Delta t} \epsilon(t)  \\
dx &= \underbrace{- \frac{1}{2} \beta(t)  x(t) }_{f(x,t)} dt  + \underbrace{\sqrt{\beta(t)}}_{g(t)} dw 
&dw \sim \mathcal{N}(0, dt)
\end{align*}
```

## Reverse Process Sampling

### Reverse time SDE 

```math
dx = [f(x,t)-g(t)^2 \nabla \log p_t(x)]dt + g(t)d \bar{\omega}
```

Here is the simplest euler method 

```python
    @torch.no_grad()
    def euler_sde_step(self, model, x, t, delta_t):
        # Compute the score function (∇ log p_t(x))
        score = self.forward(model, x, t)

        # Compute the drift term (f(x, t)) and diffusion coefficient (g(t))
        drift = self.drift_coef(x, t)
        g = self.diffusion_coef(t)
        g = self.match_dim(g, x)
        
        # Noise term for stochasticity
        if t[0] <= self.eps:
            z = 0 
        else:
            z = torch.randn_like(x) 
        
        # SDE update : dx = [f(x,t)-g(t)^2 ∇ log p_t(x)]dt + g(t) dw
        x_mean = x - (drift - (g**2) * score) * delta_t  # Deterministic update
        x_new = x_mean + g * np.sqrt(delta_t) * z  # dw = (dt)^0.5 * N(I, 0)

        return x_new

    @torch.no_grad()
    def euler_sde_sample(self, model, shape, device, n_steps=500):
        x_t = torch.randn(shape).to(device)

        # Define time steps (from t=1 to t=eps)
        time_steps = np.linspace(1, self.eps, n_steps)
        delta_t = time_steps[0] - time_steps[1] # Time step size

        for t in tqdm(time_steps):
            time_batch = torch.ones(shape[0], device=device) * t
            x_t = self.euler_sde_step(model, x_t, time_batch, delta_t)
        
        return x_t
```

### Reverse time ODE

```math
dx = [f(x,t)- \frac{1}{2} g(t)^2 \nabla \log p_t(x)]dt
```

Here is the simplest euler method 

```python
    @torch.no_grad()
    def euler_ode_step(self, model, x, t, delta_t):
        # Compute the score function (∇ log p_t(x))
        score = self.forward(model, x, t)
        
        # Compute the drift term (f(x, t)) and diffusion coefficient (g(t))
        drift = self.drift_coef(x, t)
        g = self.diffusion_coef(t)
        g = self.match_dim(g, x)
        
        # ODE update: dx = [f(x, t) - 0.5 * g(t)^2 * ∇ log p_t(x)] dt
        dx = (drift - 0.5 * (g**2) * score) * delta_t
        x_new = x - dx  # Note the negative sign for reverse-time ODE
        
        return x_new

    @torch.no_grad()
    def euler_ode_sample(self, model, shape, device, n_steps=500):
        x_t = torch.randn(shape).to(device)
        
        # Define time steps (from t=1 to t=eps)
        time_steps = np.linspace(1, self.eps, n_steps)
        delta_t = time_steps[0] - time_steps[1]  # Time step size
        
        for t in tqdm(time_steps):
            time_batch = torch.ones(shape[0], device=device) * t
            x_t = self.euler_ode_step(model, x_t, time_batch, delta_t)
        
        return x_t
```

## Training

### From Forward SDE to $q(x_t|x_0)$

```math
\begin{align*}
dx &= -\frac{1}{2} \beta(t) x(t) dt + \sqrt{\beta(t)} dw & \text{SDE in standard form} \\
\mu(t) &= \exp\left( \int_0^t f(s) ds \right) = \exp\left( -\frac{1}{2} \int_0^t \beta(s) ds \right) & \text{Integrating factor} \\
d\mu(t) &= -\frac{1}{2} \beta(t) \mu(t) dt & \text{Differential of integrating factor} \\
d(\mu(t) x(t)) &= \mu(t) dx + x(t) d\mu(t) & \text{Product rule for stochastic calculus} \\
d(\mu(t) x(t)) &= \mu(t) \left( -\frac{1}{2} \beta(t) x(t) dt + \sqrt{\beta(t)} dw \right) + x(t) \left( -\frac{1}{2} \beta(t) \mu(t) dt \right) & \text{Substitute } dx \text{ and } d\mu(t) \\
d(\mu(t) x(t)) &= -\frac{1}{2} \beta(t) \mu(t) x(t) dt + \mu(t) \sqrt{\beta(t)} dw - \frac{1}{2} \beta(t) \mu(t) x(t) dt & \text{Expand terms} \\
d(\mu(t) x(t)) &= \mu(t) \sqrt{\beta(t)} dw & \text{Simplify by canceling terms} \\
\mu(t) x(t) &= x(0) + \int_0^t \mu(s) \sqrt{\beta(s)} dw(s) & \text{Integrate both sides} \\
x(t) &= \frac{x(0)}{\mu(t)} + \frac{1}{\mu(t)} \int_0^t \mu(s) \sqrt{\beta(s)} dw(s) & \text{Solve for } x(t) \\
x(t) &= x(0) \exp\left( \frac{1}{2} \int_0^t \beta(s) ds \right) + \exp\left( \frac{1}{2} \int_0^t \beta(s) ds \right) \int_0^t \exp\left( -\frac{1}{2} \int_0^s \beta(u) du \right) \sqrt{\beta(s)} dw(s) & \text{Substitute } \mu(t) \\
\mathbb{E}[x(t)] &= x(0) \exp\left( -\frac{1}{2} \int_0^t \beta(s) ds \right) & \text{Expectation of } x(t) \\
\text{Var}[x(t)] &= \int_0^t \left( \exp\left( -\frac{1}{2} \int_s^t \beta(u) du \right) \sqrt{\beta(s)} \right)^2 ds & \text{Variance of } x(t) \\
\text{Var}[x(t)] &= \int_0^t \exp\left( -\int_s^t \beta(u) du \right) \beta(s) ds & \text{Simplify variance expression} \\
\text{Var}[x(t)] &= 1 - \exp\left( -\int_0^t \beta(s) ds \right) & \text{Evaluate the integral} \\
q(x_t|x_0) &\sim \mathcal{N}\left( x(0) \exp\left( -\frac{1}{2} \int_0^t \beta(s) ds \right), \, 1 - \exp\left( -\int_0^t \beta(s) ds \right) \right) & \text{Final Gaussian distribution}
\end{align*}
```

In Denoising Diffusion Probabilistic Models (DDPM), the noise schedule $\beta(t)$ is defined as follows:

```math
\begin{align*}
\beta(t) &= \bar{\beta}_{min} + t (\bar{\beta}_{max} - \bar{\beta}_{min} )
\end{align*}
```

Here, $\bar{\beta}\_{min}$ and $\bar{\beta}\_{max}$ represent the minimum and maximum values of the noise schedule, respectively. The integral of $\beta(t)$ is used to compute the cumulative noise up to time $t$.

```math
\begin{align*}
\int_0^{t} \beta(s) ds = \bar{\beta}_{min} t + \frac{(\bar{\beta}_{max} - \bar{\beta}_{min} )}{2} t^2
\end{align*}
```

The forward process $q(x_t|x_0)$ in DDPM is defined as a Gaussian distribution:

```math
q(x_t|x_0) \sim \mathcal{N}\left( x(0) 
\exp\left( -\frac{1}{2} \bar{\beta}_{min} t - \frac{(\bar{\beta}_{max} - \bar{\beta}_{min} )}{4} t^2 \right), \,
 1 - \exp\left( -\bar{\beta}_{min} t - \frac{(\bar{\beta}_{max} - \bar{\beta}_{min} )}{2} t^2 \right) \right)
```

This equation describes the distribution of the noisy image \(x_t\) at time \(t\), given the original image \(x_0\). The mean of the distribution is scaled by an exponential decay factor, while the variance is derived from the cumulative noise.


### Loss Function

Given that $q(x_t | x_0)$ is a Gaussian distribution

```math
q(x_t|x_0) \sim \mathcal{N}\left( x(0)
\underbrace{ \exp\left( -\frac{1}{2} \int_0^t \beta(s) ds \right) }_{\mu_t}, \, \underbrace{ 1 - \exp\left( -\int_0^t \beta(s) ds \right) }_{\sigma_t^2} \right) 
```


$\nabla \log q(x_t | x_0)$ can be derived as follows:

```math
\begin{align*}
q(x_t | x_0) &= \frac{1}{\sqrt{2 \pi \sigma_t^2}} \exp\left(-\frac{(x_t - \mu_t x_0)^2}{2 \sigma_t^2}\right) && \text{Normal distribution PDF} \\
\log q(x_t | x_0) &= -\frac{1}{2} \log(2 \pi \sigma_t^2) - \frac{(x_t - \mu_t x_0)^2}{2 \sigma_t^2} && \text{Log PDF} \\
\nabla \log q(x_t | x_0) &= -\frac{x_t - \mu_t x_0}{\sigma_t^2} && \text{Gradient of log PDF} \\
\nabla \log q(x_t | x_0) &= -\frac{\mu_t x_0 + \sigma_t \epsilon - \mu_t x_0}{\sigma_t^2} && \because x_t = \mu_t x_0 + \sigma_t \epsilon \\

\nabla \log q(x_t | x_0) &= -\frac{\epsilon}{\sigma_t}  
\end{align*}
```

Now, we can derive the loss function step by step:

```math
\begin{align*}
\mathcal{L}_{\text{DSM}} &= \left\| s_{\theta}(x_t, t) - \nabla \log q(x_t | x_0) \right\|_2^2 & \text{Original DSM loss} \\
\mathcal{L}_{\text{DSM}} &= \left\| s_{\theta}(x_t, t) + \frac{\epsilon}{\sigma_t} \right\|_2^2 & \because\nabla \log q(x_t | x_0) = -\frac{\epsilon}{\sigma_t}   \\
\mathcal{L}_{\text{DSM}} &= \left\| \sigma_t \cdot s_{\theta}(x_t, t) + \epsilon \right\|_2^2  \\
\end{align*}
```

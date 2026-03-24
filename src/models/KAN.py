# import torch
# import torch.nn as nn

# class UnivariateFunction(nn.Module):
#     """
#     A small MLP representing a single phi_{q,p} or Phi_q function.
#     In a production KAN, this is typically replaced by parameterized B-splines.
#     """
#     def __init__(self, hidden_units=8):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(1, hidden_units),
#             nn.SiLU(), # Smooth activation is critical for functional approximation
#             nn.Linear(hidden_units, 1)
#         )
        
#     def forward(self, x):
#         return self.net(x)

# class KAN(nn.Module):
#     """
#     A rigorously structured 2-Layer Kolmogorov-Arnold Network.
#     Matches the theorem: f(x) = sum_{q} Phi_q ( sum_{p} phi_{q,p}(x_p) )
#     """
#     def __init__(self, in_dim=2, hidden_dim=5, out_dim=1, hidden_per_uni=8):
#         super().__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim # Theorem suggests 2n + 1
#         self.out_dim = out_dim
        
#         # Inner functions: phi_{q,p}
#         # Matrix of functions mapping each input to each hidden node
#         self.inner_funcs = nn.ModuleList([
#             nn.ModuleList([UnivariateFunction(hidden_per_uni) for _ in range(in_dim)])
#             for _ in range(hidden_dim)
#         ])
        
#         # Outer functions: Phi_q
#         # Matrix of functions mapping each hidden node to each output node
#         self.outer_funcs = nn.ModuleList([
#             nn.ModuleList([UnivariateFunction(hidden_per_uni) for _ in range(hidden_dim)])
#             for _ in range(out_dim)
#         ])

#     def forward(self, x):
#         batch_size = x.size(0)
        
#         # 1. Compute Inner Sums (Hidden Nodes)
#         # h_q = sum_{p=1}^{n} phi_{q,p}(x_p)
#         h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
#         for q in range(self.hidden_dim):
#             for p in range(self.in_dim):
#                 x_p = x[:, p:p+1]
#                 h[:, q:q+1] += self.inner_funcs[q][p](x_p)
                
#         # 2. Compute Outer Sums (Final Output)
#         # y_i = sum_{q=1}^{2n+1} Phi_{i,q}(h_q)
#         out = torch.zeros(batch_size, self.out_dim, device=x.device)
#         for i in range(self.out_dim):
#             for q in range(self.hidden_dim):
#                 h_q = h[:, q:q+1]
#                 out[:, i:i+1] += self.outer_funcs[i][q](h_q)
                
#         return out

import torch
import torch.nn as nn
import math

class KAN(nn.Module):
    """
    A fully vectorized 2-Layer KAN using standard tensor broadcasting instead of einsum.
    Retains Sine activations for high-frequency mapping capacity.
    """
    def __init__(self, in_dim=2, hidden_dim=5, out_dim=1, hidden_per_uni=8):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hidden_per_uni = hidden_per_uni

        # --- INNER FUNCTIONS PARAMETERS ---
        # Shape: (in_dim, hidden_dim, hidden_per_uni)
        self.w1_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim, hidden_per_uni))
        self.b1_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim, hidden_per_uni))
        self.w2_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim, hidden_per_uni))
        self.b2_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim))

        # --- OUTER FUNCTIONS PARAMETERS ---
        # Shape: (hidden_dim, out_dim, hidden_per_uni)
        self.w1_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim, hidden_per_uni))
        self.b1_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim, hidden_per_uni))
        self.w2_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim, hidden_per_uni))
        self.b2_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim))

        self._initialize_weights()

    # def _initialize_weights(self):
    #     """Standard uniform initialization."""
    #     for weight in [self.w1_inner, self.w2_inner, self.w1_outer, self.w2_outer]:
    #         nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    #     for bias in [self.b1_inner, self.b2_inner, self.b1_outer, self.b2_outer]:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w1_inner)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         nn.init.uniform_(bias, -bound, bound)

    def _initialize_weights(self):
        """
        Frequency-scaled initialization (SIREN-style) to prevent linear collapse
        and unlock the periodic mapping capacity of the sine activations.
        """
        omega_0 = 10.0 # Base spatial frequency multiplier
        
        with torch.no_grad():
            # 1. Inner Functions (First Layer)
            # We scale the weights by omega_0 to ensure the inputs span multiple sine periods
            fan_in_inner = self.in_dim
            bound_inner = 1 / fan_in_inner
            nn.init.uniform_(self.w1_inner, -bound_inner, bound_inner)
            self.w1_inner.data *= omega_0 
            
            # Biases initialized to span a full phase shift
            nn.init.uniform_(self.b1_inner, -math.pi, math.pi)

            # 2. Outer Functions (Hidden Layers)
            # Standard SIREN hidden initialization to maintain variance
            fan_in_outer = self.hidden_dim
            bound_outer = math.sqrt(6 / fan_in_outer) / omega_0
            nn.init.uniform_(self.w1_outer, -bound_outer, bound_outer)
            nn.init.uniform_(self.b1_outer, -math.pi, math.pi)
            
            # 3. Linear Projection Layers (w2, b2)
            # These sum the sine outputs, standard initialization is fine
            nn.init.kaiming_uniform_(self.w2_inner, a=math.sqrt(5))
            nn.init.zeros_(self.b2_inner)
            nn.init.kaiming_uniform_(self.w2_outer, a=math.sqrt(5))
            nn.init.zeros_(self.b2_outer)

    def forward(self, x):
        b = x.size(0)

        # ==========================================
        # 1. Compute Inner Sums (Hidden Nodes)
        # ==========================================
        # Broadcast x: (batch, in_dim, 1, 1) against w1: (1, in_dim, hidden, hidden_per_uni)
        x_expanded = x.view(b, self.in_dim, 1, 1)
        
        h1 = x_expanded * self.w1_inner.unsqueeze(0) + self.b1_inner.unsqueeze(0)
        a1 = torch.sin(h1) 
        
        # Multiply by w2 and sum across the hidden_per_uni dimension (dim=-1)
        out_inner = (a1 * self.w2_inner.unsqueeze(0)).sum(dim=-1) + self.b2_inner.unsqueeze(0)
        
        # Sum across input dimensions (dim=1) to get the intermediate hidden nodes
        h_nodes = out_inner.sum(dim=1) # Shape: (batch, hidden_dim)

        # ==========================================
        # 2. Compute Outer Sums (Final Output)
        # ==========================================
        # Broadcast h_nodes: (batch, hidden_dim, 1, 1)
        h_expanded = h_nodes.view(b, self.hidden_dim, 1, 1)
        
        h2 = h_expanded * self.w1_outer.unsqueeze(0) + self.b1_outer.unsqueeze(0)
        a2 = torch.sin(h2)
        
        # Multiply by w2 and sum across the hidden_per_uni dimension (dim=-1)
        out_outer = (a2 * self.w2_outer.unsqueeze(0)).sum(dim=-1) + self.b2_outer.unsqueeze(0)
        
        # Sum across hidden dimensions (dim=1) to get the final output
        y = out_outer.sum(dim=1) # Shape: (batch, out_dim)
        
        return y
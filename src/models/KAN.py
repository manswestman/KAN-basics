
import torch
import torch.nn as nn
import math

class KAN(nn.Module):

    def __init__(self, in_dim=2, hidden_dim=5, out_dim=1, hidden_per_uni=8):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hidden_per_uni = hidden_per_uni

        # Inner functions
        self.w1_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim, hidden_per_uni))
        self.b1_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim, hidden_per_uni))
        self.w2_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim, hidden_per_uni))
        self.b2_inner = nn.Parameter(torch.Tensor(in_dim, hidden_dim))

        # Outer functions
        self.w1_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim, hidden_per_uni))
        self.b1_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim, hidden_per_uni))
        self.w2_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim, hidden_per_uni))
        self.b2_outer = nn.Parameter(torch.Tensor(hidden_dim, out_dim))

        self._initialize_weights()


    def _initialize_weights(self):

        # Frequency-scaled initialization 

        omega_0 = 10.0 # Base spatial frequency multiplier
        
        with torch.no_grad():
            
            # Inner Functions (First Layer)
            # We scale the weights by omega_0 to ensure the inputs span multiple sine periods
            fan_in_inner = self.in_dim
            bound_inner = 1 / fan_in_inner
            nn.init.uniform_(self.w1_inner, -bound_inner, bound_inner)
            self.w1_inner.data *= omega_0 
            
            # Biases initialized to span a full phase shift
            nn.init.uniform_(self.b1_inner, -math.pi, math.pi)

            # Outer Functions (Hidden Layers)
            # Standard SIREN hidden initialization to maintain variance
            fan_in_outer = self.hidden_dim
            bound_outer = math.sqrt(6 / fan_in_outer) / omega_0
            nn.init.uniform_(self.w1_outer, -bound_outer, bound_outer)
            nn.init.uniform_(self.b1_outer, -math.pi, math.pi)
            
            # Linear Projection Layers (w2, b2)
            # These sum the sine outputs, standard initialization is fine
            nn.init.kaiming_uniform_(self.w2_inner, a=math.sqrt(5))
            nn.init.zeros_(self.b2_inner)
            nn.init.kaiming_uniform_(self.w2_outer, a=math.sqrt(5))
            nn.init.zeros_(self.b2_outer)

    def forward(self, x):
        
        b = x.size(0)

        # Compute Inner Sums (Hidden Nodes)

        # Broadcast x: (batch, in_dim, 1, 1) against w1: (1, in_dim, hidden, hidden_per_uni)
        x_expanded = x.view(b, self.in_dim, 1, 1)
        
        h1 = x_expanded * self.w1_inner.unsqueeze(0) + self.b1_inner.unsqueeze(0)
        a1 = torch.sin(h1) 
        
        # Multiply by w2 and sum across the hidden_per_uni dimension (dim=-1)
        out_inner = (a1 * self.w2_inner.unsqueeze(0)).sum(dim=-1) + self.b2_inner.unsqueeze(0)
        
        # Sum across input dimensions (dim=1) to get the intermediate hidden nodes
        h_nodes = out_inner.sum(dim=1) # Shape: (batch, hidden_dim)


        # Compute Outer Sums (Final Output)

        # Broadcast h_nodes: (batch, hidden_dim, 1, 1)
        h_expanded = h_nodes.view(b, self.hidden_dim, 1, 1)
        
        h2 = h_expanded * self.w1_outer.unsqueeze(0) + self.b1_outer.unsqueeze(0)
        a2 = torch.sin(h2)
        
        # Multiply by w2 and sum across the hidden_per_uni dimension (dim=-1)
        out_outer = (a2 * self.w2_outer.unsqueeze(0)).sum(dim=-1) + self.b2_outer.unsqueeze(0)
        
        # Sum across hidden dimensions (dim=1) to get the final output
        y = out_outer.sum(dim=1) # Shape: (batch, out_dim)
        
        return y
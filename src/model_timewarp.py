import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

class AtomEmbedder(nn.Module):
    """Atom embedding layer"""
    def __init__(self, num_atom_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)

    def forward(self, atom_types: Tensor) -> Tensor:
        """
        Args:
            atom_types: [batch_size, num_atoms] - Atom type indices
        Returns:
            [batch_size, num_atoms, embedding_dim] - Atom embeddings
        """
        return self.embedding(atom_types)

class KernelSelfAttention(nn.Module):
    """
    Kernel Self-Attention (RBF kernel-based self-attention)
    Paper equations (10) and (11)
    """
    def __init__(self, input_dim: int, output_dim: int, lengthscales: list):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lengthscales = lengthscales
        self.num_heads = len(lengthscales)

        # Ensure output_dim is divisible by num_heads
        self.head_dim = output_dim // self.num_heads
        if output_dim % self.num_heads != 0:
            # Adjust head_dim to ensure dimension matching
            self.head_dim = output_dim // self.num_heads + 1
            print(f"Warning: Adjusting head_dim from {output_dim // self.num_heads} to {self.head_dim} to ensure divisibility")

        self.total_head_dim = self.head_dim * self.num_heads

        # Value transformation matrix V for each head
        self.value_projections = nn.ModuleList([
            nn.Linear(input_dim, self.head_dim)
            for _ in range(self.num_heads)
        ])

        # Final output projection - from concatenated head dimensions to expected output dimension
        self.output_projection = nn.Linear(self.total_head_dim, output_dim)

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, input_dim] - Input features
            coords: [batch_size, num_atoms, 3] - Atom coordinates
        Returns:
            [batch_size, num_atoms, output_dim] - Output features
        """
        batch_size, num_atoms, _ = features.shape

        # Compute pairwise distance matrix - Paper equation (10)
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        distances_sq = torch.sum((coords_i - coords_j) ** 2, dim=-1)  # [B, N, N]

        # Multi-head attention
        head_outputs = []
        for head_idx, lengthscale in enumerate(self.lengthscales):
            # Compute RBF kernel attention weights (Equation 10)
            attention_weights = torch.exp(-distances_sq / (lengthscale ** 2))  # [B, N, N]

            # Normalize weights
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # Apply attention (Equation 11)
            value = self.value_projections[head_idx](features)  # [B, N, head_dim]
            attended_features = torch.bmm(attention_weights, value)  # [B, N, head_dim]
            head_outputs.append(attended_features)

        # Concatenate multi-head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, N, total_head_dim]

        # Final projection to expected output dimension
        return self.output_projection(multi_head_output)

class AtomTransformer(nn.Module):
    """
    Atom Transformer - Core component in the paper, used as s_θ and t_θ functions
    Paper Section 4 and Figure 2 Middle
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # φ_in: Input MLP
        # Input is [x_p_i(t), h_i, z_v_i] or [x_p_i(t), h_i, z_p_i] - Paper Section 4
        # We also need to include the velocity information z_v or z_p
        self.input_mlp = nn.Sequential(
            nn.Linear(3 + embedding_dim + 3, hidden_dim),  # coords + embedding + latent (pos or vel)
            nn.ReLU()
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, lengthscales)
            for _ in range(num_layers)
        ])

        # φ_out: Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Output 3D vector
        )

    def forward(self, latent_vars: Tensor, x_coords: Tensor, atom_embeddings: Tensor) -> Tensor:
        """
        Args:
            latent_vars: [B, N, 3] - z_v or z_p
            x_coords: [B, N, 3] - Conditional coordinates x_p(t)
            atom_embeddings: [B, N, embedding_dim] - Atom embeddings h_i
        Returns:
            [B, N, 3] - Scale or shift vector
        """
        # Concatenate inputs: [x_p_i(t), h_i, z_v_i] - Paper Section 4
        input_features = torch.cat([x_coords, atom_embeddings, latent_vars], dim=-1)

        # φ_in
        features = self.input_mlp(input_features)

        # Transformer layers - using x_coords for kernel attention
        for layer in self.transformer_layers:
            features = layer(features, x_coords)

        # φ_out
        output = self.output_mlp(features)

        return output

class TransformerBlock(nn.Module):
    """Transformer block (containing Kernel Self-Attention)"""
    def __init__(self, hidden_dim: int, lengthscales: list):
        super().__init__()
        self.kernel_attention = KernelSelfAttention(hidden_dim, hidden_dim, lengthscales)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network - Paper refers to as "atom-wise MLP"
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, hidden_dim]
            coords: [batch_size, num_atoms, 3]
        Returns:
            [batch_size, num_atoms, hidden_dim]
        """
        # Self-attention + residual connection + norm
        attended = self.kernel_attention(features, coords)
        features = self.norm1(features + attended)

        # Feed-forward + residual connection + norm
        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)

        return features

class TimewarpCouplingLayer(nn.Module):
    """
    Timewarp RealNVP coupling layer - Paper equations (8) and (9)
    This is the core innovation of the paper: using Atom Transformer as s_θ and t_θ functions
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list):
        super().__init__()

        # Atom Transformers for position transformations - Paper equation (8)
        self.scale_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

        # Atom Transformers for velocity transformations - Paper equation (9)
        self.scale_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

    def forward(self, z_p: Tensor, z_v: Tensor, x_coords: Tensor,
                atom_embeddings: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z_p: [B, N, 3] - Position latent variable
            z_v: [B, N, 3] - Velocity latent variable
            x_coords: [B, N, 3] - Conditional coordinates x^p(t)
            atom_embeddings: [B, N, embedding_dim] - Atom embeddings
            reverse: Whether to use reverse (sampling) mode
        Returns:
            z_p_new, z_v_new, log_det_jacobian
        """
        if not reverse:
            # Forward pass - Paper equations (8) and (9)

            # Step 1: Transform positions - z^p_{ℓ+1} = s^p_{ℓ,θ}(z^v_ℓ; x^p(t)) ⊙ z^p_ℓ + t^p_{ℓ,θ}(z^v_ℓ; x^p(t))
            scale_p = self.scale_transformer_p(z_v, x_coords, atom_embeddings)  # s^p_{ℓ,θ}(z^v_ℓ; x^p(t))
            shift_p = self.shift_transformer_p(z_v, x_coords, atom_embeddings)  # t^p_{ℓ,θ}(z^v_ℓ; x^p(t))

            z_p_new = torch.exp(scale_p) * z_p + shift_p
            log_det_p = scale_p.sum(dim=-1)  # [B, N]

            # Step 2: Transform velocities - z^v_{ℓ+1} = s^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t)) ⊙ z^v_ℓ + t^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))
            scale_v = self.scale_transformer_v(z_p_new, x_coords, atom_embeddings)  # s^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))
            shift_v = self.shift_transformer_v(z_p_new, x_coords, atom_embeddings)  # t^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))

            z_v_new = torch.exp(scale_v) * z_v + shift_v
            log_det_v = scale_v.sum(dim=-1)  # [B, N]

            total_log_det = log_det_p + log_det_v  # [B, N]

        else:
            # Reverse pass (sampling)

            # Step 1: Reverse transform velocities
            scale_v = self.scale_transformer_v(z_p, x_coords, atom_embeddings)
            shift_v = self.shift_transformer_v(z_p, x_coords, atom_embeddings)

            z_v_new = (z_v - shift_v) * torch.exp(-scale_v)
            log_det_v = -scale_v.sum(dim=-1)

            # Step 2: Reverse transform positions
            scale_p = self.scale_transformer_p(z_v_new, x_coords, atom_embeddings)
            shift_p = self.shift_transformer_p(z_v_new, x_coords, atom_embeddings)

            z_p_new = (z_p - shift_p) * torch.exp(-scale_p)
            log_det_p = -scale_p.sum(dim=-1)

            total_log_det = log_det_p + log_det_v

        return z_p_new, z_v_new, total_log_det

class TimewarpModel(nn.Module):
    """
    Complete Timewarp model - Strictly following the paper implementation
    Core idea: Use conditional normalizing flow to learn μ(x(t+τ)|x(t))
    """
    def __init__(
        self,
        num_atom_types: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_coupling_layers: int = 12,
        lengthscales: list = [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]
    ):
        super().__init__()

        # 1. Atom embedder
        self.atom_embedder = AtomEmbedder(num_atom_types, embedding_dim)

        # 2. RealNVP coupling layer stack - Paper Figure 2 Left
        self.coupling_layers = nn.ModuleList([
            TimewarpCouplingLayer(embedding_dim, hidden_dim, lengthscales)
            for _ in range(num_coupling_layers)
        ])

        # 3. Base distribution scale parameter (learnable)
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1)))

    def forward(
        self,
        atom_types: Tensor,      # [batch_size, num_atoms] - Atom types
        x_coords: Tensor,        # [batch_size, num_atoms, 3] - Conditional coordinates x^p(t)
        x_velocs: Tensor,        # [batch_size, num_atoms, 3] - Conditional velocities x^v(t)
        y_coords: Tensor = None, # [batch_size, num_atoms, 3] - Target coordinates x^p(t+τ) (training)
        y_velocs: Tensor = None, # [batch_size, num_atoms, 3] - Target velocities x^v(t+τ) (training)
        reverse: bool = False    # Whether in sampling mode
    ) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        """
        Args:
            atom_types: Atom type indices
            x_coords: Conditional coordinates x^p(t)
            x_velocs: Conditional velocities x^v(t)
            y_coords: Target coordinates x^p(t+τ) (used during training)
            y_velocs: Target velocities x^v(t+τ) (used during training)
            reverse: False=training mode, True=sampling mode
        Returns:
            output_state: (output_coords, output_velocs)
            log_likelihood: Log-likelihood (training only)
        """
        batch_size, num_atoms = atom_types.shape

        # 1. Atom embedding - Paper Section 4
        atom_embeddings = self.atom_embedder(atom_types)  # [B, N, embedding_dim]

        # 2. Center coordinates (translation equivariance) - Paper Appendix A.2
        x_coords_centered = self._center_coordinates(x_coords)

        if not reverse:
            # Training mode: Compute p_θ(x(t+τ)|x(t))
            if y_coords is None or y_velocs is None:
                raise ValueError("Training mode requires target coordinates and velocities y_coords, y_velocs")

            # Center target coordinates
            y_coords_centered = self._center_coordinates(y_coords)

            # Sample auxiliary variable - Paper Section 3.3 Augmented Normalizing Flows
            z_v = y_velocs # Use target velocity as auxiliary variable
            z_p = y_coords_centered  # Use centered target position as main variable

            # Through coupling layers (forward)
            total_log_det = torch.zeros(batch_size, num_atoms, device=x_coords.device)

            for layer in self.coupling_layers:
                z_p, z_v, log_det = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=False)
                total_log_det += log_det

            # Compute log probability of base distribution - N(0, σ²I)
            scale = torch.exp(self.log_scale)
            log_prior_p = -0.5 * torch.sum((z_p / scale) ** 2, dim=-1)  # [B, N]
            log_prior_v = -0.5 * torch.sum((z_v / scale) ** 2, dim=-1)  # [B, N]
            log_prior = log_prior_p + log_prior_v

            # Total log-likelihood
            log_likelihood = log_prior + total_log_det  # [B, N]

            return (y_coords, y_velocs), log_likelihood

        else:
            # Sampling mode: Generate x(t+τ) ~ p_θ(·|x(t))

            # Sample from base distribution
            scale = torch.exp(self.log_scale)
            z_p = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale
            z_v = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale

            # Through coupling layers (reverse)
            for layer in reversed(self.coupling_layers):
                z_p, z_v, _ = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=True)

            # z_p is now centered output coordinates, z_v is output velocity
            output_coords = self._uncenter_coordinates(z_p, x_coords)
            output_velocs = z_v

            return (output_coords, output_velocs), None

    def _center_coordinates(self, coords: Tensor) -> Tensor:
        """Center coordinates - Paper Appendix A.2"""
        centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
        return coords - centroid

    def _uncenter_coordinates(self, centered_coords: Tensor, reference_coords: Tensor) -> Tensor:
        """Restore coordinate center"""
        reference_centroid = reference_coords.mean(dim=1, keepdim=True)
        return centered_coords + reference_centroid

    def sample(self, atom_types: Tensor, x_coords: Tensor, x_velocs: Tensor, num_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """Convenient sampling interface"""
        self.eval()
        with torch.no_grad():
            if num_samples == 1:
                (output_coords, output_velocs), _ = self.forward(atom_types, x_coords, x_velocs, reverse=True)
                return output_coords, output_velocs
            else:
                # Batch sampling
                samples_coords = []
                samples_velocs = []
                for _ in range(num_samples):
                    (output_coords, output_velocs), _ = self.forward(atom_types, x_coords, x_velocs, reverse=True)
                    samples_coords.append(output_coords)
                    samples_velocs.append(output_velocs)
                return torch.stack(samples_coords, dim=0), torch.stack(samples_velocs, dim=0)


def create_timewarp_model(config: dict) -> TimewarpModel:
    """Factory function to create Timewarp model"""
    return TimewarpModel(
        num_atom_types=config.get('num_atom_types', 10),
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_coupling_layers=config.get('num_coupling_layers', 12),
        lengthscales=config.get('lengthscales', [0.1, 0.2, 0.5, 0.7, 1.0, 1.2])
    )

# Configuration parameters from the paper
paper_config = {
    'num_atom_types': 20,        # 20 amino acid types
    'embedding_dim': 64,         # Paper Table 3
    'hidden_dim': 128,           # Paper Table 3
    'num_coupling_layers': 12,   # Paper Table 3 - AD dataset
    'lengthscales': [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]  # Paper Appendix F
}

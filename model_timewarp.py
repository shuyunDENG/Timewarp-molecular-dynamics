import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

class AtomEmbedder(nn.Module):
    """原子嵌入层"""
    def __init__(self, num_atom_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)

    def forward(self, atom_types: Tensor) -> Tensor:
        """
        Args:
            atom_types: [batch_size, num_atoms] - 原子类型索引
        Returns:
            [batch_size, num_atoms, embedding_dim] - 原子嵌入
        """
        return self.embedding(atom_types)

class KernelSelfAttention(nn.Module):
    """
    Kernel Self-Attention (基于 RBF 核的自注意力)
    论文方程 (10) 和 (11)
    """
    def __init__(self, input_dim: int, output_dim: int, lengthscales: list):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lengthscales = lengthscales
        self.num_heads = len(lengthscales)

        # 确保 output_dim 能被 num_heads 整除
        self.head_dim = output_dim // self.num_heads
        if output_dim % self.num_heads != 0:
            # 调整 head_dim 确保维度匹配
            self.head_dim = output_dim // self.num_heads + 1
            print(f"Warning: Adjusting head_dim from {output_dim // self.num_heads} to {self.head_dim} to ensure divisibility")

        self.total_head_dim = self.head_dim * self.num_heads

        # 每个头的变换矩阵 V
        self.value_projections = nn.ModuleList([
            nn.Linear(input_dim, self.head_dim)
            for _ in range(self.num_heads)
        ])

        # 最终的输出投影 - 从拼接的头维度到期望的输出维度
        self.output_projection = nn.Linear(self.total_head_dim, output_dim)

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        """
        Args:
            features: [batch_size, num_atoms, input_dim] - 输入特征
            coords: [batch_size, num_atoms, 3] - 原子坐标
        Returns:
            [batch_size, num_atoms, output_dim] - 输出特征
        """
        batch_size, num_atoms, _ = features.shape

        # 计算原子间距离矩阵 - 论文方程 (10)
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        distances_sq = torch.sum((coords_i - coords_j) ** 2, dim=-1)  # [B, N, N]

        # 多头注意力
        head_outputs = []
        for head_idx, lengthscale in enumerate(self.lengthscales):
            # 计算 RBF 核注意力权重 (方程 10)
            attention_weights = torch.exp(-distances_sq / (lengthscale ** 2))  # [B, N, N]

            # 归一化权重
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

            # 应用注意力 (方程 11)
            value = self.value_projections[head_idx](features)  # [B, N, head_dim]
            attended_features = torch.bmm(attention_weights, value)  # [B, N, head_dim]
            head_outputs.append(attended_features)

        # 拼接多头输出
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, N, total_head_dim]

        # 最终投影到期望的输出维度
        return self.output_projection(multi_head_output)

class AtomTransformer(nn.Module):
    """
    Atom Transformer - 论文中的核心组件，用作 s_θ 和 t_θ 函数
    论文 Section 4 和 Figure 2 Middle
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # φ_in: 输入 MLP
        # 输入是 [x_p_i(t), h_i, z_v_i] 或 [x_p_i(t), h_i, z_p_i] - 论文 Section 4
        # We also need to include the velocity information z_v or z_p
        self.input_mlp = nn.Sequential(
            nn.Linear(3 + embedding_dim + 3, hidden_dim),  # coords + embedding + latent (pos or vel)
            nn.ReLU()
        )

        # Transformer 层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, lengthscales)
            for _ in range(num_layers)
        ])

        # φ_out: 输出 MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 输出 3D 向量
        )

    def forward(self, latent_vars: Tensor, x_coords: Tensor, atom_embeddings: Tensor) -> Tensor:
        """
        Args:
            latent_vars: [B, N, 3] - z_v 或 z_p
            x_coords: [B, N, 3] - 条件坐标 x_p(t)
            atom_embeddings: [B, N, embedding_dim] - 原子嵌入 h_i
        Returns:
            [B, N, 3] - scale 或 shift 向量
        """
        # 拼接输入：[x_p_i(t), h_i, z_v_i] - 论文 Section 4
        input_features = torch.cat([x_coords, atom_embeddings, latent_vars], dim=-1)

        # φ_in
        features = self.input_mlp(input_features)

        # Transformer 层 - 使用 x_coords 进行 kernel attention
        for layer in self.transformer_layers:
            features = layer(features, x_coords)

        # φ_out
        output = self.output_mlp(features)

        return output

class TransformerBlock(nn.Module):
    """Transformer 块 (包含 Kernel Self-Attention)"""
    def __init__(self, hidden_dim: int, lengthscales: list):
        super().__init__()
        self.kernel_attention = KernelSelfAttention(hidden_dim, hidden_dim, lengthscales)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward 网络 - 论文称为 "atom-wise MLP"
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
    Timewarp RealNVP 耦合层 - 论文方程 (8) 和 (9)
    这是论文的核心创新：使用 Atom Transformer 作为 s_θ 和 t_θ 函数
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list):
        super().__init__()

        # 用于位置变换的 Atom Transformers - 论文方程 (8)
        self.scale_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

        # 用于速度变换的 Atom Transformers - 论文方程 (9)
        self.scale_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

    def forward(self, z_p: Tensor, z_v: Tensor, x_coords: Tensor,
                atom_embeddings: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            z_p: [B, N, 3] - 位置潜在变量
            z_v: [B, N, 3] - 速度潜在变量
            x_coords: [B, N, 3] - 条件坐标 x^p(t)
            atom_embeddings: [B, N, embedding_dim] - 原子嵌入
            reverse: 是否反向传播
        Returns:
            z_p_new, z_v_new, log_det_jacobian
        """
        if not reverse:
            # 前向传播 - 论文方程 (8) 和 (9)

            # 步骤1：变换位置 - z^p_{ℓ+1} = s^p_{ℓ,θ}(z^v_ℓ; x^p(t)) ⊙ z^p_ℓ + t^p_{ℓ,θ}(z^v_ℓ; x^p(t))
            scale_p = self.scale_transformer_p(z_v, x_coords, atom_embeddings)  # s^p_{ℓ,θ}(z^v_ℓ; x^p(t))
            shift_p = self.shift_transformer_p(z_v, x_coords, atom_embeddings)  # t^p_{ℓ,θ}(z^v_ℓ; x^p(t))

            z_p_new = torch.exp(scale_p) * z_p + shift_p
            log_det_p = scale_p.sum(dim=-1)  # [B, N]

            # 步骤2：变换速度 - z^v_{ℓ+1} = s^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t)) ⊙ z^v_ℓ + t^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))
            scale_v = self.scale_transformer_v(z_p_new, x_coords, atom_embeddings)  # s^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))
            shift_v = self.shift_transformer_v(z_p_new, x_coords, atom_embeddings)  # t^v_{ℓ,θ}(z^p_{ℓ+1}; x^p(t))

            z_v_new = torch.exp(scale_v) * z_v + shift_v
            log_det_v = scale_v.sum(dim=-1)  # [B, N]

            total_log_det = log_det_p + log_det_v  # [B, N]

        else:
            # 反向传播 (采样)

            # 步骤1：反向变换速度
            scale_v = self.scale_transformer_v(z_p, x_coords, atom_embeddings)
            shift_v = self.shift_transformer_v(z_p, x_coords, atom_embeddings)

            z_v_new = (z_v - shift_v) * torch.exp(-scale_v)
            log_det_v = -scale_v.sum(dim=-1)

            # 步骤2：反向变换位置
            scale_p = self.scale_transformer_p(z_v_new, x_coords, atom_embeddings)
            shift_p = self.shift_transformer_p(z_v_new, x_coords, atom_embeddings)

            z_p_new = (z_p - shift_p) * torch.exp(-scale_p)
            log_det_p = -scale_p.sum(dim=-1)

            total_log_det = log_det_p + log_det_v

        return z_p_new, z_v_new, total_log_det

class TimewarpModel(nn.Module):
    """
    完整的 Timewarp 模型 - 严格按照论文实现
    核心思想：使用 conditional normalizing flow 学习 μ(x(t+τ)|x(t))
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

        # 1. 原子嵌入器
        self.atom_embedder = AtomEmbedder(num_atom_types, embedding_dim)

        # 2. RealNVP 耦合层堆叠 - 论文 Figure 2 Left
        self.coupling_layers = nn.ModuleList([
            TimewarpCouplingLayer(embedding_dim, hidden_dim, lengthscales)
            for _ in range(num_coupling_layers)
        ])

        # 3. 基础分布的尺度参数 (可学习)
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1)))

    def forward(
        self,
        atom_types: Tensor,      # [batch_size, num_atoms] - 原子类型
        x_coords: Tensor,        # [batch_size, num_atoms, 3] - 条件坐标 x^p(t)
        x_velocs: Tensor,        # [batch_size, num_atoms, 3] - 条件速度 x^v(t)
        y_coords: Tensor = None, # [batch_size, num_atoms, 3] - 目标坐标 x^p(t+τ) (训练时)
        y_velocs: Tensor = None, # [batch_size, num_atoms, 3] - 目标速度 x^v(t+τ) (训练时)
        reverse: bool = False    # 是否为采样模式
    ) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        """
        Args:
            atom_types: 原子类型索引
            x_coords: 条件坐标 x^p(t)
            x_velocs: 条件速度 x^v(t)
            y_coords: 目标坐标 x^p(t+τ) (训练时使用)
            y_velocs: 目标速度 x^v(t+τ) (训练时使用)
            reverse: False=训练模式, True=采样模式
        Returns:
            output_state: (output_coords, output_velocs)
            log_likelihood: 对数似然 (仅训练时)
        """
        batch_size, num_atoms = atom_types.shape

        # 1. 原子嵌入 - 论文 Section 4
        atom_embeddings = self.atom_embedder(atom_types)  # [B, N, embedding_dim]

        # 2. 中心化坐标 (translation equivariance) - 论文 Appendix A.2
        x_coords_centered = self._center_coordinates(x_coords)

        if not reverse:
            # 训练模式: 计算 p_θ(x(t+τ)|x(t))
            if y_coords is None or y_velocs is None:
                raise ValueError("训练模式需要提供目标坐标和速度 y_coords, y_velocs")

            # 中心化目标坐标
            y_coords_centered = self._center_coordinates(y_coords)

            # 采样辅助变量 - 论文 Section 3.3 Augmented Normalizing Flows
            z_v = y_velocs # Use target velocity as auxiliary variable
            z_p = y_coords_centered  # Use centered target position as main variable

            # 通过耦合层 (前向)
            total_log_det = torch.zeros(batch_size, num_atoms, device=x_coords.device)

            for layer in self.coupling_layers:
                z_p, z_v, log_det = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=False)
                total_log_det += log_det

            # 计算基础分布的对数概率 - N(0, σ²I)
            scale = torch.exp(self.log_scale)
            log_prior_p = -0.5 * torch.sum((z_p / scale) ** 2, dim=-1)  # [B, N]
            log_prior_v = -0.5 * torch.sum((z_v / scale) ** 2, dim=-1)  # [B, N]
            log_prior = log_prior_p + log_prior_v

            # 总对数似然
            log_likelihood = log_prior + total_log_det  # [B, N]

            return (y_coords, y_velocs), log_likelihood

        else:
            # 采样模式：生成 x(t+τ) ~ p_θ(·|x(t))

            # 从基础分布采样
            scale = torch.exp(self.log_scale)
            z_p = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale
            z_v = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale

            # 通过耦合层 (反向)
            for layer in reversed(self.coupling_layers):
                z_p, z_v, _ = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=True)

            # z_p is now centered output coordinates, z_v is output velocity
            output_coords = self._uncenter_coordinates(z_p, x_coords)
            output_velocs = z_v

            return (output_coords, output_velocs), None

    def _center_coordinates(self, coords: Tensor) -> Tensor:
        """中心化坐标 - 论文 Appendix A.2"""
        centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
        return coords - centroid

    def _uncenter_coordinates(self, centered_coords: Tensor, reference_coords: Tensor) -> Tensor:
        """恢复坐标中心"""
        reference_centroid = reference_coords.mean(dim=1, keepdim=True)
        return centered_coords + reference_centroid

    def sample(self, atom_types: Tensor, x_coords: Tensor, x_velocs: Tensor, num_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """便捷的采样接口"""
        self.eval()
        with torch.no_grad():
            if num_samples == 1:
                (output_coords, output_velocs), _ = self.forward(atom_types, x_coords, x_velocs, reverse=True)
                return output_coords, output_velocs
            else:
                # 批量采样
                samples_coords = []
                samples_velocs = []
                for _ in range(num_samples):
                    (output_coords, output_velocs), _ = self.forward(atom_types, x_coords, x_velocs, reverse=True)
                    samples_coords.append(output_coords)
                    samples_velocs.append(output_velocs)
                return torch.stack(samples_coords, dim=0), torch.stack(samples_velocs, dim=0)


def create_timewarp_model(config: dict) -> TimewarpModel:
    """创建 Timewarp 模型的工厂函数"""
    return TimewarpModel(
        num_atom_types=config.get('num_atom_types', 10),
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_coupling_layers=config.get('num_coupling_layers', 12),
        lengthscales=config.get('lengthscales', [0.1, 0.2, 0.5, 0.7, 1.0, 1.2])
    )

# 论文中的配置参数
paper_config = {
    'num_atom_types': 20,        # 20种氨基酸
    'embedding_dim': 64,         # 论文 Table 3
    'hidden_dim': 128,           # 论文 Table 3
    'num_coupling_layers': 12,   # 论文 Table 3 - AD dataset
    'lengthscales': [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]  # 论文 Appendix F
}

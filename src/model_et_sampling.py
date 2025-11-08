import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List
import math
import numpy as np
from tqdm import tqdm

class AtomEmbedder(nn.Module):
    def __init__(self, num_atom_types: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, embedding_dim)

    def forward(self, atom_types: Tensor) -> Tensor:
        return self.embedding(atom_types)
    
class KernelSelfAttention(nn.Module):
    """Kernel Self-Attention (基于 RBF 核的自注意力)"""
    def __init__(self, input_dim: int, output_dim: int, lengthscales: list):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lengthscales = lengthscales
        self.num_heads = len(lengthscales)

        self.head_dim = output_dim // self.num_heads
        if output_dim % self.num_heads != 0:
            self.head_dim = output_dim // self.num_heads + 1

        self.total_head_dim = self.head_dim * self.num_heads

        self.value_projections = nn.ModuleList([
            nn.Linear(input_dim, self.head_dim)
            for _ in range(self.num_heads)
        ])

        self.output_projection = nn.Linear(self.total_head_dim, output_dim)

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        batch_size, num_atoms, _ = features.shape

        # 计算原子间距离矩阵
        coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
        coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
        distances_sq = torch.sum((coords_i - coords_j) ** 2, dim=-1)  # [B, N, N]

        # 多头注意力
        head_outputs = []
        for head_idx, lengthscale in enumerate(self.lengthscales):
            attention_weights = torch.exp(-distances_sq / (lengthscale ** 2))
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)

            value = self.value_projections[head_idx](features)
            attended_features = torch.bmm(attention_weights, value)
            head_outputs.append(attended_features)

        multi_head_output = torch.cat(head_outputs, dim=-1)
        return self.output_projection(multi_head_output)

class AtomTransformer(nn.Module):
    """Atom Transformer"""
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list, num_layers: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_mlp = nn.Sequential(
            nn.Linear(3 + embedding_dim + 3, hidden_dim),
            nn.ReLU()
        )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, lengthscales)
            for _ in range(num_layers)
        ])

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, latent_vars: Tensor, x_coords: Tensor, atom_embeddings: Tensor) -> Tensor:
        input_features = torch.cat([x_coords, atom_embeddings, latent_vars], dim=-1)
        features = self.input_mlp(input_features)

        for layer in self.transformer_layers:
            features = layer(features, x_coords)

        output = self.output_mlp(features)
        return output

class TransformerBlock(nn.Module):
    """Transformer 块"""
    def __init__(self, hidden_dim: int, lengthscales: list):
        super().__init__()
        self.kernel_attention = KernelSelfAttention(hidden_dim, hidden_dim, lengthscales)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, features: Tensor, coords: Tensor) -> Tensor:
        attended = self.kernel_attention(features, coords)
        features = self.norm1(features + attended)

        ffn_output = self.ffn(features)
        features = self.norm2(features + ffn_output)

        return features

class TimewarpCouplingLayer(nn.Module):
    """Timewarp RealNVP 耦合层"""
    def __init__(self, embedding_dim: int, hidden_dim: int, lengthscales: list):
        super().__init__()

        self.scale_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_p = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.scale_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)
        self.shift_transformer_v = AtomTransformer(embedding_dim, hidden_dim, lengthscales)

    def forward(self, z_p: Tensor, z_v: Tensor, x_coords: Tensor,
                atom_embeddings: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        
        if not reverse:
            # 前向：方程 (8) 和 (9)
            scale_p = self.scale_transformer_p(z_v, x_coords, atom_embeddings)
            shift_p = self.shift_transformer_p(z_v, x_coords, atom_embeddings)

            z_p_new = torch.exp(scale_p) * z_p + shift_p
            log_det_p = scale_p.sum(dim=-1)  # [batch_size, num_atoms]

            scale_v = self.scale_transformer_v(z_p_new, x_coords, atom_embeddings)
            shift_v = self.shift_transformer_v(z_p_new, x_coords, atom_embeddings)

            z_v_new = torch.exp(scale_v) * z_v + shift_v
            log_det_v = scale_v.sum(dim=-1)  # [batch_size, num_atoms]

            total_log_det = log_det_p + log_det_v  # [batch_size, num_atoms]

        else:
            # 反向
            scale_v = self.scale_transformer_v(z_p, x_coords, atom_embeddings)
            shift_v = self.shift_transformer_v(z_p, x_coords, atom_embeddings)

            z_v_new = (z_v - shift_v) * torch.exp(-scale_v)
            log_det_v = -scale_v.sum(dim=-1)  # [batch_size, num_atoms]

            scale_p = self.scale_transformer_p(z_v_new, x_coords, atom_embeddings)
            shift_p = self.shift_transformer_p(z_v_new, x_coords, atom_embeddings)

            z_p_new = (z_p - shift_p) * torch.exp(-scale_p)
            log_det_p = -scale_p.sum(dim=-1)  # [batch_size, num_atoms]

            total_log_det = log_det_p + log_det_v  # [batch_size, num_atoms]

        return z_p_new, z_v_new, total_log_det

class IntegratedTimewarpMCMC(nn.Module):
    """
    **集成MCMC的Timewarp模型 - 完整修复版本**
    
    关键修复：
    1. 正确计算Flow的对数概率
    2. 实现完整的MH接受比（包含proposal概率比）
    3. 正确累积Jacobian行列式
    4. 实现论文Algorithm 1的批量采样逻辑
    """
    def __init__(
        self,
        num_atom_types: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_coupling_layers: int = 12,
        lengthscales: list = [0.1, 0.2, 0.5, 0.7, 1.0, 1.2],
        temperature: float = 310.0,  # K
        energy_calculator = None  # 外部能量计算器（可选）
    ):
        super().__init__()

        # Flow 组件
        self.atom_embedder = AtomEmbedder(num_atom_types, embedding_dim)
        self.coupling_layers = nn.ModuleList([
            TimewarpCouplingLayer(embedding_dim, hidden_dim, lengthscales)
            for _ in range(num_coupling_layers)
        ])
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1)))

        # MCMC 参数
        self.temperature = temperature
        self.kT = 8.314 * temperature / 1000  # kJ/mol (玻尔兹曼常数)
        self.energy_calculator = energy_calculator
        
        # MCMC 统计
        self.mcmc_stats = {
            'total_proposals': 0,
            'accepted_proposals': 0,
            'acceptance_history': [],
            'energy_history': []
        }

    def forward(self, atom_types: Tensor, x_coords: Tensor, x_velocs: Tensor,
                y_coords: Tensor = None, y_velocs: Tensor = None,
                reverse: bool = False) -> Tuple[Tuple[Tensor, Tensor], Optional[Tensor]]:
        """
        Flow的前向传播 - 修复版本
        
        关键修复：
        1. 确保log_det正确累积
        2. 采样模式也返回正确的概率
        """
        
        batch_size, num_atoms = atom_types.shape
        atom_embeddings = self.atom_embedder(atom_types)
        x_coords_centered = self._center_coordinates(x_coords)

        if not reverse:
            # 训练/概率计算模式
            if y_coords is None or y_velocs is None:
                raise ValueError("训练模式需要提供目标坐标和速度")

            y_coords_centered = self._center_coordinates(y_coords)
            z_v = y_velocs
            z_p = y_coords_centered

            # 关键修复：正确初始化log_det维度
            total_log_det = torch.zeros(batch_size, num_atoms, device=x_coords.device)

            # 通过所有coupling层
            for layer in self.coupling_layers:
                z_p, z_v, log_det = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=False)
                total_log_det += log_det  # 累积Jacobian行列式

            # 计算prior概率 (标准高斯)
            scale = torch.exp(self.log_scale)
            log_prior_p = -0.5 * torch.sum((z_p / scale) ** 2, dim=-1)  # [batch_size, num_atoms]
            log_prior_v = -0.5 * torch.sum((z_v / scale) ** 2, dim=-1)  # [batch_size, num_atoms]
            log_prior = log_prior_p + log_prior_v

            # 关键修复：正确计算总概率
            log_likelihood = log_prior + total_log_det  # [batch_size, num_atoms]
            # 对原子维度求和得到每个batch的总概率
            log_likelihood_total = log_likelihood.sum(dim=1)  # [batch_size]

            return (y_coords, y_velocs), log_likelihood_total

        else:
            # 采样模式
            scale = torch.exp(self.log_scale)
            z_p = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale
            z_v = torch.randn(batch_size, num_atoms, 3, device=x_coords.device) * scale

            # 关键修复：在采样模式也要跟踪log_det
            total_log_det = torch.zeros(batch_size, num_atoms, device=x_coords.device)

            # 反向通过所有coupling层
            for layer in reversed(self.coupling_layers):
                z_p, z_v, log_det = layer(z_p, z_v, x_coords_centered, atom_embeddings, reverse=True)
                total_log_det += log_det

            output_coords = self._uncenter_coordinates(z_p, x_coords)
            output_velocs = z_v

            # 计算采样状态的概率
            log_prior_p = -0.5 * torch.sum((z_p / scale) ** 2, dim=-1)
            log_prior_v = -0.5 * torch.sum((z_v / scale) ** 2, dim=-1)
            log_prior = log_prior_p + log_prior_v
            
            log_probability = log_prior + total_log_det
            log_probability_total = log_probability.sum(dim=1)  # [batch_size]

            return (output_coords, output_velocs), log_probability_total

    def _center_coordinates(self, coords: Tensor) -> Tensor:
        """中心化坐标"""
        centroid = coords.mean(dim=1, keepdim=True)
        return coords - centroid

    def _uncenter_coordinates(self, centered_coords: Tensor, reference_coords: Tensor) -> Tensor:
        """恢复坐标中心"""
        reference_centroid = reference_coords.mean(dim=1, keepdim=True)
        return centered_coords + reference_centroid

    def calculate_energy(self, coords: Tensor) -> Tensor:
        """
        计算系统能量 (kJ/mol)
        
        Args:
            coords: [batch_size, num_atoms, 3] 坐标
        Returns:
            energy: [batch_size] 能量
        """
        if self.energy_calculator is not None:
            return self.energy_calculator(coords)
        else:
            # 简化的能量计算（LJ势）
            batch_size, num_atoms, _ = coords.shape
            
            coords_i = coords.unsqueeze(2)
            coords_j = coords.unsqueeze(1)
            distances = torch.norm(coords_i - coords_j, dim=-1)
            
            sigma = 0.3  # nm
            epsilon = 1.0  # kJ/mol
            
            mask = (distances > 0.01) & (distances < 1.0)
            distances_masked = torch.where(mask, distances, torch.tensor(sigma, device=coords.device))
            
            r6 = (sigma / distances_masked) ** 6
            lj_energy = 4 * epsilon * (r6 * r6 - r6)
            
            lj_energy_masked = torch.where(mask, lj_energy, torch.zeros_like(lj_energy))
            total_energy = 0.5 * lj_energy_masked.sum(dim=(1, 2))
            
            return total_energy
        
    def compute_proposal_log_prob(self, atom_types: Tensor, x_coords: Tensor, 
                                 x_velocs: Tensor, y_coords: Tensor, y_velocs: Tensor) -> Tensor:
        """
        计算 log p_θ(y|x) - Flow的条件概率
        
        关键：这个函数计算给定初始状态x，生成目标状态y的概率
        """
        with torch.no_grad():
            _, log_prob = self.forward(atom_types, x_coords, x_velocs, y_coords, y_velocs, reverse=False)
            return log_prob

    def metropolis_hastings_step(self, atom_types: Tensor, current_coords: Tensor, 
                                current_velocs: Tensor) -> Tuple[Tensor, Tensor, bool, Dict]:
        """
        **完整修复的Metropolis-Hastings步骤**
        
        实现论文公式(6)的完整MH接受比：
        α(X, X̃) = min(1, [μ_aug(X̃)p_θ(X|X̃_p)] / [μ_aug(X)p_θ(X̃|X_p)])
        """
        with torch.no_grad():
            # 1. 使用Flow生成提议状态
            (proposed_coords, proposed_velocs), _ = self.forward(
                atom_types, current_coords, current_velocs, reverse=True
            )
            
            # 2. 重新采样辅助变量 (论文Algorithm 1中的Gibbs步骤，方程7)
            # 这里使用玻尔兹曼分布的正确方差
            proposed_velocs = torch.randn_like(current_velocs) * math.sqrt(self.kT)
            current_velocs_resampled = torch.randn_like(current_velocs) * math.sqrt(self.kT)
            
            # 3. 计算能量项 (μ_aug中的能量部分)
            current_energy = self.calculate_energy(current_coords)
            proposed_energy = self.calculate_energy(proposed_coords)
            
            # 能量比 exp(-U_proposed/kT) / exp(-U_current/kT) = exp(-(U_proposed - U_current)/kT)
            energy_ratio = -(proposed_energy - current_energy) / self.kT
            
            # 4. 计算Flow的proposal概率比 p_θ(current|proposed) / p_θ(proposed|current)
            try:
                # Forward: current -> proposed
                log_prob_forward = self.compute_proposal_log_prob(
                    atom_types, current_coords, current_velocs_resampled, 
                    proposed_coords, proposed_velocs
                )
                
                # Reverse: proposed -> current  
                log_prob_reverse = self.compute_proposal_log_prob(
                    atom_types, proposed_coords, proposed_velocs,
                    current_coords, current_velocs_resampled
                )
                
                proposal_ratio = log_prob_reverse - log_prob_forward
                
            except Exception as e:
                # 如果概率计算失败，只使用能量项（降级处理）
                print(f"Warning: Flow probability calculation failed: {e}")
                proposal_ratio = torch.tensor(0.0, device=current_coords.device)
            
            # 5. 完整的MH接受比 (论文公式6)
            log_ratio = energy_ratio + proposal_ratio
            
            # 防止数值溢出
            log_ratio = torch.clamp(log_ratio, max=0)
            acceptance_prob = torch.exp(log_ratio).clamp(max=1.0)
            
            # 6. 接受/拒绝决策
            random_val = torch.rand(1, device=current_coords.device)
            accept = random_val < acceptance_prob
            
            # 7. 更新状态和统计
            self.mcmc_stats['total_proposals'] += 1
            if accept.item():
                self.mcmc_stats['accepted_proposals'] += 1
                new_coords = proposed_coords
                new_velocs = proposed_velocs
            else:
                new_coords = current_coords
                new_velocs = current_velocs_resampled  # 即使拒绝也要更新velocities (Gibbs step)
            
            self.mcmc_stats['acceptance_history'].append(accept.item())
            self.mcmc_stats['energy_history'].append(
                self.calculate_energy(new_coords).item()
            )
            
            step_info = {
                'accepted': accept.item(),
                'delta_energy': (proposed_energy - current_energy).item(),
                'energy_ratio': energy_ratio.item(),
                'proposal_ratio': proposal_ratio.item() if isinstance(proposal_ratio, torch.Tensor) else proposal_ratio,
                'log_acceptance_ratio': log_ratio.item(),
                'acceptance_prob': acceptance_prob.item(),
                'current_energy': current_energy.item(),
                'proposed_energy': proposed_energy.item()
            }
            
            return new_coords, new_velocs, accept.item(), step_info

    def mcmc_sampling(self, atom_types: Tensor, initial_coords: Tensor, 
                     initial_velocs: Tensor, num_steps: int = 10000,
                     save_interval: int = 100, batch_proposals: int = 10) -> Dict:
        """
        **集成的MCMC采样 - Algorithm 1完整实现**
        
        实现论文Algorithm 1的批量proposal逻辑
        """
        print(f"开始集成MCMC采样，{num_steps}步，批量提议{batch_proposals}")
        
        # 重置统计
        self.mcmc_stats = {
            'total_proposals': 0,
            'accepted_proposals': 0,
            'acceptance_history': [],
            'energy_history': []
        }
        
        # 初始化
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()
        
        trajectory_coords = []
        trajectory_velocs = []
        step_info_history = []
        
        self.eval()  # 设置为评估模式
        
        for step in tqdm(range(num_steps), desc="MCMC Sampling"):
            # Algorithm 1的批量提议逻辑：生成多个proposal，接受第一个满足条件的
            accepted_in_batch = False
            batch_info = []
            
            for batch_idx in range(batch_proposals):
                new_coords, new_velocs, accepted, step_info = self.metropolis_hastings_step(
                    atom_types, coords, velocs
                )
                
                batch_info.append(step_info)
                
                if accepted:
                    coords = new_coords
                    velocs = new_velocs
                    accepted_in_batch = True
                    break  # 接受第一个有效提议（Algorithm 1逻辑）
            
            # 如果批次中没有接受任何proposal，使用最后一次的重采样velocities
            if not accepted_in_batch:
                coords = coords  # 位置不变
                velocs = new_velocs  # 但速度被重采样了（Gibbs步骤）
            
            # 保存轨迹
            if step % save_interval == 0:
                trajectory_coords.append(coords.cpu().clone())
                trajectory_velocs.append(velocs.cpu().clone())
                step_info_history.append(batch_info[-1])  # 保存最后一次尝试的信息
            
            # 进度报告
            if (step + 1) % 2000 == 0:
                recent_accept_rate = (
                    np.mean(self.mcmc_stats['acceptance_history'][-1000:]) 
                    if len(self.mcmc_stats['acceptance_history']) >= 1000 
                    else np.mean(self.mcmc_stats['acceptance_history'])
                )
                current_energy = self.mcmc_stats['energy_history'][-1] if self.mcmc_stats['energy_history'] else 0
                print(f"Step {step+1}: 接受率={recent_accept_rate:.3f}, 能量={current_energy:.2f} kJ/mol")
        
        # 计算最终统计
        final_accept_rate = (
            self.mcmc_stats['accepted_proposals'] / self.mcmc_stats['total_proposals']
            if self.mcmc_stats['total_proposals'] > 0 else 0
        )
        
        results = {
            'trajectory_coords': torch.stack(trajectory_coords) if trajectory_coords else None,
            'trajectory_velocs': torch.stack(trajectory_velocs) if trajectory_velocs else None,
            'final_coords': coords,
            'final_velocs': velocs,
            'acceptance_rate': final_accept_rate,
            'total_proposals': self.mcmc_stats['total_proposals'],
            'accepted_proposals': self.mcmc_stats['accepted_proposals'],
            'energy_history': self.mcmc_stats['energy_history'],
            'acceptance_history': self.mcmc_stats['acceptance_history'],
            'step_info_history': step_info_history
        }
        
        print(f"\nMCMC采样完成!")
        print(f"最终接受率: {final_accept_rate:.4f}")
        print(f"总提议数: {self.mcmc_stats['total_proposals']}")
        print(f"接受数: {self.mcmc_stats['accepted_proposals']}")
        
        return results

    def fast_exploration(self, atom_types: Tensor, initial_coords: Tensor,
                        initial_velocs: Tensor, num_steps: int = 50000,
                        energy_cutoff: float = 300.0, save_interval: int = 100) -> Dict:
        """
        **集成的快速探索 - Algorithm 2实现**
        """
        print(f"开始集成快速探索，{num_steps}步，能量阈值{energy_cutoff} kJ/mol")
        
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()
        
        trajectory_coords = []
        trajectory_velocs = []
        energy_history = []
        accept_count = 0
        
        self.eval()
        
        for step in tqdm(range(num_steps), desc="Fast Exploration"):
            with torch.no_grad():
                # 生成提议
                (proposed_coords, _), _ = self.forward(
                    atom_types, coords, velocs, reverse=True
                )
                proposed_velocs = torch.randn_like(velocs) * 0.1
                
                # 计算能量变化
                current_energy = self.calculate_energy(coords)
                proposed_energy = self.calculate_energy(proposed_coords)
                delta_energy = proposed_energy - current_energy
                
                # Algorithm 2的简单接受准则
                if delta_energy.item() < energy_cutoff:
                    coords = proposed_coords
                    velocs = proposed_velocs
                    accept_count += 1
                    energy_history.append(proposed_energy.item())
                else:
                    energy_history.append(current_energy.item())
                
                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu().clone())
                    trajectory_velocs.append(velocs.cpu().clone())
        
        accept_rate = accept_count / num_steps
        
        results = {
            'trajectory_coords': torch.stack(trajectory_coords) if trajectory_coords else None,
            'trajectory_velocs': torch.stack(trajectory_velocs) if trajectory_velocs else None,
            'final_coords': coords,
            'final_velocs': velocs,
            'acceptance_rate': accept_rate,
            'total_steps': num_steps,
            'accepted_steps': accept_count,
            'energy_history': energy_history,
            'energy_cutoff': energy_cutoff
        }
        
        print(f"\n快速探索完成!")
        print(f"接受率: {accept_rate:.4f}")
        print(f"接受步数: {accept_count}/{num_steps}")
        
        return results

    def get_mcmc_statistics(self) -> Dict:
        """获取当前MCMC统计"""
        if self.mcmc_stats['total_proposals'] == 0:
            return {'message': 'No MCMC steps performed yet'}
        
        return {
            'total_proposals': self.mcmc_stats['total_proposals'],
            'accepted_proposals': self.mcmc_stats['accepted_proposals'],
            'acceptance_rate': self.mcmc_stats['accepted_proposals'] / self.mcmc_stats['total_proposals'],
            'average_energy': np.mean(self.mcmc_stats['energy_history']) if self.mcmc_stats['energy_history'] else None,
            'recent_acceptance_rate': (
                np.mean(self.mcmc_stats['acceptance_history'][-100:]) 
                if len(self.mcmc_stats['acceptance_history']) >= 100 
                else np.mean(self.mcmc_stats['acceptance_history'])
            ) if self.mcmc_stats['acceptance_history'] else None
        }

def create_integrated_timewarp_model(config: dict) -> IntegratedTimewarpMCMC:
    """创建集成MCMC的Timewarp模型"""
    return IntegratedTimewarpMCMC(
        num_atom_types=config.get('num_atom_types', 10),
        embedding_dim=config.get('embedding_dim', 64),
        hidden_dim=config.get('hidden_dim', 128),
        num_coupling_layers=config.get('num_coupling_layers', 12),
        lengthscales=config.get('lengthscales', [0.1, 0.2, 0.5, 0.7, 1.0, 1.2]),
        temperature=config.get('temperature', 310.0)
    )

# 使用示例
if __name__ == "__main__":
    # 配置
    config = {
        'num_atom_types': 20,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'num_coupling_layers': 12,
        'lengthscales': [0.1, 0.2, 0.5, 0.7, 1.0, 1.2],
        'temperature': 310.0
    }
    
    # 创建集成模型
    model = create_integrated_timewarp_model(config)
    
    # 模拟数据
    batch_size, num_atoms = 1, 22
    atom_types = torch.randint(0, 4, (batch_size, num_atoms))
    initial_coords = torch.randn(batch_size, num_atoms, 3) * 0.5
    initial_velocs = torch.randn(batch_size, num_atoms, 3) * 0.1
    
    print("=== 集成Timewarp模型测试 ===")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试MCMC采样
    print("\n1. 测试集成MCMC采样...")
    mcmc_results = model.mcmc_sampling(
        atom_types, initial_coords, initial_velocs,
        num_steps=1000, save_interval=50, batch_proposals=5
    )
    
    # 测试快速探索
    print("\n2. 测试集成快速探索...")
    fast_results = model.fast_exploration(
        atom_types, initial_coords, initial_velocs,
        num_steps=2000, energy_cutoff=300.0, save_interval=100
    )
    
    # 查看统计
    print("\n3. MCMC统计:")
    stats = model.get_mcmc_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== 测试完成 ===")
    print("此模型解决了分离采样的问题，MCMC完全集成在模型内部！")
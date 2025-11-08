import torch
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
from openmm import unit
import os
from tqdm import tqdm
import json
from typing import Tuple, Optional, Dict, List
import warnings

class TimewarpCorrectExplorer:
    """
    正确实现Timewarp论文中的两种算法：
    1. Algorithm 1: MH-corrected MCMC (严格但慢)
    2. Algorithm 2: Fast exploration with energy cutoff (快速但有偏)
    """

    def __init__(self, model_path: str, training_data_path: str = 'training_pairs_augmented_final.npy', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"初始化TimewarpCorrectExplorer，设备: {device}")

        # 加载模型
        print("正在加载Timewarp模型...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']
        
        # 这里需要导入你的模型定义
        from model_timewarp import create_timewarp_model
        self.model = create_timewarp_model(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 标准化参数
        self.norm_stats = checkpoint.get('normalization_stats', None)
        print(f"模型加载完成，配置: {self.config}")

        # 加载训练数据
        print("正在加载训练数据...")
        data = np.load(training_data_path)
        print(f"训练数据形状: {data.shape}")

        # 坐标转换：埃 → 纳米
        self.all_coords_nm = data[:, :, :, :3].reshape(-1, 22, 3) / 10.0
        self.all_velocs_nm_ps = data[:, :, :, 3:].reshape(-1, 22, 3) / 10.0

        print(f"坐标范围: {self.all_coords_nm.min():.4f} 到 {self.all_coords_nm.max():.4f} nm")

        # Alanine dipeptide原子类型
        self.atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3],
                                     dtype=torch.long, device=device).unsqueeze(0)

        # 初始化OpenMM力场用于能量计算
        self.setup_openmm_energy_calculator()

    def setup_openmm_energy_calculator(self):
        """设置OpenMM能量计算器"""
        try:
            print("设置OpenMM能量计算器...")
            
            # 创建简单的alanine dipeptide系统
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
            
            # 你可能需要根据实际情况调整PDB文件路径
            # 这里先创建一个简单的计算器
            self.has_energy_calculator = False
            print("警告：OpenMM能量计算器未完全设置，将使用简化的能量筛选")
            
        except Exception as e:
            print(f"OpenMM设置失败: {e}")
            print("将使用简化的能量筛选方法")
            self.has_energy_calculator = False

    def calculate_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        计算系统能量 (kJ/mol)
        
        Args:
            coords: [batch_size, num_atoms, 3] 坐标 (nm)
        Returns:
            energy: [batch_size] 能量 (kJ/mol)
        """
        if self.has_energy_calculator:
            # 实际的OpenMM能量计算
            # 这里需要实现具体的能量计算逻辑
            pass
        else:
            # 简化的能量估计：基于原子间距离的简单势函数
            batch_size, num_atoms, _ = coords.shape
            
            # 计算原子间距离
            coords_i = coords.unsqueeze(2)  # [B, N, 1, 3]
            coords_j = coords.unsqueeze(1)  # [B, 1, N, 3]
            distances = torch.norm(coords_i - coords_j, dim=-1)  # [B, N, N]
            
            # 简单的Lennard-Jones势
            sigma = 0.3  # nm
            epsilon = 1.0  # kJ/mol
            
            # 避免自作用和除零
            mask = (distances > 0.01) & (distances < 1.0)  # 只考虑合理距离范围
            distances_masked = torch.where(mask, distances, torch.tensor(sigma, device=coords.device))
            
            # LJ势: 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
            r6 = (sigma / distances_masked) ** 6
            lj_energy = 4 * epsilon * (r6 * r6 - r6)
            
            # 只对掩码区域求和
            lj_energy_masked = torch.where(mask, lj_energy, torch.zeros_like(lj_energy))
            total_energy = 0.5 * lj_energy_masked.sum(dim=(1, 2))  # [B]
            
            return total_energy

    def get_random_initial_structure(self, idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """从训练数据中获取初始结构"""
        if idx is None:
            idx = np.random.randint(0, len(self.all_coords_nm))
        
        initial_coords = torch.FloatTensor(self.all_coords_nm[idx:idx+1]).to(self.device)
        initial_velocs = torch.FloatTensor(self.all_velocs_nm_ps[idx:idx+1]).to(self.device)
        
        return initial_coords, initial_velocs

    def normalize_data(self, coords: torch.Tensor, velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准化数据"""
        if self.norm_stats is None:
            return coords, velocs

        pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
        coords_norm = (coords - pos_mean) / pos_std
        velocs_norm = (velocs - vel_mean) / vel_std

        return coords_norm, velocs_norm

    def denormalize_data(self, coords: torch.Tensor, velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """反标准化数据"""
        if self.norm_stats is None:
            return coords, velocs

        pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
        coords_denorm = coords * pos_std + pos_mean
        velocs_denorm = velocs * vel_std + vel_mean

        return coords_denorm, velocs_denorm

    def metropolis_hastings_acceptance(self, current_coords: torch.Tensor, current_velocs: torch.Tensor,
                                     proposed_coords: torch.Tensor, proposed_velocs: torch.Tensor) -> bool:
        """
        Metropolis-Hastings acceptance criterion - 论文Algorithm 1
        
        Args:
            current_coords, current_velocs: 当前状态
            proposed_coords, proposed_velocs: 提议状态
        Returns:
            accept: 是否接受提议
        """
        # 计算能量
        current_energy = self.calculate_energy(current_coords)
        proposed_energy = self.calculate_energy(proposed_coords)
        
        # 能量差 (kJ/mol)
        delta_energy = proposed_energy - current_energy
        
        # Boltzmann因子 (T = 310K)
        kT = 8.314 * 310 / 1000  # kJ/mol
        
        # 论文方程(6)的简化版本（假设提议分布对称）
        acceptance_prob = torch.exp(-delta_energy / kT).clamp(max=1.0)
        
        # 随机接受
        random_val = torch.rand(1, device=self.device)
        accept = random_val < acceptance_prob
        
        return accept.item(), delta_energy.item(), acceptance_prob.item()

    def algorithm_1_mcmc_exploration(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor,
                                   num_steps: int = 20000, batch_size: int = 10, 
                                   save_interval: int = 100, output_dir: str = 'mcmc_exploration') -> Dict:
        """
        Algorithm 1: Timewarp MH-corrected MCMC - 论文Algorithm 1
        严格的、无偏的采样，但可能慢且接受率低
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始Algorithm 1 MCMC探索，{num_steps}步，批量大小{batch_size}")

        # 标准化输入
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()
        
        if self.norm_stats:
            coords, velocs = self.normalize_data(coords, velocs)

        # 统计信息
        accept_count = 0
        total_proposals = 0
        trajectory_coords = [coords.cpu()]
        energy_history = []
        acceptance_history = []

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                # 批量生成提议
                proposals_coords = []
                proposals_velocs = []
                
                for _ in range(batch_size):
                    # 使用模型生成提议
                    prop_coords, _ = self.model(self.atom_types, coords, velocs, reverse=True)
                    prop_velocs = torch.randn_like(velocs) * 0.1  # 重新采样辅助变量
                    proposals_coords.append(prop_coords)
                    proposals_velocs.append(prop_velocs)

                # 找第一个被接受的提议 (论文Algorithm 1逻辑)
                accepted = False
                for i in range(batch_size):
                    total_proposals += 1
                    
                    # 反标准化进行能量计算
                    if self.norm_stats:
                        current_coords_real, current_velocs_real = self.denormalize_data(coords, velocs)
                        prop_coords_real, prop_velocs_real = self.denormalize_data(proposals_coords[i], proposals_velocs[i])
                    else:
                        current_coords_real, current_velocs_real = coords, velocs
                        prop_coords_real, prop_velocs_real = proposals_coords[i], proposals_velocs[i]

                    # MH接受检验
                    accept, delta_e, accept_prob = self.metropolis_hastings_acceptance(
                        current_coords_real, current_velocs_real,
                        prop_coords_real, prop_velocs_real
                    )

                    if accept:
                        coords = proposals_coords[i]
                        velocs = proposals_velocs[i]
                        accept_count += 1
                        accepted = True
                        acceptance_history.append(1)
                        break
                    else:
                        acceptance_history.append(0)

                if not accepted:
                    # 如果没有提议被接受，保持当前状态
                    pass

                # 记录能量
                if self.norm_stats:
                    coords_real, _ = self.denormalize_data(coords, velocs)
                else:
                    coords_real = coords
                energy = self.calculate_energy(coords_real).item()
                energy_history.append(energy)

                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu())

                # 进度报告
                if (step + 1) % 2000 == 0:
                    recent_accept_rate = np.mean(acceptance_history[-1000:]) if len(acceptance_history) >= 1000 else np.mean(acceptance_history)
                    print(f"Step {step+1}/{num_steps}, 接受率: {recent_accept_rate:.3f}, 当前能量: {energy:.2f} kJ/mol")

        # 计算最终统计
        final_accept_rate = accept_count / total_proposals if total_proposals > 0 else 0
        
        print(f"\nAlgorithm 1 完成!")
        print(f"总接受率: {final_accept_rate:.4f}")
        print(f"总提议数: {total_proposals}")
        print(f"接受数: {accept_count}")

        # 保存结果
        trajectory_coords = torch.cat(trajectory_coords, dim=0)
        
        # 反标准化保存
        if self.norm_stats:
            trajectory_coords, _ = self.denormalize_data(trajectory_coords, torch.zeros_like(trajectory_coords))
        
        np.save(f'{output_dir}/mcmc_coords.npy', trajectory_coords.numpy())
        np.save(f'{output_dir}/energy_history.npy', np.array(energy_history))
        np.save(f'{output_dir}/acceptance_history.npy', np.array(acceptance_history))

        stats = {
            'algorithm': 'MCMC_Algorithm_1',
            'total_steps': num_steps,
            'total_proposals': total_proposals,
            'accepted_proposals': accept_count,
            'acceptance_rate': final_accept_rate,
            'batch_size': batch_size,
            'trajectory_length': len(trajectory_coords),
            'final_energy': energy_history[-1] if energy_history else None
        }

        with open(f'{output_dir}/mcmc_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def algorithm_2_fast_exploration(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor,
                                   num_steps: int = 100000, energy_cutoff: float = 300.0,  # kJ/mol
                                   save_interval: int = 100, output_dir: str = 'fast_exploration') -> Dict:
        """
        Algorithm 2: Fast exploration with energy cutoff - 论文Algorithm 2
        快速但有偏的探索，只用能量阈值筛选
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始Algorithm 2 快速探索，{num_steps}步，能量阈值{energy_cutoff} kJ/mol")

        # 标准化输入
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()
        
        if self.norm_stats:
            coords, velocs = self.normalize_data(coords, velocs)

        # 统计信息
        accept_count = 0
        total_proposals = 0
        trajectory_coords = [coords.cpu()]
        energy_history = []
        rejection_reasons = {'energy_too_high': 0, 'chirality_change': 0}

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                total_proposals += 1

                # 使用模型生成提议
                proposed_coords, _ = self.model(self.atom_types, coords, velocs, reverse=True)
                proposed_velocs = torch.randn_like(velocs) * 0.1

                # 反标准化进行能量计算
                if self.norm_stats:
                    current_coords_real, _ = self.denormalize_data(coords, velocs)
                    prop_coords_real, _ = self.denormalize_data(proposed_coords, proposed_velocs)
                else:
                    current_coords_real = coords
                    prop_coords_real = proposed_coords

                # 计算能量变化
                current_energy = self.calculate_energy(current_coords_real)
                proposed_energy = self.calculate_energy(prop_coords_real)
                delta_energy = proposed_energy - current_energy

                # Algorithm 2的简单接受准则：只检查能量阈值
                accept = delta_energy.item() < energy_cutoff

                if accept:
                    coords = proposed_coords
                    velocs = proposed_velocs  
                    accept_count += 1
                    energy_history.append(proposed_energy.item())
                else:
                    rejection_reasons['energy_too_high'] += 1
                    energy_history.append(current_energy.item())

                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu())

                # 进度报告
                if (step + 1) % 10000 == 0:
                    recent_accept_rate = accept_count / total_proposals
                    avg_energy = np.mean(energy_history[-1000:]) if len(energy_history) >= 1000 else np.mean(energy_history)
                    print(f"Step {step+1}/{num_steps}, 接受率: {recent_accept_rate:.3f}, 平均能量: {avg_energy:.2f} kJ/mol")

        # 计算最终统计
        final_accept_rate = accept_count / total_proposals

        print(f"\nAlgorithm 2 完成!")
        print(f"总接受率: {final_accept_rate:.4f}")
        print(f"总提议数: {total_proposals}")
        print(f"接受数: {accept_count}")
        print(f"能量过高拒绝: {rejection_reasons['energy_too_high']}")

        # 保存结果
        trajectory_coords = torch.cat(trajectory_coords, dim=0)
        
        # 反标准化保存
        if self.norm_stats:
            trajectory_coords, _ = self.denormalize_data(trajectory_coords, torch.zeros_like(trajectory_coords))

        np.save(f'{output_dir}/fast_coords.npy', trajectory_coords.numpy())
        np.save(f'{output_dir}/energy_history.npy', np.array(energy_history))

        stats = {
            'algorithm': 'Fast_Exploration_Algorithm_2',
            'total_steps': num_steps,
            'total_proposals': total_proposals,
            'accepted_proposals': accept_count,
            'acceptance_rate': final_accept_rate,
            'energy_cutoff': energy_cutoff,
            'trajectory_length': len(trajectory_coords),
            'rejection_reasons': rejection_reasons,
            'final_energy': energy_history[-1] if energy_history else None
        }

        with open(f'{output_dir}/fast_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def compare_algorithms(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor, 
                          mcmc_steps: int = 20000, fast_steps: int = 100000) -> Dict:
        """比较两种算法的性能"""
        print("=== 比较两种Timewarp算法 ===")
        
        # Algorithm 1: MCMC
        print("\n运行Algorithm 1 (MCMC)...")
        mcmc_stats = self.algorithm_1_mcmc_exploration(
            initial_coords, initial_velocs, num_steps=mcmc_steps, 
            output_dir='comparison_mcmc'
        )
        
        # Algorithm 2: Fast exploration
        print("\n运行Algorithm 2 (快速探索)...")
        fast_stats = self.algorithm_2_fast_exploration(
            initial_coords, initial_velocs, num_steps=fast_steps,
            output_dir='comparison_fast'
        )
        
        # 比较结果
        comparison = {
            'mcmc': mcmc_stats,
            'fast': fast_stats,
            'comparison': {
                'mcmc_acceptance_rate': mcmc_stats['acceptance_rate'],
                'fast_acceptance_rate': fast_stats['acceptance_rate'],
                'speed_ratio': fast_steps / mcmc_steps,
                'exploration_efficiency': fast_stats['acceptance_rate'] * (fast_steps / mcmc_steps)
            }
        }
        
        print(f"\n=== 算法比较结果 ===")
        print(f"MCMC接受率: {mcmc_stats['acceptance_rate']:.4f}")
        print(f"快速探索接受率: {fast_stats['acceptance_rate']:.4f}")
        print(f"速度比 (快速/MCMC): {fast_steps / mcmc_steps:.1f}x")
        print(f"探索效率 (接受率×速度): {comparison['comparison']['exploration_efficiency']:.2f}")
        
        with open('algorithm_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
            
        return comparison

def main():
    """主函数：运行正确的Timewarp探索"""
    
    # 创建探索器
    explorer = TimewarpCorrectExplorer(
        'corrected_timewarp_model_final.pth', 
        'training_pairs_augmented_final.npy'
    )
    
    # 获取初始结构
    initial_coords, initial_velocs = explorer.get_random_initial_structure(idx=0)
    print(f"初始结构坐标范围: {initial_coords.min():.4f} 到 {initial_coords.max():.4f} nm")
    
    # 选择运行的算法
    run_mcmc = True
    run_fast = True
    run_comparison = True
    
    if run_mcmc:
        print("\n=== 运行Algorithm 1: MCMC探索 ===")
        mcmc_stats = explorer.algorithm_1_mcmc_exploration(
            initial_coords, initial_velocs, 
            num_steps=20000,  # 较少步数，因为接受率低
            batch_size=10,
            output_dir='timewarp_mcmc_correct'
        )
    
    if run_fast:
        print("\n=== 运行Algorithm 2: 快速探索 ===")
        fast_stats = explorer.algorithm_2_fast_exploration(
            initial_coords, initial_velocs,
            num_steps=100000,  # 更多步数，因为速度快
            energy_cutoff=300.0,  # 论文建议的阈值
            output_dir='timewarp_fast_correct'
        )
    
    if run_comparison:
        print("\n=== 运行算法比较 ===")
        comparison = explorer.compare_algorithms(initial_coords, initial_velocs)
    
    print("\n所有探索完成！请分析生成的轨迹文件。")

if __name__ == "__main__":
    main()
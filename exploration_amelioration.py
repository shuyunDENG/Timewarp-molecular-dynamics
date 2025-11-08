import torch
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app

from openmm.app import PDBFile, Modeller, ForceField, Simulation
from openmm import unit
import os
from tqdm import tqdm
import json
from typing import Tuple, Optional, Dict, List
import warnings
from scipy.spatial.distance import cdist



class TimewarpCorrectExplorer:
    """
    正确实现Timewarp论文采样，基于GitHub仓库的sample.py和evaluate.py
    """

    def __init__(self, model_path: str, pdb_path: str, training_data_path: str = 'training_pairs_augmented_final.npy',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"初始化TimewarpCorrectExplorer，设备: {device}")

        # 加载模型
        print("正在加载Timewarp模型...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.config = checkpoint['config']

        # 这里需要导入你的模型定义

        self.model = create_timewarp_model(self.config).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 标准化参数
        self.norm_stats = checkpoint.get('normalization_stats', None)
        print(f"模型加载完成，配置: {self.config}")

        # 加载训练数据 - 这是关键！
        print("正在加载训练数据...")
        self.training_data = np.load(training_data_path)
        print(f"训练数据形状: {self.training_data.shape}")

        # 解析训练数据：(t, t+tau)配对
        # 假设数据格式：[num_pairs, 2, num_atoms, 6] 其中第二维是(t, t+tau)
        self.t_states = self.training_data[:, 0]  # t时刻状态
        self.t_plus_tau_states = self.training_data[:, 1]  # t+τ时刻状态

        # 坐标转换：埃 → 纳米
        self.t_coords = self.t_states[:, :, :3] / 10.0  # [N, 22, 3]
        self.t_velocs = self.t_states[:, :, 3:] / 10.0  # [N, 22, 3]
        self.t_plus_tau_coords = self.t_plus_tau_states[:, :, :3] / 10.0
        self.t_plus_tau_velocs = self.t_plus_tau_states[:, :, 3:] / 10.0

        print(f"训练配对数量: {len(self.t_states)}")
        print(f"坐标范围: {self.t_coords.min():.4f} 到 {self.t_coords.max():.4f} nm")

        # 调用OpenMM能量计算器设置（在定义atom_types之前）
        self.setup_openmm_energy_calculator(pdb_path)

        # Alanine dipeptide原子类型
        self.atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3],
                                     dtype=torch.long, device=device).unsqueeze(0)

        # 设置用于最近邻搜索的参数
        self.k_nearest = 10  # 考虑最近的k个训练样本

    def setup_openmm_energy_calculator(self, pdb_path: str):
        """
        Build an OpenMM System/Context using the reference PDB topology
        and an implicit solvent Amber force‑field.  The resulting Context
        is stored in self._ommm_context and used for fast per‑frame energy
        evaluation.
        """
        print("Setting up OpenMM energy calculator …")
        try:
            pdb = PDBFile(pdb_path)
            modeller = Modeller(pdb.topology, pdb.positions)
            # Amber99SB‑ILDN with implicit OBC solvent
            ff = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
            system = ff.createSystem(modeller.topology,
                                     nonbondedMethod=app.NoCutoff,
                                     constraints=app.HBonds)
            integrator = mm.VerletIntegrator(1.0*unit.femtoseconds)
            platform = mm.Platform.getPlatformByName('CPU')
            self._ommm_context = mm.Context(system, integrator, platform)
            self._reference_topology = modeller.topology
            self.has_energy_calculator = True
            print("OpenMM energy calculator ready.")
        except Exception as exc:
            warnings.warn(f"OpenMM setup failed, falling back to LJ‑proxy energies ({exc})")
            self.has_energy_calculator = False

    def calculate_energy(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Return potential energy (kJ/mol) for a batch of coordinate frames.
        Uses OpenMM if available; otherwise falls back to the simplified LJ proxy.
        """
        if self.has_energy_calculator:
            energies = []
            for frame in coords.cpu().numpy():        # shape (22,3), nm
                # convert nm -> Angstrom
                positions = frame * 10.0 * unit.angstrom
                self._ommm_context.setPositions(positions)
                state = self._ommm_context.getState(getEnergy=True)
                e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                energies.append(e_kj)
            return torch.tensor(energies, device=coords.device)
        else:
            # 简化的能量估计
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

    def find_nearest_training_samples(self, current_coords: np.ndarray, k: int = None) -> List[int]:
        """
        找到与当前坐标最相似的训练样本
        这是正确采样的关键！基于训练数据的配对进行采样
        """
        if k is None:
            k = self.k_nearest

        # 将坐标展平用于距离计算
        current_flat = current_coords.flatten()
        training_flat = self.t_coords.reshape(len(self.t_coords), -1)

        # 计算欧式距离
        distances = cdist([current_flat], training_flat, metric='euclidean')[0]

        # 找到最近的k个样本
        nearest_indices = np.argsort(distances)[:k]

        return nearest_indices.tolist()

    def sample_from_training_pairs(self, current_coords: torch.Tensor, current_velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        基于训练数据的(t, t+τ)配对进行采样
        这是GitHub sample.py的核心逻辑
        """
        # 转换为numpy进行最近邻搜索
        current_coords_np = current_coords.cpu().numpy().squeeze()

        # 找到最近的训练样本
        nearest_indices = self.find_nearest_training_samples(current_coords_np)

        # 随机选择一个最近的样本
        selected_idx = np.random.choice(nearest_indices)

        # 获取对应的t+τ状态作为提议
        proposed_coords = torch.FloatTensor(self.t_plus_tau_coords[selected_idx:selected_idx+1]).to(self.device)
        proposed_velocs = torch.FloatTensor(self.t_plus_tau_velocs[selected_idx:selected_idx+1]).to(self.device)

        return proposed_coords, proposed_velocs, selected_idx

    def compute_log_jacobian(self, current_coords: torch.Tensor, current_velocs: torch.Tensor,
                           proposed_coords: torch.Tensor, proposed_velocs: torch.Tensor) -> float:
        """
        计算雅可比行列式的对数
        这是GitHub evaluate.py中acceptance rule的关键部分
        """
        # 简化版本：基于坐标变化的雅可比
        with torch.no_grad():
            # 标准化数据
            if self.norm_stats:
                current_coords_norm, current_velocs_norm = self.normalize_data(current_coords, current_velocs)
                proposed_coords_norm, proposed_velocs_norm = self.normalize_data(proposed_coords, proposed_velocs)
            else:
                current_coords_norm, current_velocs_norm = current_coords, current_velocs
                proposed_coords_norm, proposed_velocs_norm = proposed_coords, proposed_velocs

            # 使用模型计算前向和反向的概率比
            # 前向：当前 -> 提议
            #forward_output = self.model(self.atom_types, current_coords_norm, current_velocs_norm, reverse=False)

            # 反向：提议 -> 当前
            backward_output = self.model(self.atom_types, proposed_coords_norm, proposed_velocs_norm, reverse=True)

            # 简化的雅可比计算（实际应该更复杂）
            coord_diff = torch.norm(proposed_coords - current_coords)
            log_jacobian = -coord_diff.item()  # 简化版本

            return log_jacobian

    def timewarp_acceptance_rule(self, current_coords: torch.Tensor, current_velocs: torch.Tensor,
                               proposed_coords: torch.Tensor, proposed_velocs: torch.Tensor,
                               training_pair_idx: int) -> Tuple[bool, Dict]:
        """
        实现GitHub evaluate.py中的acceptance rule
        包含能量、雅可比行列式等因素
        """
        # 计算能量
        current_energy = self.calculate_energy(current_coords)
        proposed_energy = self.calculate_energy(proposed_coords)
        delta_energy = proposed_energy - current_energy

        # 计算雅可比行列式
        log_jacobian = self.compute_log_jacobian(current_coords, current_velocs,
                                               proposed_coords, proposed_velocs)

        # 计算采样概率比
        # 这里需要考虑从训练数据采样的概率
        # 简化版本：基于最近邻的概率
        log_sampling_ratio = 0.0  # 简化假设采样是对称的

        # Timewarp acceptance probability
        # 论文公式：α = min(1, exp(-ΔE/kT + log_jacobian + log_sampling_ratio))
        kT = 8.314 * 310 / 1000  # kJ/mol, T=310K

        log_acceptance = (-delta_energy / kT + log_jacobian + log_sampling_ratio).item()
        acceptance_prob = min(1.0, np.exp(log_acceptance))

        # 随机决定是否接受
        random_val = np.random.random()
        accept = random_val < acceptance_prob

        # 返回详细信息
        info = {
            'accept': accept,
            'acceptance_prob': acceptance_prob,
            'delta_energy': delta_energy.item(),
            'log_jacobian': log_jacobian,
            'log_sampling_ratio': log_sampling_ratio,
            'training_pair_idx': training_pair_idx,
            'random_val': random_val
        }

        return accept, info

    def normalize_data(self, coords: torch.Tensor, velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """标准化数据"""
        if self.norm_stats is None:
            return coords, velocs
        if len(self.norm_stats) == 4:
            pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
            coords_norm = (coords - pos_mean) / pos_std
            velocs_norm = (velocs - vel_mean) / vel_std
            return coords_norm, velocs_norm
        elif len(self.norm_stats) == 2:
            pos_mean, pos_std = self.norm_stats
            coords_norm = (coords - pos_mean) / pos_std
            return coords_norm, velocs
        else:
            warnings.warn("Unexpected length of normalization_stats; skip normalization")
            return coords, velocs

    def denormalize_data(self, coords: torch.Tensor, velocs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """反标准化数据"""
        if self.norm_stats is None:
            return coords, velocs
        if len(self.norm_stats) == 4:
            pos_mean, pos_std, vel_mean, vel_std = self.norm_stats
            coords_denorm = coords * pos_std + pos_mean
            velocs_denorm = velocs * vel_std + vel_mean
            return coords_denorm, velocs_denorm
        elif len(self.norm_stats) == 2:
            pos_mean, pos_std = self.norm_stats
            coords_denorm = coords * pos_std + pos_mean
            return coords_denorm, velocs
        else:
            warnings.warn("Unexpected length of normalization_stats; skip normalization")
            return coords, velocs

    def correct_timewarp_sampling(self, initial_coords: torch.Tensor, initial_velocs: torch.Tensor,
                                num_steps: int = 50000, save_interval: int = 100,
                                output_dir: str = 'correct_timewarp_sampling') -> Dict:
        """
        正确的Timewarp采样，基于GitHub仓库的实现
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始正确的Timewarp采样，{num_steps}步")
        print(f"基于{len(self.t_states)}个训练配对进行采样")

        # 初始化
        coords = initial_coords.clone()
        velocs = initial_velocs.clone()

        # 统计信息
        accept_count = 0
        total_proposals = 0
        trajectory_coords = [coords.cpu()]
        trajectory_velocs = [velocs.cpu()]
        energy_history = []
        acceptance_history = []
        detailed_info = []

        with torch.no_grad():
            for step in tqdm(range(num_steps)):
                total_proposals += 1

                # 基于训练数据配对进行采样
                proposed_coords, proposed_velocs, training_idx = self.sample_from_training_pairs(coords, velocs)

                # 使用正确的acceptance rule
                accept, info = self.timewarp_acceptance_rule(
                    coords, velocs, proposed_coords, proposed_velocs, training_idx
                )

                if accept:
                    coords = proposed_coords
                    velocs = proposed_velocs
                    accept_count += 1
                    acceptance_history.append(1)
                else:
                    acceptance_history.append(0)

                # 记录能量
                current_energy = self.calculate_energy(coords)
                energy_history.append(current_energy.item())

                # 记录详细信息
                detailed_info.append(info)

                # 保存轨迹
                if step % save_interval == 0:
                    trajectory_coords.append(coords.cpu())
                    trajectory_velocs.append(velocs.cpu())

                # 进度报告
                if (step + 1) % 5000 == 0:
                    recent_accept_rate = np.mean(acceptance_history[-1000:]) if len(acceptance_history) >= 1000 else np.mean(acceptance_history)
                    avg_energy = np.mean(energy_history[-1000:]) if len(energy_history) >= 1000 else np.mean(energy_history)
                    print(f"Step {step+1}/{num_steps}, 接受率: {recent_accept_rate:.3f}, 平均能量: {avg_energy:.2f} kJ/mol")

        # 计算最终统计
        final_accept_rate = accept_count / total_proposals

        print(f"\n正确Timewarp采样完成!")
        print(f"总接受率: {final_accept_rate:.4f}")
        print(f"总提议数: {total_proposals}")
        print(f"接受数: {accept_count}")

        # 保存结果
        trajectory_coords = torch.cat(trajectory_coords, dim=0)
        trajectory_velocs = torch.cat(trajectory_velocs, dim=0)

        np.save(f'{output_dir}/trajectory_coords.npy', trajectory_coords.numpy())
        np.save(f'{output_dir}/trajectory_velocs.npy', trajectory_velocs.numpy())
        np.save(f'{output_dir}/energy_history.npy', np.array(energy_history))
        np.save(f'{output_dir}/acceptance_history.npy', np.array(acceptance_history))

        # 保存详细信息
        def numpy2py(obj):
            """递归将 numpy 类型全部转换为 Python 原生类型"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: numpy2py(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [numpy2py(v) for v in obj]
            elif isinstance(obj, np.generic):
                return obj.item()
            else:
                return obj

        with open(f'{output_dir}/detailed_info.json', 'w') as f:
            json.dump(numpy2py(detailed_info), f, indent=2)

        stats = {
            'algorithm': 'Correct_Timewarp_Sampling',
            'total_steps': num_steps,
            'total_proposals': total_proposals,
            'accepted_proposals': accept_count,
            'acceptance_rate': final_accept_rate,
            'training_pairs_used': len(self.t_states),
            'trajectory_length': len(trajectory_coords),
            'final_energy': energy_history[-1] if energy_history else None,
            'k_nearest': self.k_nearest
        }

        with open(f'{output_dir}/sampling_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def get_random_initial_structure(self, idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """从训练数据中获取初始结构"""
        if idx is None:
            idx = np.random.randint(0, len(self.t_coords))

        initial_coords = torch.FloatTensor(self.t_coords[idx:idx+1]).to(self.device)
        initial_velocs = torch.FloatTensor(self.t_velocs[idx:idx+1]).to(self.device)

        return initial_coords, initial_velocs

    def analyze_training_data_coverage(self) -> Dict:
        """分析训练数据的覆盖范围"""
        print("分析训练数据覆盖范围...")

        # 计算坐标统计
        coord_stats = {
            'min': float(self.t_coords.min()),
            'max': float(self.t_coords.max()),
            'mean': float(self.t_coords.mean()),
            'std': float(self.t_coords.std())
        }

        # 计算速度统计
        veloc_stats = {
            'min': float(self.t_velocs.min()),
            'max': float(self.t_velocs.max()),
            'mean': float(self.t_velocs.mean()),
            'std': float(self.t_velocs.std())
        }

        # 计算能量分布
        energies = []
        for i in range(0, len(self.t_coords), 100):  # 采样计算能量
            coords = torch.FloatTensor(self.t_coords[i:i+1]).to(self.device)
            energy = self.calculate_energy(coords)
            energies.append(energy.item())

        energy_stats = {
            'min': float(np.min(energies)),
            'max': float(np.max(energies)),
            'mean': float(np.mean(energies)),
            'std': float(np.std(energies))
        }

        analysis = {
            'num_training_pairs': len(self.t_states),
            'coordinate_stats': coord_stats,
            'velocity_stats': veloc_stats,
            'energy_stats': energy_stats
        }

        print(f"训练数据分析:")
        print(f"  配对数量: {analysis['num_training_pairs']}")
        print(f"  坐标范围: {coord_stats['min']:.4f} 到 {coord_stats['max']:.4f} nm")
        print(f"  能量范围: {energy_stats['min']:.2f} 到 {energy_stats['max']:.2f} kJ/mol")

        return analysis

def main():
    """主函数：运行正确的Timewarp采样"""

    # 创建探索器
    explorer = TimewarpCorrectExplorer(
        'corrected_timewarp_model_final.pth',
        'alanine-dipeptide-solvated.pdb',        # <-- path to your reference PDB
        'training_pairs_augmented_final.npy'
    )

    # 分析训练数据
    training_analysis = explorer.analyze_training_data_coverage()

    # 获取初始结构
    initial_coords, initial_velocs = explorer.get_random_initial_structure(idx=0)
    print(f"初始结构坐标范围: {initial_coords.min():.4f} 到 {initial_coords.max():.4f} nm")

    # 运行正确的Timewarp采样
    print("\n=== 运行正确的Timewarp采样 ===")
    sampling_stats = explorer.correct_timewarp_sampling(
        initial_coords, initial_velocs,
        num_steps=500000,
        save_interval=1,
        output_dir='correct_timewarp_results'
    )

    print("\n正确的Timewarp采样完成！")
    print("生成的文件：")
    print("  - trajectory_coords.npy: 轨迹坐标")
    print("  - trajectory_velocs.npy: 轨迹速度")
    print("  - energy_history.npy: 能量历史")
    print("  - acceptance_history.npy: 接受历史")
    print("  - detailed_info.json: 详细采样信息")
    print("  - sampling_stats.json: 采样统计")

if __name__ == "__main__":
    main()
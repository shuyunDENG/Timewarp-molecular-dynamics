import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import os

class MolecularDataset(Dataset):
    """分子动力学数据集 - 适配新模型架构"""
    def __init__(self, data_indices, full_data, num_atom_types=4):
        """
        Args:
            data_indices: 数据索引列表
            full_data: 完整数据数组
            num_atom_types: 原子类型数量
        """
        self.data_indices = data_indices
        self.full_data = full_data
        self.num_atom_types = num_atom_types
        _, _, self.num_atoms, _ = full_data.shape

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]
        pair = self.full_data[data_idx]  # [2, num_atoms, 6]

        # 分离时间步
        t_data = pair[0]      # [num_atoms, 6] - t时刻数据
        t_tau_data = pair[1]  # [num_atoms, 6] - t+τ时刻数据

        # 分离位置 (新模型只需要位置，不需要速度作为输入)
        x_coords = t_data[:, :3]      # t时刻位置 - 条件输入
        y_coords = t_tau_data[:, :3]  # t+τ时刻位置 - 目标输出

        # 注意：不再手动中心化，模型内部会处理
        # x_coords = x_coords - np.mean(x_coords, axis=0, keepdims=True)
        # y_coords = y_coords - np.mean(y_coords, axis=0, keepdims=True)

        # alanine-dipeptide的原子类型序列
        atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)

        return {
            'atom_types': atom_types,
            'x_coords': torch.FloatTensor(x_coords),      # 条件坐标
            'y_coords': torch.FloatTensor(y_coords),      # 目标坐标
            'data_idx': data_idx  # 添加数据索引用于调试
        }

def analyze_data_distribution(data_path):
    """分析数据分布"""
    print("=== 数据分析 ===")
    data = np.load(data_path)
    print(f"数据形状: {data.shape}")

    # 只分析位置的分布 (新模型只关心位置)
    positions = data[:, :, :, :3].reshape(-1, 3)

    print(f"位置统计:")
    print(f"  均值: {positions.mean(axis=0)}")
    print(f"  标准差: {positions.std(axis=0)}")
    print(f"  范围: {positions.min(axis=0)} 到 {positions.max(axis=0)}")

    # 分析位置变化的统计
    position_changes = []
    for i in range(data.shape[0]):
        pos_change = data[i, 1, :, :3] - data[i, 0, :, :3]  # t+τ - t
        position_changes.append(pos_change)

    position_changes = np.array(position_changes).reshape(-1, 3)
    print(f"位置变化统计:")
    print(f"  均值: {position_changes.mean(axis=0)}")
    print(f"  标准差: {position_changes.std(axis=0)}")

    return data

def normalize_data(data):
    """标准化数据 - 只标准化位置"""
    print("=== 数据标准化 ===")
    normalized_data = data.copy()

    # 只标准化位置
    positions = data[:, :, :, :3]

    # 计算全局统计量
    pos_mean = positions.mean()
    pos_std = positions.std()

    print(f"位置标准化: 均值={pos_mean:.4f}, 标准差={pos_std:.4f}")

    # 标准化位置
    normalized_data[:, :, :, :3] = (positions - pos_mean) / pos_std
    # 保持速度不变 (虽然新模型不用，但保持数据完整性)
    normalized_data[:, :, :, 3:] = data[:, :, :, 3:]

    return normalized_data, (pos_mean, pos_std)

def compute_physics_metrics(model, dataloader, device):
    """计算物理相关的指标 - 适配新模型"""
    model.eval()
    total_position_error = 0
    total_likelihood = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            atom_types = batch['atom_types'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)

            # 计算似然 (训练模式)
            _, log_likelihood = model(
                atom_types, x_coords, y_coords, reverse=False
            )
            avg_likelihood = log_likelihood.mean()

            # 采样预测 (推理模式)
            pred_coords, _ = model(
                atom_types, x_coords, reverse=True
            )

            # 计算位置L2误差
            pos_error = torch.mean((pred_coords - y_coords) ** 2)

            total_position_error += pos_error.item()
            total_likelihood += avg_likelihood.item()
            total_samples += 1

    return total_position_error / total_samples, total_likelihood / total_samples

def sample_and_visualize(model, dataloader, device, num_samples=5):
    """采样并可视化结果"""
    model.eval()

    with torch.no_grad():
        # 获取一个批次的数据
        batch = next(iter(dataloader))
        atom_types = batch['atom_types'][:num_samples].to(device)
        x_coords = batch['x_coords'][:num_samples].to(device)
        y_coords = batch['y_coords'][:num_samples].to(device)

        # 多次采样来评估模型的随机性
        samples = []
        for _ in range(10):
            pred_coords, _ = model(atom_types, x_coords, reverse=True)
            samples.append(pred_coords.cpu().numpy())

        samples = np.array(samples)  # [10, num_samples, num_atoms, 3]

        # 计算采样方差
        sample_variance = np.var(samples, axis=0).mean()
        print(f"采样方差: {sample_variance:.6f}")

        # 可视化第一个分子的第一个原子的轨迹
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.scatter(samples[:, 0, 0, 0], samples[:, 0, 0, 1], alpha=0.6, label='预测')
        plt.scatter(y_coords[0, 0, 0].cpu(), y_coords[0, 0, 1].cpu(), color='red', s=100, label='真实')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('原子轨迹 (XY平面)')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.hist(samples[:, 0, 0, 0], bins=20, alpha=0.7, label='X坐标分布')
        plt.axvline(y_coords[0, 0, 0].cpu(), color='red', linestyle='--', label='真实X')
        plt.xlabel('X坐标')
        plt.ylabel('频次')
        plt.title('X坐标分布')
        plt.legend()

        plt.subplot(1, 3, 3)
        # 计算与真实值的距离
        distances = np.sqrt(np.sum((samples - y_coords.cpu().numpy()) ** 2, axis=-1))
        mean_distances = distances.mean(axis=(1, 2))
        plt.plot(mean_distances, 'o-')
        plt.xlabel('采样次数')
        plt.ylabel('平均距离误差')
        plt.title('预测误差分布')

        plt.tight_layout()
        plt.savefig('sampling_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

def train_timewarp_model_corrected(
    data_path='training_pairs_augmented_final.npy',
    num_epochs=200,
    batch_size=16,
    learning_rate=1e-4,
    test_size=0.2,
    normalize=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """修正后的训练函数"""

    print("=== 开始训练修正版Timewarp模型 ===")

    # 1. 分析和加载数据
    full_data = analyze_data_distribution(data_path)

    # 2. 数据标准化
    if normalize:
        full_data, norm_stats = normalize_data(full_data)

    # 3. 划分训练集和测试集
    num_samples = len(full_data)
    indices = np.arange(num_samples)
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42, shuffle=True
    )

    print(f"训练集大小: {len(train_indices)}")
    print(f"测试集大小: {len(test_indices)}")

    # 4. 创建数据集
    train_dataset = MolecularDataset(train_indices, full_data)
    test_dataset = MolecularDataset(test_indices, full_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. 创建模型 - 使用能被头数整除的维度配置
    config = {
        'num_atom_types': 4,              # alanine dipeptide的原子类型
        'embedding_dim': 32,              # 32能被多种头数整除
        'hidden_dim': 96,                 # 96 = 32 * 3，能被3整除
        'num_coupling_layers': 6,         # 较少的耦合层
        'lengthscales': [0.1, 0.3, 0.8]   # 3个头，96/3=32维度完美匹配
    }

    model = create_timewarp_model(config).to(device)

    # 使用论文中的优化器设置
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)

    print(f"模型参数量: {sum([p.numel() for p in model.parameters()]):,}")

    # 6. 训练历史记录
    train_losses = []
    test_losses = []
    train_likelihoods = []
    test_likelihoods = []
    position_errors = []

    best_test_likelihood = float('-inf')

    # 7. 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_likelihood = 0
        num_train_batches = 0

        for batch in train_loader:
            atom_types = batch['atom_types'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)

            optimizer.zero_grad()

            # 计算负对数似然
            _, log_likelihood = model(
                atom_types, x_coords, y_coords, reverse=False
            )

            # 负对数似然作为损失
            loss = -log_likelihood.mean()

            # 检查数值稳定性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到NaN或Inf损失在epoch {epoch}")
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_likelihood += log_likelihood.mean().item()
            num_train_batches += 1

        avg_train_loss = train_loss / num_train_batches
        avg_train_likelihood = train_likelihood / num_train_batches

        # 测试阶段
        model.eval()
        test_loss = 0
        test_likelihood = 0
        num_test_batches = 0

        with torch.no_grad():
            for batch in test_loader:
                atom_types = batch['atom_types'].to(device)
                x_coords = batch['x_coords'].to(device)
                y_coords = batch['y_coords'].to(device)

                _, log_likelihood = model(
                    atom_types, x_coords, y_coords, reverse=False
                )

                loss = -log_likelihood.mean()
                test_loss += loss.item()
                test_likelihood += log_likelihood.mean().item()
                num_test_batches += 1

        avg_test_loss = test_loss / num_test_batches
        avg_test_likelihood = test_likelihood / num_test_batches

        # 学习率调度
        scheduler.step(avg_test_likelihood)

        # 计算物理指标
        if epoch % 10 == 0:
            pos_error, _ = compute_physics_metrics(model, test_loader, device)
            position_errors.append(pos_error)

            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Train LL: {avg_train_likelihood:.4f}, "
                  f"Test LL: {avg_test_likelihood:.4f}, "
                  f"Pos Error: {pos_error:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Train LL: {avg_train_likelihood:.4f}, "
                  f"Test LL: {avg_test_likelihood:.4f}")

        # 保存最佳模型
        if avg_test_likelihood > best_test_likelihood:
            best_test_likelihood = avg_test_likelihood
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'test_likelihood': avg_test_likelihood
            }, 'best_timewarp_model.pth')

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_likelihoods.append(avg_train_likelihood)
        test_likelihoods.append(avg_test_likelihood)

        # 早停检查 (基于似然而不是损失)
        if epoch > 50 and len(test_likelihoods) > 20:
            recent_likelihoods = test_likelihoods[-20:]
            if all(recent_likelihoods[i] >= recent_likelihoods[i+1] for i in range(len(recent_likelihoods)-5)):
                print(f"早停：测试似然停止改善")
                break

    # 8. 绘制训练曲线
    plt.figure(figsize=(20, 8))

    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Test Loss')

    plt.subplot(2, 3, 2)
    plt.plot(train_likelihoods, label='Train Likelihood', alpha=0.7)
    plt.plot(test_likelihoods, label='Test Likelihood', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Log Likelihood')
    plt.legend()
    plt.title('Training vs Test Likelihood')

    plt.subplot(2, 3, 3)
    epochs_with_metrics = list(range(0, len(position_errors) * 10, 10))
    plt.plot(epochs_with_metrics, position_errors, 'o-', label='Position Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Position Prediction Error')
    plt.yscale('log')

    plt.subplot(2, 3, 4)
    # 显示学习率变化
    lr_history = []
    for epoch in range(len(train_losses)):
        if epoch % 10 == 0:
            lr_history.append(optimizer.param_groups[0]['lr'])
    lr_epochs = list(range(0, len(lr_history) * 10, 10))
    plt.plot(lr_epochs, lr_history, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')

    plt.subplot(2, 3, 5)
    # 显示损失的梯度
    if len(train_losses) > 1:
        train_grad = np.gradient(train_losses)
        test_grad = np.gradient(test_losses)
        plt.plot(train_grad, label='Train Loss Gradient', alpha=0.7)
        plt.plot(test_grad, label='Test Loss Gradient', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss Gradient')
        plt.legend()
        plt.title('Loss Gradients')

    plt.subplot(2, 3, 6)
    # 显示似然的移动平均
    window = 10
    if len(test_likelihoods) > window:
        test_ll_smooth = np.convolve(test_likelihoods, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(test_likelihoods)), test_ll_smooth, label='Test LL (Smoothed)')
        plt.xlabel('Epoch')
        plt.ylabel('Smoothed Log Likelihood')
        plt.legend()
        plt.title('Smoothed Test Likelihood')

    plt.tight_layout()
    plt.savefig('corrected_training_diagnostics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 9. 采样分析
    print("\n=== 采样分析 ===")
    sample_and_visualize(model, test_loader, device)

    # 10. 保存最终结果
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_likelihoods': train_likelihoods,
        'test_likelihoods': test_likelihoods,
        'position_errors': position_errors,
        'normalization_stats': norm_stats if normalize else None,
        'best_test_likelihood': best_test_likelihood
    }, 'corrected_timewarp_model_final.pth')

    return model, train_losses, test_losses, train_likelihoods, test_likelihoods

def test_model_predictions_corrected(model_path='corrected_timewarp_model_final.pth'):
    """测试修正模型的预测质量"""
    print("=== 测试修正模型预测质量 ===")

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = create_timewarp_model(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 生成测试数据
    batch_size = 5
    num_atoms = 22
    atom_types = torch.tensor([[0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3]] * batch_size)
    x_coords = torch.randn(batch_size, num_atoms, 3) * 0.5  # 更大的初始坐标范围

    with torch.no_grad():
        # 多次预测，检查一致性
        predictions = []
        likelihoods = []

        for _ in range(20):
            pred_coords, _ = model(atom_types, x_coords, reverse=True)
            predictions.append(pred_coords)

        # 计算预测的统计特性
        coord_preds = torch.stack(predictions)  # [20, batch_size, num_atoms, 3]

        # 检查采样多样性
        coord_variance = torch.var(coord_preds, dim=0).mean()
        coord_mean = torch.mean(coord_preds, dim=0)
        coord_std = torch.std(coord_preds, dim=0).mean()

        print(f"坐标预测方差: {coord_variance:.6f}")
        print(f"坐标预测标准差: {coord_std:.6f}")
        print(f"坐标预测范围: {coord_mean.min():.4f} 到 {coord_mean.max():.4f}")

        # 检查物理合理性
        # 计算原子间距离
        distances = []
        for pred in predictions[:5]:  # 取前5个预测
            for i in range(batch_size):
                coords = pred[i]  # [num_atoms, 3]
                dist_matrix = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                # 排除对角线
                mask = ~torch.eye(num_atoms, dtype=bool)
                distances.extend(dist_matrix[mask].tolist())

        distances = torch.tensor(distances)
        print(f"原子间距离统计:")
        print(f"  最小距离: {distances.min():.4f}")
        print(f"  平均距离: {distances.mean():.4f}")
        print(f"  最大距离: {distances.max():.4f}")

        # 检查是否有不合理的近距离
        close_contacts = (distances < 0.1).sum().item()
        print(f"过近接触 (<0.1): {close_contacts} / {len(distances)} ({100*close_contacts/len(distances):.2f}%)")

# 使用示例
if __name__ == "__main__":
    # 训练修正后的模型
    print("开始训练修正版Timewarp模型...")
    model, train_losses, test_losses, train_ll, test_ll = train_timewarp_model_corrected(
        data_path='training_pairs_augmented_final.npy',
        num_epochs=150,
        batch_size=16,
        learning_rate=1e-4,
        normalize=True
    )

    # 测试模型质量
    print("\n开始测试模型...")
    test_model_predictions_corrected()
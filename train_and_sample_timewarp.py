import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# ====== 你的模型文件，确保导入 create_integrated_timewarp_model ======
from model_et_sampling import create_integrated_timewarp_model

# ====== 数据集定义（用你现成的类）======
class MolecularDataset(Dataset):
    def __init__(self, data_indices, full_data):
        self.data_indices = data_indices
        self.full_data = full_data
        _, _, self.num_atoms, _ = full_data.shape

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        data_idx = self.data_indices[idx]
        pair = self.full_data[data_idx]  # [2, num_atoms, 6]
        t_data = pair[0]
        t_tau_data = pair[1]
        x_coords = t_data[:, :3]
        y_coords = t_tau_data[:, :3]
        atom_types = torch.tensor([0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3], dtype=torch.long)
        return {
            'atom_types': atom_types,
            'x_coords': torch.FloatTensor(x_coords),
            'y_coords': torch.FloatTensor(y_coords),
            'data_idx': data_idx
        }

def normalize_data(data):
    positions = data[:, :, :, :3]
    pos_mean = positions.mean()
    pos_std = positions.std()
    normed = data.copy()
    normed[:, :, :, :3] = (positions - pos_mean) / pos_std
    return normed, (pos_mean, pos_std)

# ========== 训练主循环 ==========
def train_timewarp(
    data_path='training_pairs_augmented_final.npy',
    num_epochs=150,
    batch_size=16,
    learning_rate=1e-4,
    test_size=0.2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    model_ckpt='best_timewarp_model.pth'
):
    print("=== 开始训练 Timewarp 集成模型 ===")
    full_data = np.load(data_path)
    full_data, norm_stats = normalize_data(full_data)

    num_samples = len(full_data)
    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42, shuffle=True)

    train_dataset = MolecularDataset(train_idx, full_data)
    test_dataset = MolecularDataset(test_idx, full_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    config = {
        'num_atom_types': 4,
        'embedding_dim': 32,
        'hidden_dim': 96,
        'num_coupling_layers': 6,
        'lengthscales': [0.1, 0.3, 0.8],
        'temperature': 310.0
    }
    model = create_integrated_timewarp_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)
    best_test_ll = float('-inf')

    train_losses, test_losses, train_lls, test_lls = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, epoch_ll = 0, 0
        for batch in train_loader:
            atom_types = batch['atom_types'].to(device)
            x_coords = batch['x_coords'].to(device)
            y_coords = batch['y_coords'].to(device)

            optimizer.zero_grad()
            _, log_likelihood = model(atom_types, x_coords, torch.zeros_like(x_coords), y_coords, torch.zeros_like(y_coords), reverse=False)
            loss = -log_likelihood.mean()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_ll += log_likelihood.mean().item()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_ll = epoch_ll / len(train_loader)

        model.eval()
        test_loss, test_ll = 0, 0
        with torch.no_grad():
            for batch in test_loader:
                atom_types = batch['atom_types'].to(device)
                x_coords = batch['x_coords'].to(device)
                y_coords = batch['y_coords'].to(device)
                _, log_likelihood = model(atom_types, x_coords, torch.zeros_like(x_coords), y_coords, torch.zeros_like(y_coords), reverse=False)
                loss = -log_likelihood.mean()
                test_loss += loss.item()
                test_ll += log_likelihood.mean().item()
        avg_test_loss = test_loss / len(test_loader)
        avg_test_ll = test_ll / len(test_loader)

        scheduler.step(avg_test_ll)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_lls.append(avg_train_ll)
        test_lls.append(avg_test_ll)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Train LL: {avg_train_ll:.4f} | Test LL: {avg_test_ll:.4f}")

        # 保存最佳
        if avg_test_ll > best_test_ll:
            best_test_ll = avg_test_ll
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'norm_stats': norm_stats,
                'epoch': epoch,
                'test_ll': avg_test_ll
            }, model_ckpt)

    # 画学习曲线
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.title('Loss')
    plt.savefig('train_loss_curve.png', dpi=120)
    plt.close()
    print("训练结束！最佳模型已保存：", model_ckpt)

def sample_with_trained_model(model_ckpt='best_timewarp_model.pth', num_steps=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_ckpt, map_location=device)
    model = create_integrated_timewarp_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 示例采样
    batch_size, num_atoms = 1, 22
    atom_types = torch.tensor([[0, 1, 0, 2, 3, 3, 3, 0, 3, 0, 2, 1, 3, 0, 0, 2, 3, 3, 3, 3, 3, 3]], dtype=torch.long).to(device)
    initial_coords = torch.randn(batch_size, num_atoms, 3).to(device) * 0.5
    initial_velocs = torch.randn(batch_size, num_atoms, 3).to(device) * 0.1

    print("正在执行集成 MCMC 采样...")
    mcmc_results = model.mcmc_sampling(atom_types, initial_coords, initial_velocs, num_steps=num_steps, save_interval=50, batch_proposals=5)
    print("采样结束！最后采样能量：", mcmc_results['energy_history'][-1])
    return mcmc_results

if __name__ == '__main__':
    # 步骤1：训练模型
    train_timewarp(
        data_path='training_pairs_augmented_final.npy',
        num_epochs=100,
        batch_size=16,
        learning_rate=1e-5,
        model_ckpt='best_timewarp_model.pth'
    )
    # 步骤2：采样
    sample_with_trained_model(model_ckpt='best_timewarp_model.pth', num_steps=1000)
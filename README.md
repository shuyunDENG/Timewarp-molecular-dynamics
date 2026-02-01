# Timewarp Molecular Dynamics

Implementation of **Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics** (NeurIPS 2023) for accelerated molecular dynamics simulation using conditional normalizing flows.

## Overview

This project implements the Timewarp architecture for learning accelerated molecular dynamics. Timewarp uses conditional normalizing flows to directly model the conditional distribution p(x(t+τ)|x(t)), allowing large timestep predictions while maintaining physical accuracy. This enables:

- **Accelerated sampling** of molecular configurations
- **Long timestep predictions** (e.g., 100 fs instead of 1 fs)
- **Transferable dynamics** across different molecular systems
- **MCMC sampling** from Boltzmann distributions

### Key Features

- **RealNVP-based architecture** with coupling layers for invertible transformations
- **Kernel Self-Attention** mechanism using RBF kernels for local atomic interactions
- **Atom Transformer** networks for scale and shift parameter generation
- **Augmented normalizing flows** incorporating velocity variables
- **Translation/rotation equivariant** coordinate handling
- **Metropolis-Hastings MCMC** for sampling with acceptance/rejection

## Architecture

The Timewarp model consists of several key components:

### 1. Atom Embedder
Embeds discrete atom types into continuous representations:
```
atom_types (categorical) → embedding vectors (continuous)
```

### 2. Kernel Self-Attention
Uses RBF (Radial Basis Function) kernels to capture local atomic interactions at multiple length scales:
```
K(r_ij) = exp(-r_ij² / (2σ²))
```
Multiple lengthscales (σ) enable capturing interactions at different spatial ranges.

### 3. Atom Transformer
Neural network that processes atomic features to generate scale (s_θ) and shift (t_θ) parameters for coupling layers. Uses kernel self-attention followed by MLPs.

### 4. RealNVP Coupling Layers
Affine coupling transformations that ensure invertibility:
```
z₁ = x₁
z₂ = x₂ ⊙ exp(s_θ(x₁, c)) + t_θ(x₁, c)
```
where c is the conditional input (previous timestep).

### 5. Flow Model
Stacks multiple coupling layers with alternating coordinate splitting to build expressive conditional distributions.

## System Tested

**Alanine Dipeptide**: A 22-atom molecular system commonly used as a benchmark in molecular dynamics:
- 22 atoms with 4 different atom types
- Simulation temperature: 310K
- Training timestep: τ = 100 fs (compared to standard 1 fs MD)

## Dataset

Training data consists of molecular trajectory pairs:
- **Size**: ~1,500 training pairs
- **Format**: [x(t), x(t+τ)] coordinate pairs
- **Source**: OpenMM molecular dynamics simulations
- **Normalization**: Mean-centered coordinates with standardized positions
- **Augmentation**: Translation and rotation augmentation for equivariance

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0

# Molecular dynamics (for data generation and energy evaluation)
openmm>=8.0.0
mdtraj>=1.9.7

# Optional for visualization
nglview
```

### Setup

```bash
git clone https://github.com/shuyunDENG/Timewarp-molecular-dynamics.git
cd Timewarp-molecular-dynamics
pip install -r requirements.txt
```

## Usage

### Training

Train the Timewarp model on molecular dynamics data:

```python
from train_and_sample_timewarp import train_timewarp

# Train with default parameters
train_timewarp(
    data_path='training_pairs_augmented_final.npy',
    num_epochs=100,
    batch_size=16,
    learning_rate=1e-5,
    model_ckpt='best_timewarp_model.pth'
)
```

### Sampling

Generate molecular configurations using trained model:

```python
from train_and_sample_timewarp import sample_with_trained_model

# Sample with MCMC
mcmc_results = sample_with_trained_model(
    model_ckpt='best_timewarp_model.pth',
    num_steps=1000
)
```

### Exploration with Physics-Based Sampling

Use the corrected explorer for MCMC sampling with proper energy evaluation:

```python
from exploration_amelioration import TimewarpCorrectExplorer

# Initialize explorer with trained model
explorer = TimewarpCorrectExplorer(
    model_path='best_timewarp_model.pth',
    training_pairs_path='training_pairs_augmented_final.npy',
    pdb_structure='alanine_dipeptide.pdb',
    temperature=310.0
)

# Run MCMC exploration
results = explorer.explore(
    num_steps=1000,
    initial_coords=initial_positions,
    mcmc_interval=10,
    save_interval=50
)
```

## File Structure

### Core Implementation Files

- **model_timewarp.py**: Main Timewarp model implementation
  - `AtomEmbedder`: Atom type embedding layer
  - `KernelSelfAttention`: RBF kernel-based attention mechanism
  - `AtomTransformer`: Neural network for s_θ and t_θ functions
  - `TimewarpCouplingLayer`: RealNVP coupling layer
  - `TimewarpModel`: Complete integrated model

- **train_and_sample_timewarp.py**: Training and sampling script
  - `train_timewarp()`: Main training loop with likelihood maximization
  - `sample_with_trained_model()`: MCMC sampling with trained model

- **exploration_amelioration.py**: Corrected MCMC explorer
  - `TimewarpCorrectExplorer`: Physics-based sampling with OpenMM energy evaluation
  - Proper acceptance rule with Jacobian determinants

- **Train_test.py**: Training utilities and physics metrics
  - `train_timewarp_model_corrected()`: Training with diagnostics
  - `compute_physics_metrics()`: Physics-based evaluation metrics

### Data Files

- `training_pairs_augmented_final.npy`: Molecular trajectory training pairs
- `alanine_dipeptide.pdb`: Molecular structure file
- `*.pth`: Saved model checkpoints

### Documentation

- `Rapport_Timewarp.pdf`: Implementation report (French)
- `1889_Timewarp_Transferable_Acc.pdf`: Original NeurIPS 2023 paper

## Training Results

Based on experiments conducted:

- **Training Dataset**: ~1,500 pairs of molecular configurations
- **Training Stability**: Converged negative log-likelihood loss
- **Test Performance**: Stable generalization to test set
- **MCMC Acceptance Rate**: ~10% (typical for this system)
- **Computational Efficiency**: 100x timestep acceleration (100 fs vs 1 fs)

### Limitations

- **Limited dataset size**: ~1,500 pairs may not fully capture phase space
- **Acceptance rates**: Moderate acceptance (~10%) suggests room for improvement
- **Generalization**: Tested primarily on alanine dipeptide

## Model Configuration

Default hyperparameters used:

```python
config = {
    'num_atom_types': 4,           # Atom types in alanine dipeptide
    'embedding_dim': 32,            # Atom embedding dimension
    'hidden_dim': 96,               # Hidden layer dimension
    'num_coupling_layers': 6,       # Number of RealNVP layers
    'lengthscales': [0.1, 0.3, 0.8], # RBF kernel lengthscales (Å)
    'temperature': 310.0            # Simulation temperature (K)
}
```

## References

### Paper
```
@inproceedings{timewarp2023,
  title={Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics},
  author={Chennakesava, Leon and Greenberg, Fiona and Ma, Frank and Strahan, Jarrid and Noh, Hyunho and Fung, Nicholas and Noé, Frank and Chodera, John D and Schoenholz, Samuel S and Köhler, Jonas},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

### Related Work

- **Normalizing Flows**: Rezende & Mohamed (2015), Dinh et al. (2017)
- **Molecular Dynamics**: OpenMM (Eastman et al., 2017)
- **Augmented Normalizing Flows**: Huang et al. (2018)

## License

This implementation is for research and educational purposes. Please refer to the original Timewarp paper for the official implementation and licensing details.

## Acknowledgments

- Original Timewarp authors for the architecture and methodology
- OpenMM development team for molecular dynamics simulation tools
- NeurIPS 2023 reviewers and community

## Contact

For questions or issues, please open an issue on the GitHub repository.

"""
Basic Usage Example for Timewarp Model

This script demonstrates:
1. Creating a Timewarp model
2. Training on molecular dynamics data
3. Sampling new molecular configurations
"""

import torch
import sys
sys.path.append('..')

from src.model_timewarp import create_timewarp_model, paper_config

def example_model_creation():
    """Example: Create a Timewarp model"""
    print("=" * 50)
    print("Example 1: Creating Timewarp Model")
    print("=" * 50)

    # Define configuration for alanine dipeptide (22 atoms, 4 atom types)
    config = {
        'num_atom_types': 4,           # Number of different atom types
        'embedding_dim': 32,            # Atom embedding dimension
        'hidden_dim': 96,               # Hidden layer dimension
        'num_coupling_layers': 6,       # Number of RealNVP layers
        'lengthscales': [0.1, 0.3, 0.8] # RBF kernel lengthscales (Angstroms)
    }

    # Create model
    model = create_timewarp_model(config)
    print(f"Created Timewarp model with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Architecture: {config['num_coupling_layers']} coupling layers")
    print(f"Kernel lengthscales: {config['lengthscales']}")

    return model


def example_forward_pass():
    """Example: Forward pass through the model"""
    print("\n" + "=" * 50)
    print("Example 2: Forward Pass (Training Mode)")
    print("=" * 50)

    config = {
        'num_atom_types': 4,
        'embedding_dim': 32,
        'hidden_dim': 96,
        'num_coupling_layers': 6,
        'lengthscales': [0.1, 0.3, 0.8]
    }

    model = create_timewarp_model(config)

    # Create dummy data for alanine dipeptide (22 atoms)
    batch_size = 2
    num_atoms = 22

    # Atom types (4 different types for alanine dipeptide)
    atom_types = torch.randint(0, 4, (batch_size, num_atoms))

    # Coordinates at time t (initial state)
    x_coords = torch.randn(batch_size, num_atoms, 3) * 2.0  # In Angstroms
    x_velocs = torch.zeros(batch_size, num_atoms, 3)         # Zero velocities

    # Coordinates at time t+τ (target state)
    y_coords = torch.randn(batch_size, num_atoms, 3) * 2.0
    y_velocs = torch.randn(batch_size, num_atoms, 3) * 0.1

    # Forward pass: Compute log-likelihood
    model.train()
    (output_coords, output_velocs), log_likelihood = model(
        atom_types, x_coords, x_velocs, y_coords, y_velocs, reverse=False
    )

    print(f"Input shape: {x_coords.shape}")
    print(f"Output shape: {output_coords.shape}")
    print(f"Log-likelihood: {log_likelihood.mean().item():.4f}")
    print(f"Per-atom log-likelihood: {log_likelihood.mean().item() / num_atoms:.4f}")


def example_sampling():
    """Example: Sampling from the model"""
    print("\n" + "=" * 50)
    print("Example 3: Sampling (Generative Mode)")
    print("=" * 50)

    config = {
        'num_atom_types': 4,
        'embedding_dim': 32,
        'hidden_dim': 96,
        'num_coupling_layers': 6,
        'lengthscales': [0.1, 0.3, 0.8]
    }

    model = create_timewarp_model(config)
    model.eval()

    # Create initial state
    batch_size = 1
    num_atoms = 22
    atom_types = torch.randint(0, 4, (batch_size, num_atoms))
    x_coords = torch.randn(batch_size, num_atoms, 3) * 2.0
    x_velocs = torch.zeros(batch_size, num_atoms, 3)

    # Sample future state
    with torch.no_grad():
        (sampled_coords, sampled_velocs), _ = model(
            atom_types, x_coords, x_velocs, reverse=True
        )

    print(f"Initial coordinates shape: {x_coords.shape}")
    print(f"Sampled coordinates shape: {sampled_coords.shape}")
    print(f"Sampled velocities shape: {sampled_velocs.shape}")

    # Compute displacement
    displacement = (sampled_coords - x_coords).norm(dim=-1)
    print(f"Average atomic displacement: {displacement.mean().item():.4f} Å")
    print(f"Maximum atomic displacement: {displacement.max().item():.4f} Å")


def example_multi_sampling():
    """Example: Generate multiple samples"""
    print("\n" + "=" * 50)
    print("Example 4: Multiple Sample Generation")
    print("=" * 50)

    config = {
        'num_atom_types': 4,
        'embedding_dim': 32,
        'hidden_dim': 96,
        'num_coupling_layers': 6,
        'lengthscales': [0.1, 0.3, 0.8]
    }

    model = create_timewarp_model(config)

    # Initial state
    batch_size = 1
    num_atoms = 22
    atom_types = torch.randint(0, 4, (batch_size, num_atoms))
    x_coords = torch.randn(batch_size, num_atoms, 3) * 2.0
    x_velocs = torch.zeros(batch_size, num_atoms, 3)

    # Generate 5 samples
    num_samples = 5
    samples_coords, samples_velocs = model.sample(
        atom_types, x_coords, x_velocs, num_samples=num_samples
    )

    print(f"Generated {num_samples} samples")
    print(f"Samples shape: {samples_coords.shape}")  # [num_samples, batch_size, num_atoms, 3]

    # Analyze sample diversity
    for i in range(num_samples):
        displacement = (samples_coords[i] - x_coords).norm(dim=-1)
        print(f"Sample {i+1} - Avg displacement: {displacement.mean().item():.4f} Å")


if __name__ == '__main__':
    # Run all examples
    example_model_creation()
    example_forward_pass()
    example_sampling()
    example_multi_sampling()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)

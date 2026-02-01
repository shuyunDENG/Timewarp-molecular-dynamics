# Examples

This directory contains example code and Jupyter notebooks demonstrating various aspects of the Timewarp molecular dynamics implementation.

## Python Examples

### `basic_usage.py`

Demonstrates fundamental usage of the Timewarp model:
- Creating a Timewarp model with custom configuration
- Forward pass (training mode) to compute log-likelihood
- Sampling (generative mode) to generate future molecular states
- Multiple sample generation for exploring the learned distribution

**Run it:**
```bash
cd examples/
python basic_usage.py
```

## Jupyter Notebooks

### `model_timewarp.ipynb`

Interactive notebook exploring the Timewarp model architecture:
- Model component visualization
- Step-by-step walkthrough of the coupling layers
- Analysis of kernel self-attention mechanism

### `Normalising_Flow_data.ipynb`

Data preparation and normalizing flow basics:
- Loading molecular dynamics trajectory data
- Data normalization and augmentation
- Understanding the flow-based modeling approach

### `exploration_final.ipynb`

Advanced MCMC sampling and exploration:
- Physics-based MCMC sampling with OpenMM
- Acceptance rate analysis
- Ramachandran plot generation for validation
- Energy landscape exploration

### `Comparaison_Final.ipynb`

Comparative analysis:
- Comparison with standard MD simulations
- Performance benchmarking
- Statistical analysis of generated samples

## Usage Notes

1. **Data Requirements**: Most examples require the training data file `training_pairs_augmented_final.npy` to be present in the `data/` directory.

2. **GPU Acceleration**: Examples will automatically use CUDA if available. Set `device='cpu'` to force CPU execution.

3. **Dependencies**: Ensure all requirements are installed:
   ```bash
   pip install -r ../requirements.txt
   ```

4. **Path Setup**: Notebooks and scripts assume they are run from the `examples/` directory with parent directory in Python path.

## Expected Outputs

Running the examples will produce:
- Console output showing model statistics and metrics
- Visualization plots (saved to `results/` directory)
- Trained model checkpoints (saved to `results/` directory)

## Next Steps

After running the examples, you can:
1. Modify configurations to experiment with different hyperparameters
2. Use your own molecular dynamics data
3. Extend the MCMC sampling for longer simulations
4. Analyze the learned representations and latent space

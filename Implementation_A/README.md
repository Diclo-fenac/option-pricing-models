# Neural Surrogate for Option Pricing

A high-performance neural surrogate model designed to approximate Monte Carlo simulations for European Call Option pricing, achieving significant computational speedups.

## Why?
Traditional Monte Carlo (MC) simulations for option pricing are computationally expensive and slow, requiring millions of random paths to converge. This latency is a bottleneck in real-time financial applications. This project replaces the slow MC simulation with a **Neural Surrogate Model** that provides near-instantaneous pricing with high accuracy.

## What?
- **Task:** Price European Call Options using the Black-Scholes framework.
- **Data:** 1,000 parameter combinations generated using Monte Carlo simulations (1,000,000 samples per option).
- **Model:** A Deep Neural Network (3-64-64-1) with Tanh activations.
- **Parameters:** 
  - Stock Price ($S_0$): [80.0, 120.0]
  - Volatility ($\sigma$): [0.15, 0.45]
  - Time to Maturity ($T$): [0.5, 2.0]
  - Fixed: Strike ($K=100$), Risk-free rate ($r=0.05$).

## How?
1. **Data Generation:** Synthetic dataset created via high-fidelity Monte Carlo simulations.
2. **Preprocessing:** Feature scaling using standard scalers (`X_scaler.pkl`, `y_scaler.pkl`).
3. **Training:** The model was trained for 3,000 epochs using the Adam optimizer to minimize Mean Squared Error (MSE) between NN predictions and MC ground truth.
4. **Integration:** The trained model (`.pth`) is used as a drop-in replacement for the simulation during inference.

## Results
The neural surrogate achieves a massive reduction in latency while maintaining strong predictive performance:

| Metric | Value |
| :--- | :--- |
| **Speedup Factor** | **~8882.7x** |
| **Inference Time** | **~0.0009 ms** (vs 8.77s for MC) |
| **R² Score** | **0.870434** |
| **Mean Absolute Error** | **$1.5739** |
| **Avg. Relative Error** | **5.12%** |

The model demonstrates that a neural surrogate can effectively capture the non-linear dynamics of option pricing, making it suitable for high-frequency trading and large-scale risk management.

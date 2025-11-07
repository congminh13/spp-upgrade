# SPP-Upgrade  
*Upgraded version of the SPP app: comparing LSTM vs. Linear Regression*

This project is an enhanced version of the SPP app, built to explore and compare time series forecasting using two model types: a classic Linear Regression model and a deep-learning Long Short‑Term Memory (LSTM) model. It’s ideal for demonstrating your data science & ML skills and for learning how different modeling approaches behave on the same dataset.

---

## Features

- Load historical data (e.g., stock prices, other time series)  
- Pre-process data: normalization, windowing / sequence creation for LSTM  
- Train & evaluate:  
  - Linear Regression model (baseline)  
  - LSTM model (advanced)  
- Compare performance: MSE, MAE, maybe other metrics  
- Visualize results: actual vs predicted plots, residuals, training curves  
- Optional: UI or notebook interface to run experiments interactively  

---

## Tech Stack

- Python 3.7+  
- Key libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow` / `keras` (for LSTM)  
- Visualization: `matplotlib`, `seaborn`, `plotly`  
- Notebook/Script format: Jupyter Notebook(s) + Python scripts  
- (Optional) Streamlined CLI or simple UI for switching between model modes  

---

## Getting Started

### 1. Clone the repository  
```bash
git clone https://github.com/congminh13/spp-upgrade.git
cd spp-upgrade

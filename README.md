[README.md](https://github.com/user-attachments/files/24350524/README.md)
# 🛰️ Spacecraft Anomaly Detection System

A complete end-to-end Machine Learning pipeline for detecting anomalies in spacecraft telemetry data using **LSTM Autoencoders** and **SHAP** for explainability.

![Dashboard Preview](assets/dashboard_main.png)

## 📌 Project Overview
This project implements a robust anomaly detection system for the **NASA SMAP (Soil Moisture Active Passive)** dataset. It uses a **Long Short-Term Memory (LSTM) Autoencoder** to learn the normal patterns of sensor data and detects anomalies based on high reconstruction errors.

Crucially, it includes an **Explainable AI (XAI)** module using **SHAP (SHapley Additive exPlanations)**, which highlights exactly *why* a specific time step was flagged as an anomaly.

### Key Features
- **Per-Channel Training**: Handles the SMAP dataset's variable feature dimensions (e.g., 25 vs 55 features) by training 54 separate models.
- **Deep Learning Model**: 2-layer LSTM Autoencoder implemented in PyTorch.
- **Explainability**: Custom SHAP integration (explaining MSE Loss) to pinpoint contributing features.
- **Interactive Dashboard**: Streamlit app for real-time visualization and analysis.

## 🚀 Installation

Ensure you have Python 3.8+ installed.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/spacecraft-anomaly-detection.git
    cd spacecraft-anomaly-detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

    *Note: This project requires standard libraries like `torch`, `shap`, `streamlit`, `pandas`, `numpy`, and `plotly`.*

## 🏃 Usage

### 1. Launch the Dashboard (Recommended)
The easiest way to explore the system is through the pre-trained models via the dashboard.

```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 2. Retraining Models
To retrain the models on the dataset:

```bash
# Train all 54 channels
python src/train.py --epochs 5

# Train specific channel (e.g., P-1)
python src/train.py --channel P-1
```

### 3. Evaluation
To compute global metrics (Precision, Recall, F1) across the entire dataset:

```bash
python src/evaluate.py
```

## 📊 Results

The system was evaluated on the full **54-channel SMAP test set**.

| Metric | LSTM Autoencoder | EWMA Baseline |
| :--- | :--- | :--- |
| **F1 Score** | **0.4513** | 0.4331 |
| **Precision** | **0.5991** | 0.2837 |
| **Recall** | 0.3620 | **0.9145** |
| **ROC AUC** | **0.5612** | 0.4690 |

*The LSTM Autoencoder demonstrates significantly higher precision and overall F1 score compared to the simple Exponential Weighted Moving Average (EWMA) baseline.*

## 🔍 Visualizations

### Signal & Anomalies
View the raw telemetry signals with Ground Truth anomalies marked in Red.
![Signal View](assets/signal_view.png)

### Explainability (SHAP)
Understand **why** an anomaly was detected. The heatmap below shows the contribution of each feature to the reconstruction error over time.
![SHAP Explanation](assets/shap_explanation.png)

## 📁 Project Structure

```
├── assets/                 # Screenshots and images
├── data/
│   ├── raw/                # Original SMAP/MSL .npy files
│   └── processed/          # Trained models (.pth) and configs (.json)
├── src/
│   ├── data_loader.py      # Data preprocessing and windowing
│   ├── model.py            # LSTM Autoencoder class
│   ├── train.py            # Training pipeline
│   └── evaluate.py         # Evaluation and metrics calculation
├── app.py                  # Streamlit Dashboard application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Troubleshooting
- **WinError 1114 (DLL Load Failed)**: If you encounter this with PyTorch, ensure you are using a compatible version (tested with PyTorch 2.8.0 on Windows).
- **SHAP "grad can be implicitly created only for scalar outputs"**: This codebase uses a custom `MSELossModel` wrapper in `app.py` to fix this issue by explaining the scalar reconstruction loss.

---
*Built for the Spacecraft Anomaly Detection Challenge.*

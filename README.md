# **LLM-Inferbench (MLX Edition)**  
*A side project for data-driven benchmarking of Large Language Model inference on Apple Silicon (M5, 16GB).*

---

## 📌 Overview

**LLM-Inferbench** is a lightweight, fully reproducible benchmarking framework designed to measure **inference performance** of Large Language Models (LLMs) running locally on **Apple Silicon (M-series) hardware** using **Apple’s MLX framework**.

This project focuses exclusively on **inference** — *no model training or fine-tuning is performed.*  
The goal is to systematically evaluate how model size, context length, batch size, and quantization affect:

- **Latency per token (ms)**
- **Throughput (tokens/sec)**
- **Memory usage**
- **Feasible configurations on 16GB unified memory systems**

The resulting data is used for **data science analysis**, including visualization, exploratory analysis, and simple ML models that predict inference performance.

---

## 🎯 Project Goals

- Benchmark GPT-style models (e.g., GPT-2 family) using **MLX**.
- Collect structured performance metrics across multiple inference configurations.
- Evaluate performance dependencies on:
  - Model size  
  - Context length  
  - Batch size  
  - Precision / quantization  
- Generate a clean dataset for analysis.
- Build lightweight prediction models (Regression / Classification) using only the benchmark data.
- Provide insights into optimal configurations for Apple Silicon laptops with **16GB unified memory**.

---

## ⚙️ Why MLX?

MLX is Apple’s official machine learning framework, offering:

- Direct control over GPU/CPU execution  
- Native support for Apple Silicon acceleration  
- Support for GPT-style models  
- Precise measurements of inference latency and throughput  
- Low-overhead, Python-based API for reproducible benchmarking  

Compared to higher-level runtimes (e.g., Ollama), MLX exposes the **raw inference performance**, making it ideal for scientific benchmarking.

---

## 🧪 Benchmarking Methodology

Each experiment measures:

- Token-level latency (ms/token)  
- Tokens per second (throughput)  
- Peak memory usage (if available)  
- Success/failure (e.g., out-of-memory events)  

Experiments are repeated multiple times per configuration to reduce noise.

Results are logged to CSV/JSON for downstream analysis in notebooks.

---

## 📊 Data Science Components

The collected benchmark dataset will be used for:

- Exploratory Data Analysis (EDA)  
- Correlation and variance analysis (e.g., latency vs context length)  
- Visualization (latency curves, scaling trends)  
- Simple prediction models (Latency Regression / Real-time Classification)  
- Hardware-specific insights for developers on M-series chips  

---

## 🧠 Machine Learning Component

This project includes *ML models trained only on performance data*:

### 🧮 Regression  

Predict token latency using:

- Model size  
- Context length  
- Batch size  
- Precision  

### ✅ Classification  

Predict whether a configuration meets an “interactive threshold” (e.g., **< 200 ms/token**).

These models help formalize performance patterns for practical recommendation systems.

---

## 💻 Hardware Used

All measurements are taken on:

- **MacBook Pro 14” (2024)**  
- **Apple M5 chip**  
- **16GB unified memory**  
- **512GB SSD**  

This reflects a realistic, consumer-grade ML development environment.

---

## 🚀 Getting Started

### 1️⃣ Install MLX

pip install mlx

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run a Benchmark

python src/benchmark/runner.py --model gpt2 --context 128 --generate 32 --precision fp16

### 4️⃣ Explore Results

Open the included notebooks:

- [`notebooks/benchmark_analysis.ipynb`](notebooks/benchmark_analysis.ipynb)
- [`notebooks/performance_comparison.ipynb`](notebooks/performance_comparison.ipynb)

## 🧩 Models Supported

The initial implementation focuses on:

- **GPT-2 (124M)**
- **GPT-2 Medium (355M)**
- **GPT-2 Large (774M)**
- **(Optional)** LLaMA-style small models supported by MLX

---

## 📈 Example Tasks

- See how latency scales with context length  
- Measure the impact of quantization on throughput  
- Identify OOM boundaries on 16GB unified memory  
- Build a regression model to predict latency  

---

## 📝 Project Status

This is an actively developing **side project**, with emphasis on:

- Data-driven experimental design  
- Performance analytics  
- Lightweight predictive modeling  
- Reproducibility on Apple Silicon  

---

## 🤝 Contributing

PRs, suggestions, and optimizations for MLX inference or metrics are welcome.  
Feel free to open an issue or submit a pull request with improvements.


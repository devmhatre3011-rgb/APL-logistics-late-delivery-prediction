# 🚢 APL Logistics: Machine Learning-Based Late Delivery Risk Prediction Desk

An end-to-end data science framework and interactive Streamlit web application designed to transform reactive supply chain tracking into proactive, predictive delivery risk intelligence.

---

## 📌 Project Overview
Global supply chains face major financial and operational losses due to unexpected delivery delays. This project analyzes a dataset of **180,519 operational records** from **APL Logistics** to predict late delivery risks before orders leave the warehouse.

By constructing domain-specific indicators (such as fulfillment pressure and order complexity scores) and utilizing an advanced **Random Forest Classifier**, the system generates real-time delay probability scores ($0.0$ to $1.0$) and categorizes shipments into actionable risk tiers to guide operational rerouting.

---

## 🛠️ Key Features
* **Interactive Control Desk:** Live Streamlit application with custom sidebar filters for Shipping Mode, Global Market Region, and Customer Segment.
* **Dynamic Risk Threshold Optimization:** Adjustable slider allowing operations managers to fine-tune risk sensitivity based on seasonal capacity.
* **Order-Level Forensic Diagnostics:** Deep-dive inspection tab providing AI explainability and prescriptive action directives for individual shipments.
* **Macro Bottleneck Heatmaps:** Visual analytics mapping delay likelihood across global geographic corridors and shipping modes.
* **Urgent Action Rerouting Queue:** Priority dispatch desk isolating high-risk exception orders for immediate cross-docking or carrier switching.

---

## ⚙️ Custom Feature Engineering
Raw logistics data is transformed using engineered domain metrics:
1. **Shipping Pressure Index (SPI):** Measures scheduled delivery velocity relative to order item volume.
2. **Mode Risk Flag (MRF):** Assigns empirical risk weights to transport methods (Same Day vs. Standard Class).
3. **Order Complexity Score (OCS):** Quantifies total order value footprint to measure processing and customs friction.

---

## 📊 Model Evaluation Summary
The predictive engine benchmarks a **Random Forest Classifier** against a baseline Logistic Regression model:
* **ROC-AUC:** ~0.80+ (High distinction between on-time and late shipments)
* **Precision & Recall:** Balanced to prevent false delay alarms while successfully catching high-risk exceptions.

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/devmhatre3011-rgb/APL-logistics-late-delivery-prediction.git](https://github.com/devmhatre3011-rgb/APL-logistics-late-delivery-prediction.git)
cd APL-logistics-late-delivery-prediction

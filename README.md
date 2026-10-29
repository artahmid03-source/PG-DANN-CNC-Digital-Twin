<img width="3270" height="1410" alt="fig3_parity_plots" src="https://github.com/user-attachments/assets/725878b3-b0b9-4a08-af7f-44126f59f941" />
<img width="3160" height="1830" alt="fig6_iiot_digital_twin_dashboard" src="https://github.com/user-attachments/assets/e82fb53c-12bb-4c80-b057-33f04a51e3db" />
# Physics-Guided Digital Twin for CNC Tool Wear ⚙️

Bridging metal-cutting physics with deep learning for Zero-Defect Manufacturing (ZDM).

## 🚀 Project Overview
Traditional AI models fail when CNC machines switch materials (e.g., Cast Iron to Steel) because they ignore cutting physics, often resulting in impossible "self-healing" predictions. This project introduces **PG-DANN** (Physics-Guided Domain-Adversarial Neural Network), which:
* Uses a 2D Finite Element Analysis (FEA) scaling module to normalize sensor data based on cutting energy and contact stress.
* Uses thermodynamic wear laws to eliminate physical AI errors (achieving **0.0% monotonicity violations**).
* Deploys on a live Industrial IoT (IIoT) edge pipeline using MQTT, InfluxDB 3, and Grafana.

## 📊 Key Results
* **Accuracy:** R² > 0.82 across both cross-material and cross-feed transfer tasks.
* **Speed:** 0.52 ms neural inference latency / 7.43 ms end-to-end telemetry latency.
* **Sustainability:** Saves 389 cutting edges and prevents 707 kg CO₂e emissions per CNC spindle-year.

## 🖼️ System Dashboards & Performance
*(Drag and drop fig6_iiot_digital_twin_dashboard.png from your computer here!)*

*(Drag and drop fig3_parity_plots.png from your computer here!)*

## 🛠️ Tech Stack
* **Machine Learning:** PyTorch, Domain-Adversarial Neural Networks (DANN), scikit-learn
* **Physics & Engineering:** scikit-fem (2D FEA), SciPy, Johnson-Cook Mechanics
* **Industrial IoT:** MQTT (Paho), InfluxDB 3 (Apache Parquet), Grafana, Docker

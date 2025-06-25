
# 🏃‍♂️ Bias-Mitigation in Sport Talent Evaluation with GCNs

**Author:** Florian Gbaguidi  
📎 [Project Notebook](https://github.com/femi-25/Bias-mitigation-in-sport-talent-evaluation/blob/main/Player_talent_prediction_with_GNNs_Florian_GBAGUIDI.ipynb)

## 📌 Project Summary
This project tackles bias in player evaluation models used in sports analytics. Initial experiments with classical models and basic GCNs revealed biased predictions, especially favoring players from dominant nationalities. We develop a fairness-aware GCN architecture that boosts predictive performance while reducing group-level disparities.

## 🎯 Problem Statement
Traditional player‑potential models amplify demographic imbalances in the dataset, particularly nationality bias. Goal: deliver **accurate** and **fair** predictions.

## 🔧 Methods
- **Graph Construction:** PCA‑based similarity on full feature space.
- **Model:** GCN with categorical embeddings + GATv2 attention.
- **Bias Diagnosis:** Fairlearn `MetricFrame` (MAE per nationality).
- **Baseline Comparisons:** Random Forest, Linear Regression, Top‑feature GCN.

## 📈 Key Results
| Model | MAE | RMSE | Best Group | Worst Group |
|-------|-----|------|-----------|-------------|
| GCN (Top Features) | 6.93 | 8.71 | Saudi Arabia (4.36) | Brazil (9.42) |
| **GCN + PCA (Final)** | **5.35** | **6.76** | Japan (4.22) | China PR (6.52) |

![Group-wise MAE](readme_assets/group_mae.png)

## 🗺️ Model Architecture
![GCN Architecture](readme_assets/architecture.png)

## ⚠️ Challenges & Fixes
| Problem | Solution |
|---------|----------|
| GAT instability | Switched to **GATv2Conv** |
| Noisy edge weights | PCA on features |
| Group bias | Fairlearn metrics |
| Overfitting | Best‑model tracking |

## 🔄 Next Steps
- Hyperparameter sweep (dropout, LR, attention heads).  
- Cosine similarity for edge construction.  
- GNNExplainer for interpretability.  
- Adversarial debiasing experiments.

## 🧾 Conclusion
PCA‑based edges + fairness metrics produced a robust, balanced GCN outperforming baselines in both accuracy and fairness.

Multi-Paradigm Cognitive Workload Classification Pipeline

 **Author:** Delucie Rurangwa  
 **Institution:** African Leadership University (ALU)  
 **Core Domain:** Machine Learning, Deep Learning Topologies, Tabular Telemetry, Human-Computer Interaction (HCI)



  Project Overview & Engineering Goals
In high-stress operational environments, human operators are constantly bombarded with intense streams of data telemetry. When dynamic task complexity breaches an individual's psychological or cognitive thresholds, cognitive fatigue triggers sharp drops in performance and operational breakdowns. 

This project establishes a comprehensive, end-to-end data processing and modeling pipeline to classify human cognitive workload into three distinct operational bands (**Low, Medium, and High**) using high-dimensional tabular telemetry data. 

Rather than deploying an arbitrary model architecture, this repository documents a rigorous **7-stage incremental experimental progression**. It systematically evaluates the structural and performance trade-offs between classical statistical learning frameworks (Scikit-Learn) and deep neural network topologies (TensorFlow/Keras). The ultimate objective is to demonstrate how custom structural modifications—specifically multi-branch shortcut paths (skip-connections), input dropout shields, and validation-driven early stopping—can optimize non-linear representation capacity on sparse tabular arrays while ensuring absolute generalization safety.



##  Automated Preprocessing Architecture & Data Integrity
To protect against data leakage and ensure 100% algorithmic reproducibility across different computational runs, the data pipeline follows a strict functional sequence:

1. **Environmental Reproducibility:** Internal pseudo-random number generators for both NumPy and TensorFlow are explicitly anchored to a fixed seed (`42`), removing stochastic variance during dataset splits and weight initializations.
2. **Categorical Mapping:** Qualitative environmental context features are isolated and transformed via **One-Hot Encoding**. This maps string data into expanded binary vectors, ensuring distance-based and gradient-based algorithms process categorical tokens without assuming an implicit ordinal hierarchy.
3. **Numerical Standardization:** Continuous metrics with disparate measurement boundaries (e.g., raw milliseconds vs. scale ratings) are adjusted using a standardization routine to establish uniform variance ($\mu = 0, \sigma = 1$). This step creates equal scale priority across the input space and stabilizes gradient steps.
4. **Data Streaming Optimization:** The processed arrays are passed directly into the **`tf.data` API pipeline**, applying automated batching and prefetching algorithms to optimize low-memory execution and training throughput.
5. **Deterministic Matrix Splitting:** The data pool is partitioned into a clean **80% Training subset** for optimization and a completely isolated **20% Testing subset** to validate true model generalization.



##  The 7-Experiment Architecture & Progression Rationale

The optimization pipeline systematically iterates through the following structural configurations to evaluate core learning theories:

**Experiment 1: Logistic Regression Baseline** * *Configuration:* L2 Regularization, Inverse Strength $C=1.0$, Max Iterations $= 1000$.  
  * *Rationale:* Establishes a foundational performance baseline for linear hyper-plane separation.
* **Experiment 2: Random Forest Ensemble** * *Configuration:* 100 Estimators, Unbounded Depth (`None`), Gini Impurity, Balanced Class Weights.  
  * *Rationale:* Tests non-linear ensemble behavior and orthogonal splitting logic on tabular attributes.
* **Experiment 3: Sequential Neural Network Baseline** * *Configuration:* Keras Sequential API; 3-Layer Stack: `Dense(64, ReLU)` $\rightarrow$ `Dense(32, ReLU)` $\rightarrow$ `Dense(3, Softmax)`; Learning Rate $= 0.001$.  
  * *Rationale:* Translates tabular array inputs into a baseline, rigidly serialized deep multi-layer perceptron.
* **Experiment 4: Functional Neural Network (Skip-Connections)** * *Configuration:* Keras Functional API; Multi-branch topology featuring parallel feature streaming and a direct low-level shortcut path. Learning Rate $= 0.001$.  
  * *Rationale:* Tests whether multi-branch shortcut paths preserve baseline signal representation and mitigate deep gradient decay.
* **Experiment 5: Tuned Learning Rate Network** * *Configuration:* Functional Topology; Learning Rate scaled down 10-fold to **$0.0001$**.  
  * *Rationale:* Maps model sensitivity to gradient optimization speed and tracks convergence bottleneck boundaries.
* **Experiment 6: Dropout Regularized Network** * *Configuration:* Functional Topology; Learning Rate $= 0.001$; **30% Hidden Layer Dropout**.  
  * *Rationale:* Maps over-regularization limits and measures capacity degradation under sparse activation pathways.
* **Experiment 7: Master Apex Model** * *Configuration:* Functional Topology; Learning Rate $= 0.001$; **20% Input-Only Dropout Shield** + Automated Validation-Loss **EarlyStopping** (patience bounds activated).  
  * *Rationale:* Blends structural regularization and dynamic training termination to reach optimal bias-variance equilibrium.

---

##  Consolidated Performance Log & Key Insights

### Master Experimentation Log
| Exp | Model Paradigm | Core Hyperparameters & Structural Configuration | Test Accuracy | Iterative Insight & Progression Rationale |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Logistic Regression | Regularization: L2, Inverse Strength $C=1.0$, Max Iterations $= 1000$ | **100.00%** | Preprocessing pipeline successfully establishes perfectly distinct linear feature spaces under static parameters. |
| **2** | Random Forest | Estimators $= 100$, Max Depth $=$ None, Gini Impurity, Balanced Weights | **66.67%** | Enters architectural collapse; orthogonal decision splits struggle heavily with high-dimensional sparse hot-encoded arrays. |
| **3** | Sequential Neural Net | 3-Layer Stack: Dense(64, ReLu) $\rightarrow$ Dense(32, ReLu) $\rightarrow$ Softmax; LR $= 0.001$ | **79.17%** | Confirms neural layers manage sparse tabular input rows more effectively than orthogonal trees, though serialization limits capacity. |
| **4** | Functional Neural Net | Concatenated Multi-Branch Skip-Connections; Learning Rate $= 0.001$ | **95.83%** | Major performance leap; multi-branch shortcut paths preserve early raw signal telemetry and stabilize deep weight updates. |
| **5** | Tuned Learning Rate NN | Functional Topology; Learning Rate scaled down to **0.0001** | **70.83%** | Optimization bottleneck occurs; excessively small step sizes fail to escape local minima within fixed epoch thresholds. |
| **6** | Dropout Regularized NN | Functional Topology; Learning Rate $= 0.001$; **30% Layer Dropout** | **83.33%** | Over-regularization occurs; blinding a high ratio of hidden node activations restricts model capacity given finite sample density. |
| **7** | Master Apex Model | Functional Topology; **20% Input Dropout** + **Val-Loss Early Stopping** | **95.83%** | Reaches optimal balance; structural stabilization shields inputs while dynamic stopping prevents late-epoch validation decay. |

### Core Theoretical Takeaways
1. **The Tree-Sparsity Crisis:** The dramatic drop in Random Forest performance (66.67%) highlights a classic data science limitation: while tree ensembles handle dense tabular arrays exceptionally well, they struggle with high-dimensional, sparse binary fields generated by one-hot encoding. Because tree boundaries split orthogonally, they get isolated in empty spaces within sparse matrices, leading to structural misclassifications.
2. **The Functional Skip-Connection Advantage:** Shifting from a rigid Sequential stack to a multi-branch Functional API architecture drove accuracy up to 95.83%. By integrating a shortcut path that passes baseline input signals directly past the intermediate activation blocks, the network retains early representation matrices. This prevents gradient degradation across deep training steps and allows the model to map tabular features effectively.

---

##  Master Apex Model: Graphical Architecture Topology
The structural assembly of the final generalized network layout follows this data flow logic:

```text
                  [ Shared Continuous & Categorical Telemetry Input ]
                                           |
                                           v
                        [ 20% Input Spatial Dropout Shield ]
                                           |
                +--------------------------+--------------------------+
                |                          |                          |
                v                          v                          v
      [ Branch A: Dense Block ]  [ Branch B: Residual Block ]  [ Branch C: Skip Path ]
         Dense Layer (64, ReLU)     Dense Layer (32, ReLU)        Direct Vector Pass
                   |                          |                          |
            Batch Normalization       Residual Skip Add                  |
                |                          |                          |
                +--------------------------+--------------------------+
                                           |
                                           v
                            [ Vector Concatenation Layer ]
                                           |
                                           v
                              [ Output Layer: Softmax ]
                                           |
                                           v
                    [ Final Multiclass Workload Prediction Out ]

```

---

##  Environment Setup & Local Reproduction Guide

### System Dependencies

To run the training notebook and verify metrics without runtime errors, ensure your Python environment (Python 3.8+) has the following explicit package allocations installed:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn jupyter

```

### Execution Steps

1. Clone this repository locally:
```bash
git clone <your-repository-url-placeholder>
cd <repository-folder-name>

```


2. Verify that your dataset file (`telemetry_workload_dataset.csv`) resides within the same directory as the master notebook.
3. Launch your local Jupyter environment or upload the files directly to Google Colab:
```bash
jupyter notebook

```


4. Open `Model_Training_and_Evaluation.ipynb` and execute the runtime cells sequentially from top to bottom (**Runtime > Restart and run all**).

---

##  Repository Contents

* `Model_Training_and_Evaluation.ipynb` - Completed master Jupyter Notebook containing all 7 experiment cell configurations, dependencies, and analytical printout blocks.
* `telemetry_workload_dataset.csv` - Preprocessed, high-dimensional tabular matrix containing user workload tracking metrics.
* `README.md` - Technical engineering pipeline documentation and reproduction blueprints.

```

```

- NOTE: Datafiles/datasets used is too large to upload, it can be found in the google drive here: https://drive.google.com/drive/folders/1frjG76-22-k_iUcQd7Qus8fHQldMwEBd?usp=sharing
-----

# Interpretable Surrogate Modeling for TCGA RNA-Seq Cancer Type Classification

This repository contains the code and resources for the project "Interpretable Surrogate Modeling for TCGA RNA-Seq Cancer Type Classification," a class project for CSS 490/590 at the University of Washington, Bothell.

**Authors:** Subhash Saravanan*, Manish Ram*

" * " = Equal Contributions 

-----

## 📋 Project Overview

This project tackles the critical challenge of balancing predictive accuracy and model interpretability in clinical genomics. While deep learning models, like Convolutional Neural Networks (CNNs), achieve high accuracy in classifying cancer types from RNA-Seq data, their "black box" nature limits trust and clinical adoption.

Our solution is a hybrid system that:

1.  **Develops a high-performance 1D-CNN** to classify 32 cancer types from The Cancer Genome Atlas (TCGA), achieving **95.4% test accuracy**.
2.  **Employs knowledge distillation** to transfer the predictive power of the complex CNN (the "teacher") to a simple, interpretable Soft Decision Tree or SDT (the "student").
3.  **Extracts human-readable rules**, or a "Decision Fingerprint," from the trained SDT, providing a transparent rationale for each prediction.

The resulting SDT surrogate model achieves **87.5% accuracy** and **88.6% fidelity** to the teacher model, demonstrating that it's possible to make powerful deep learning models transparent without a significant loss in performance.

-----

## ⚙️ Methodology

The project follows a systematic pipeline, from data acquisition to the final extraction of interpretable rules.

1.  **Data Foundation**:

      * **Acquisition**: Publicly available RNA-Seq and clinical data were sourced from the TCGA Pan-Cancer cohort via the UCSC Xena platform.
      * **Preprocessing**: The raw dataset, containing 20,531 genes, was cleaned, integrated, and filtered to a final feature space of 12,854 informative genes. Genes with low variance were removed, and the data was standardized.

2.  **White-Box Model Analysis**:

      * Baseline models like **Logistic Regression**, **Decision Trees**, and **Random Forest** were trained to establish performance benchmarks.
      * **SHAP (SHapley Additive exPlanations)** was used to analyze feature importance and uncover initial data biases, such as the model leveraging gender-specific genes.

3.  **Black-Box CNN Development**:

      * A **1D-CNN** was developed as the primary "teacher" model. This architecture was chosen for its excellent balance of high performance (95.02% 5-fold CV accuracy) and computational efficiency (\~1 MB model size).

4.  **Knowledge Distillation**:

      * The trained 1D-CNN generated probabilistic outputs ("soft labels") for the training data.
      * An **interpretable Soft Decision Tree (SDT)** was trained to mimic the CNN by using these soft labels as its target, effectively learning the teacher's complex decision boundaries.

5.  **Surrogate Model Evaluation & Rule Extraction**:

      * The SDT was evaluated on its **fidelity** (how well it mimics the CNN) and its **accuracy** (how well it predicts the true labels).
      * The final trained SDT was deconstructed to produce a "Decision Fingerprint" for each cancer type—a clear, step-by-step rule path.

-----

## 📊 Key Results

### Model Performance

| Model | Accuracy | Balanced Accuracy | Macro F1-Score | Fidelity to CNN |
| :--- | :--- | :--- | :--- | :--- |
| **1D-CNN (Teacher)** | 95.4% | 90.0% | 94.7% | - |
| **Soft Decision Tree (Student)** | 87.5% | - | - | 88.6% |
| **Logistic Regression** | 96.5% | 93.7% | 94.2% | - |
| **Standard Decision Tree (Depth 5)**| 45.2% | 20.8% | 18.4% | - |

The knowledge distillation process created a model (SDT) that is vastly superior to a standard decision tree of the same complexity (87.5% vs 45% accuracy) and retains much of the predictive power of its CNN teacher.

### Interpretable "Decision Fingerprint"

The final output is a transparent, rule-based path for classification. For example, the path to classify a sample as **Breast Invasive Carcinoma** follows a series of checkpoints, each supported or opposed by specific gene expressions.

  * **Checkpoint 1 (Node 0):** Decision **LEFT** supported by `CALCB` and `SFTPA2`.
  * **Checkpoint 2 (Node 1):** Decision **RIGHT** supported by `PTPN20B` and `HCG11`.
  * **Checkpoint 3 (Node 4):** Decision **RIGHT** supported by `LOC149837`.
  * **Checkpoint 4 (Node 10):** Decision **LEFT** supported by a specific gene signature.
  * **Checkpoint 5 (Node 21):** Decision **LEFT** supported by `S100A1` and `RRH`.
  * **Conclusion:** Classified as Breast Invasive Carcinoma.

This provides a clear, verifiable rationale that can be reviewed and trusted by clinicians.

-----

## 🚀 Getting Started

### Prerequisites

This project is built using Python. Ensure you have the following libraries installed:

  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `tensorflow` / `keras`
  * `shap`
  * `matplotlib`
  * `seaborn`

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Data**: Download the TCGA Pan-Cancer RNA-Seq data from the [UCSC Xena platform](https://xenabrowser.net/datapages/). Place the data files in the `data/` directory.
2.  **Run Notebooks**: Follow the Jupyter notebooks in numerical order to replicate the analysis:
      * `01_Data_Preprocessing.ipynb`: Loads and cleans the data.
      * `02_White_Box_Models.ipynb`: Trains and evaluates baseline models.
      * `03_CNN_Teacher_Model.ipynb`: Trains and evaluates the 1D-CNN.
      * `04_Knowledge_Distillation.ipynb`: Trains the SDT surrogate model.
      * `05_Rule_Extraction.ipynb`: Visualizes the SDT and extracts Decision Fingerprints.




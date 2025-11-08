
# Startup Failure Prediction

## Project Summary

**Goal:** Predict if a startup will fail after its last funding round.  
**Model:** Logistic Regression.  
**Dataset:** startups (filtered), features include funding, market, country, state, and funding type flags. 
**Documentation:** The complete information, with all steps are in the document called `Project-Document.md`.

**Key Results (Test Set):**  
- **Accuracy:** 0.72  
- **ROC-AUC:** 0.749  
- **Best F1 threshold:** 0.43 (used 0.5 for final predictions)

## Model Evaluation
### **1. ROC-Curve:**  
<p align="center">
  <img src="images/roc_curve.png" width="300">
</p>


The ROC curve shows how well the model distinguishes between active and failed startups. With an AUC of 0.749, the model demonstrates good discriminatory power. A perfect model would reach the top-left corner.

### **2. Predicted Probability Histogram:**  
<p align="center">
  <img src="images/pred_probs.png" width="300">
</p>


The histogram shows the distribution of predicted probabilities for both classes. Ideally, we expect two clearly separated peaks, one near 0 (active startups) and another near 1 (failed startups).

In this model, the peaks are distinguishable but overlap slightly, suggesting that while most predictions are accurate, there remains some uncertainty in borderline cases.

### **3. Precision-Recall Curve:**  
<p align="center">
  <img src="images/precision_recall_curve.png" width="300">
</p>

The Precision-Recall curve illustrates how the model balances identifying failed startups (recall) while keeping predictions accurate (precision).

The curve starts with very high precision for a small subset of predictions, then gradually declines as recall increases, stabilizing around moderate precision at full recall.

This behavior indicates that the model is particularly confident in detecting clear failure cases, while uncertainty increases for borderline startups.

### **4. Calibration Curve:**  
<p align="center">
  <img src="images/calibration_curve.png" width="300">
</p>


The calibration curve compares the model’s predicted probabilities with the actual observed outcomes.

We see that the curve stays slightly below the diagonal, meaning the model tends to be under-confident, it predicts slightly lower probabilities than the true likelihood of startup failure.
Still, the curve follows the ideal line fairly closely, indicating that the model’s probability estimates are generally well-calibrated.

### **5. F1-Score Otimization:**  
<p align="center">
  <img src="images/f1_score_vs_threshold.png" width="300">
</p>


This plot illustrates how the F1 score changes as we vary the classification threshold.

We observe a clear peak around the optimal threshold, where the balance between precision and recall is maximized.

The red dashed line marks this best point (~0.43), but for practical consistency across models, we keep the threshold at 0.5, accepting a slightly lower F1 in exchange for a more interpretable and standardized decision boundary.

### **6. Confusion Matrix:**  
<p align="center">
  <img src="images/confusion_matrix.png" width="300">
</p>


We can see that the model correctly identifies a majority of both active (360/548) and failed (313/426) startups.

There are some misclassifications, with slightly more active startups incorrectly labeled as failed (188) than failed startups mislabeled as active (113).

Overall, the model demonstrates a reasonable balance between sensitivity (recall for failed startups) and specificity (recall for active startups).

#### **Takeaways:**  
- Higher total funding, more funding rounds, and certain markets/states are associated with lower failure risk.  
- Logistic regression provides a stable, interpretable model with reasonable predictive performance.

## Using the Model

### 1. Requirements
Before running the project, make sure you have:
- Python 3.12+
- Pipenv

### 2. Local Installation

Clone this repository.

### 3. Build Docker image

```
docker build -t startup-predict .
```

### 4. Run the container

```
docker run -p 9696:9696 startup-predict
```
- The Flask API will start automatically inside the container.
- Available at http://localhost:9696/predict.

### 5. Testing the model
With the container created and running, you can open another terminal to test the API.

In this project, we already have a sample script called `test-predict.py` that sends a request to the model endpoint using example data.

You can simply run it with:

```
python test-predict.py
```

### 6. Video Tutorial
You can see a tutorial in this [video](https://youtu.be/CmCtuXhqBwg).
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MNIST Multi-Class Classification Project</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.7;
            margin: 40px;
            background-color: #f9f9f9;
            color: #222;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        code {
            background: #eef;
            padding: 3px 6px;
            border-radius: 4px;
        }
        .box {
            background: #ffffff;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        ul {
            margin-left: 20px;
        }
        .diagram {
            background: #f0f0f0;
            padding: 15px;
            font-family: monospace;
            white-space: pre;
            overflow-x: auto;
            border-radius: 6px;
        }
    </style>
</head>

<body>

<h1>MNIST Digit Classification using Machine Learning</h1>

<p>
This project implements a complete <b>end-to-end machine learning pipeline</b> on the
<b>MNIST handwritten digit dataset</b>.  
Multiple classification models are trained, evaluated, compared using
<b>ROC–AUC curves</b>, and the best-performing model is further optimized using
<b>GridSearchCV</b>.
</p>

<hr>

<h2>📌 Project Workflow Overview</h2>

<div class="diagram">
Raw MNIST Data
      ↓
Null Value Check
      ↓
Train-Test Split
      ↓
Label Binarization
      ↓
Model Training
      ↓
Predictions & Probabilities
      ↓
Evaluation Metrics
      ↓
ROC–AUC Curve Comparison
      ↓
Best Model Selection (SVM)
      ↓
Hyperparameter Tuning (GridSearchCV)
      ↓
Final Evaluation
      ↓
Model Serialization (Pickle)
</div>

<hr>

<h2>📊 Dataset Description</h2>

<ul>
    <li>Dataset: <b>MNIST Handwritten Digits</b></li>
    <li>Classes: Digits from <b>0 to 9</b></li>
    <li>Each image is converted into numerical pixel values</li>
</ul>

<hr>

<h2>🧹 Data Preprocessing</h2>

<div class="box">
<ul>
    <li>Checked for <b>null values</b> in the dataset</li>
    <li>Split the dataset into <b>training</b> and <b>testing</b> sets using <code>train_test_split</code></li>
    <li>Applied <b>label binarization</b> on test labels using <code>label_binarize</code></li>
</ul>
</div>

<p>
Label binarization converts multi-class labels into binary format, which is essential for:
</p>

<ul>
    <li>ROC curve plotting</li>
    <li>ROC–AUC score calculation</li>
</ul>

<hr>

<h2>🤖 Machine Learning Models Used</h2>

<div class="box">
<ul>
    <li>K-Nearest Neighbors (KNN)</li>
    <li>Naive Bayes</li>
    <li>Logistic Regression</li>
    <li>Decision Tree</li>
    <li>Random Forest</li>
    <li>AdaBoost</li>
    <li>Gradient Boosting</li>
    <li>XGBoost</li>
    <li>Support Vector Machine (SVM)</li>
</ul>
</div>

<p>
Each model was implemented using <b>classes and objects</b> to ensure modular, reusable,
and clean code architecture.
</p>

<hr>

<h2>📈 Model Training & Prediction</h2>

<ul>
    <li>Each model was trained on the training dataset</li>
    <li>Predictions were made on both training and testing data</li>
    <li>Performance was evaluated using:</li>
</ul>

<ul>
    <li><code>accuracy_score</code></li>
    <li><code>classification_report</code></li>
    <li><code>confusion_matrix</code></li>
</ul>

<hr>

<h2>📉 ROC–AUC Curve Explanation</h2>

<div class="box">
<p>
For ROC–AUC calculation:
</p>
<ul>
    <li>Probability predictions were obtained using <code>predict_proba()</code></li>
    <li>The true labels and predicted probabilities were flattened using <code>ravel()</code></li>
</ul>

<p>
Flattening converts 2D arrays into a 1D format, which helps in:
</p>

<ul>
    <li>Calculating False Positive Rate (FPR)</li>
    <li>Calculating True Positive Rate (TPR)</li>
    <li>Accurate AUC score computation</li>
</ul>
</div>

<div class="diagram">
Predicted Probabilities
        ↓
Flatten using ravel()
        ↓
Compare with True Labels
        ↓
Calculate FPR & TPR
        ↓
Plot ROC Curve
        ↓
Compute AUC
</div>

<p>
ROC–AUC curves were plotted for <b>all models</b> to compare their performance visually.
</p>

<hr>

<h2>🏆 Best Model Selection</h2>

<p>
Based on ROC–AUC score and overall performance:
</p>

<div class="box">
<h3>✅ Support Vector Machine (SVM)</h3>
<ul>
    <li>Achieved the highest ROC–AUC score</li>
    <li>Showed strong generalization on test data</li>
</ul>
</div>

<hr>

<h2>⚙️ Hyperparameter Tuning</h2>

<p>
The SVM model was further optimized using <b>GridSearchCV</b>:
</p>

<ul>
    <li>Automatically searched for best hyperparameters</li>
    <li>Used cross-validation for robust performance</li>
</ul>

<p>
After tuning, the following were evaluated again:
</p>

<ul>
    <li>Training accuracy</li>
    <li>Training classification report</li>
    <li>Training confusion matrix</li>
    <li>Test accuracy</li>
    <li>Test classification report</li>
    <li>Test confusion matrix</li>
</ul>

<hr>

<h2>💾 Model Saving</h2>

<div class="box">
<p>
The final optimized SVM model was saved using <b>Pickle</b>, allowing the model to be
loaded later without retraining.
</p>
</div>

<hr>

<h2>🛠️ Libraries Used</h2>

<ul>
    <li>NumPy</li>
    <li>Pandas</li>
    <li>Matplotlib</li>
    <li>Scikit-learn</li>
    <li>Scikit-learn Classification Models</li>
    <li>Accuracy Score</li>
    <li>Classification Report</li>
    <li>Confusion Matrix</li>
</ul>

<hr>

<h2>📌 Conclusion</h2>

<p>
This project demonstrates a <b>complete machine learning workflow</b> —
from preprocessing and model comparison to ROC–AUC evaluation, hyperparameter tuning,
and model deployment readiness.
</p>

<p>
The modular design using classes and objects makes the project scalable, readable,
and suitable for real-world machine learning applications.
</p>

</body>
</html>

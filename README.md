# Biomedical Triage System: classifying breast tissue samples

Coursework project for the EMS506U machine learning module at Queen Mary University of London (marked 100%). I trained a small PyTorch neural network to classify breast tissue samples as benign or malignant, and looked at what happens to the number of missed cancers when the decision threshold is lowered.

In triage, a missed cancer costs far more than a false alarm, so the question is how many extra false alarms it takes to catch more cancers.

## Data

`BiomedicalEngineering_Dataset.csv` is provided by the module and is not included in this repository. It has 2,000 samples with 10 numeric features (including radius, perimeter, area and concave points): 1,208 benign and 792 malignant. Upload it to the Colab file area before running the notebook.

## Method

* Stratified 80/20 split (seed 2004): 1,600 training and 400 test samples (242 benign, 158 malignant).
* `StandardScaler` fitted on the training data only, to avoid test-set leakage.
* Baseline: scikit-learn logistic regression at the default 0.5 threshold.
* Network: 10-16-8-1 feedforward network (ReLU hidden layers, sigmoid output), binary cross-entropy, Adam (learning rate 0.01), 150 full-batch epochs. 320 of the training samples are held out to monitor validation loss.
* Threshold experiment: the network's decision threshold lowered from 0.5 to 0.3.
* Depth experiment: 1 to 5 hidden layers of width 16.

## Results (test set, n = 400)

| | Logistic regression (0.5) | Network (0.5) | Network (0.3) |
| :--- | :---: | :---: | :---: |
| True negatives | 228 | 219 | 204 |
| False positives | 14 | 23 | 38 |
| **Missed cancers (false negatives)** | **37** | **30** | **23** |
| True positives | 121 | 128 | 135 |
| Malignant recall | 76.6% | 81.0% | 85.4% |
| Malignant precision | 89.6% | 84.8% | 78.0% |
| Overall accuracy | 87.3% | 86.8% | 84.8% |
| ROC AUC | 0.90 | 0.90 | 0.90 (same model) |

* Missed cancers fall from 37 to 30 to 23. The 38% reduction (37 to 23) is against the logistic regression at its default threshold. Against the network at 0.5 it is 23% (30 to 23).
* The cost is more false alarms: 14, then 23, then 38.
* Both models have an AUC of 0.90, so the network does not rank cases better than the logistic regression. Of the 14 fewer misses, 7 come from the network at the default threshold and 7 from moving the threshold to 0.3.

## Limitations

* One stratified split and one seed, with 158 malignant cases in the test set. A difference of a few cases is not statistically established.
* The 0.3 threshold was set by hand and evaluated on the test set. It was not chosen on validation data.
* The logistic regression was only evaluated at its default threshold. A fairer comparison would tune both thresholds on validation data.
* Probability calibration was not assessed, and the threshold was not derived from the cost of each error.

## Follow-up

My later project, [TrustTriage](https://github.com/AdamDebs/trusttriage-medmnist), addresses these limits on two open medical-image datasets: calibrated probabilities (Platt scaling), thresholds chosen on validation data from a stated cost, and monitoring for dataset shift.

## Files

* `Biomedical_Triage_System.ipynb`: the notebook. The written commentary was corrected in November 2026 to match the printed outputs (see the last cell). The code and outputs are unchanged.
* Python, pandas, NumPy, scikit-learn, PyTorch, Matplotlib and seaborn.

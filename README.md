# DataMiningHW2
11 models for Polish+companies+bankruptcy+data

### Table 1 Final Results of all-5-years data for Comparative Models with unbalanced training data

| ModelName | Accuracy  | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB |	0.897 |	0.627 |	0.046 |	0.542 |	0.011 |
| GaussianNB |	0.863 |	0.66 |	0.049 |	0.531 |	0.012  |
| SVM |	0.976 |	0.5 |	0.024 |	0.494 |	1.470 |
| DecisionTree |	0.965	 |0.657 |	0.114 |	0.647 |	0.750  |
| SGD |	0.876 |	0.642 |	0.046 |	0.533 |	0.042 |
| Nearest_Neighbors |	0.723 |	0.675 |	0.042 |	0.467 |	0.007  |
| AdaBoost |	0.978 |	0.582 |	0.131 |	0.628 |	7.771 |
| GradientBoosting |	0.978 |	0.588 |	0.145 |	0.636 |	6.518 |
| **HistGradientBoosting** |	**0.982** |	**0.64** |	**0.262** |	**0.708** |	**9.174** |
| MLP |	0.977 |	0.623 |	0.158 |	0.667 |	113.919 |


### Table 1year.arff with unbalanced training data
| ModelName | Accuracy  | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.954 |0.010 | 0.505 | 0.050 | 0.499 | 0.005 |
| GaussianNB | 0.739 |0.594 | 0.670 | 0.078 | 0.508 | 0.005 |
| SVM | 0.954 |0.000 | 0.500 | 0.046 | 0.488 | 0.400 |
| DecisionTree | 0.959 |0.531 | 0.755 | 0.313 | 0.759 | 0.245 |
| SGD | 0.955 |0.010 | 0.505 | 0.055 | 0.499 | 0.027 |
| NearestCentroid | 0.685 |0.656 | 0.671 | 0.075 | 0.483 | 0.004 |
| AdaBoost | 0.962 |0.271 | 0.633 | 0.229 | 0.687 | 2.649 |
| GradientBoosting | 0.968 |0.385 | 0.691 | 0.345 | 0.754 | 2.078 |
| HistGradientBoosting | 0.974 |0.490 | 0.744 | 0.466 | 0.811 | 120.715 |
| MLP | 0.953 |0.312 | 0.648 | 0.178 | 0.675 | 42.656 |
| LogisticRegression(Ours) | 0.710 |0.143 | 0.430 | 0.011 | 0.421 | 1.581 |


### Table 2year.arff with unbalanced training data
| ModelName | Accuracy  | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.961 |0.000 | 0.499 | 0.038 | 0.490 | 0.007 |
| GaussianNB | 0.827 |0.243 | 0.547 | 0.043 | 0.500 | 0.007 |
| SVM | 0.962 |0.000 | 0.500 | 0.038 | 0.490 | 1.047 |
| DecisionTree | 0.954 |0.409 | 0.692 | 0.182 | 0.688 | 0.367 |
| SGD | 0.962 |0.000 | 0.500 | 0.038 | 0.490 | 0.040 |
| NearestCentroid | 0.644 |0.626 | 0.635 | 0.054 | 0.447 | 0.006 |
| AdaBoost | 0.963 |0.070 | 0.534 | 0.078 | 0.553 | 3.831 |
| GradientBoosting | 0.961 |0.096 | 0.545 | 0.075 | 0.568 | 3.053 |
| HistGradientBoosting | 0.978 |0.435 | 0.717 | 0.431 | 0.792 | 49.879 |
| MLP | 0.950 |0.261 | 0.619 | 0.109 | 0.629 | 59.709 |
| LogisticRegression(Ours) | 0.787 |0.231 | 0.514 | 0.018 | 0.458 | 1.690 |


### Table 3year.arff with unbalanced training data
| ModelName | Accuracy  | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.957 |0.000 | 0.500 | 0.043 | 0.489 | 0.007 |
| GaussianNB | 0.831 |0.336 | 0.594 | 0.059 | 0.525 | 0.015 |
| SVM | 0.957 |0.000 | 0.500 | 0.043 | 0.489 | 1.276 |
| DecisionTree | 0.930 |0.299 | 0.628 | 0.101 | 0.614 | 0.438 |
| SGD | 0.957 |0.000 | 0.500 | 0.043 | 0.489 | 0.046 |
| NearestCentroid | 0.697 |0.522 | 0.614 | 0.058 | 0.472 | 0.005 |
| AdaBoost | 0.956 |0.112 | 0.552 | 0.084 | 0.577 | 4.065 |
| GradientBoosting | 0.959 |0.134 | 0.565 | 0.117 | 0.599 | 3.281 |
| HistGradientBoosting | 0.968 |0.313 | 0.655 | 0.292 | 0.720 | 101.454 |
| MLP | 0.948 |0.224 | 0.602 | 0.108 | 0.620 | 60.222 |
| LogisticRegression(Ours) |  0.872 |0.182 | 0.536 | 0.028 | 0.499 | 1.868 |


### Table 4year.arff with unbalanced training data
| ModelName | Accuracy  | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.948 |0.007 | 0.503 | 0.058 | 0.493 | 0.007 |
| GaussianNB | 0.864 |0.464 | 0.675 | 0.112 | 0.593 | 0.013 |
| SVM | 0.948 |0.000 | 0.500 | 0.052 | 0.487 | 0.956 |
| DecisionTree | 0.948 |0.516 | 0.744 | 0.283 | 0.740 | 0.451 |
| SGD | 0.948 |0.007 | 0.503 | 0.055 | 0.493 | 0.039 |
| NearestCentroid | 0.718 |0.654 | 0.688 | 0.093 | 0.512 | 0.005 |
| AdaBoost | 0.960 |0.307 | 0.652 | 0.285 | 0.712 | 3.762 |
| GradientBoosting | 0.948 |0.000 | 0.500 | 0.052 | 0.487 | 2.987 |
| HistGradientBoosting | 0.969 |0.444 | 0.721 | 0.437 | 0.792 | 96.038 |
| MLP | 0.946 |0.340 | 0.660 | 0.195 | 0.684 | 56.011 |
| LogisticRegression(Ours) | 0.924 |0.179 | 0.564 | 0.046 | 0.547 | 1.787 |


### Table 5year.arff with unbalanced training data
| ModelName | Accuracy  | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.932 |0.033 | 0.516 | 0.085 | 0.514 | 0.004 |
| GaussianNB | 0.884 |0.554 | 0.731 | 0.200 | 0.665 | 0.004 |
| SVM | 0.941 |0.157 | 0.578 | 0.193 | 0.617 | 0.347 |
| DecisionTree | 0.937 |0.595 | 0.779 | 0.347 | 0.765 | 0.183 |
| SGD | 0.935 |0.050 | 0.525 | 0.114 | 0.530 | 0.024 |
| NearestCentroid | 0.775 |0.736 | 0.757 | 0.162 | 0.587 | 0.007 |
| AdaBoost | 0.946 |0.430 | 0.707 | 0.322 | 0.746 | 2.211 |
| GradientBoosting | 0.951 |0.455 | 0.721 | 0.371 | 0.768 | 1.697 |
| HistGradientBoosting | 0.962 |0.537 | 0.765 | 0.485 | 0.818 | 110.560 |
| MLP | 0.933 |0.537 | 0.750 | 0.307 | 0.744 | 36.765 |
| LogisticRegression(Ours) | 0.925 |0.421 | 0.682 | 0.109 | 0.621 | 1.026 |



### Table 1year.arff with balanced training data via SMOTE
| ModelName | Precision | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.833 |0.292 | 0.575 | 0.058 | 0.522 | 0.008 |
| GaussianNB | 0.506 |0.771 | 0.632 | 0.063 | 0.390 | 0.013 |
| SVM | 0.812 |0.635 | 0.728 | 0.108 | 0.564 | 5.137 |
| DecisionTree | 0.923 |0.500 | 0.722 | 0.171 | 0.666 | 0.525 |
| SGD | 0.903 |0.438 | 0.681 | 0.121 | 0.620 | 0.073 |
| NearestCentroid | 0.685 |0.635 | 0.661 | 0.073 | 0.481 | 0.005 |
| AdaBoost | 0.903 |0.542 | 0.731 | 0.154 | 0.643 | 5.270 |
| GradientBoosting | 0.905 |0.542 | 0.732 | 0.156 | 0.646 | 4.317 |
| HistGradientBoosting | 0.974 |0.510 | 0.753 | 0.461 | 0.813 | 25.384 |
| MLP | 0.900 |0.562 | 0.740 | 0.157 | 0.643 | 80.322 |
| LogisticRegression(Ours) | 0.668 |0.286 | 0.479 | 0.012 | 0.410 | 20.871 |


### Table 2year.arff with balanced training data via SMOTE
| ModelName | Precision | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.780 |0.365 | 0.581 | 0.048 | 0.493 | 0.009 |
| GaussianNB | 0.595 |0.626 | 0.610 | 0.050 | 0.421 | 0.019 |
| SVM | 0.758 |0.609 | 0.686 | 0.070 | 0.509 | 11.095 |
| DecisionTree | 0.910 |0.435 | 0.682 | 0.105 | 0.609 | 0.694 |
| SGD | 0.796 |0.487 | 0.647 | 0.063 | 0.518 | 0.118 |
| NearestCentroid | 0.640 |0.617 | 0.629 | 0.053 | 0.444 | 0.007 |
| AdaBoost | 0.862 |0.470 | 0.673 | 0.081 | 0.564 | 7.616 |
| GradientBoosting | 0.865 |0.496 | 0.687 | 0.088 | 0.571 | 6.333 |
| HistGradientBoosting | 0.973 |0.400 | 0.698 | 0.334 | 0.757 | 31.196 |
| MLP | 0.885 |0.522 | 0.710 | 0.106 | 0.596 | 104.437 |
| LogisticRegression(Ours) | 0.643 |0.692 | 0.667 | 0.029 | 0.422 | 19.106 |


### Table 3year.arff with balanced training data via SMOTE
| ModelName | Precision | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.777 |0.388 | 0.591 | 0.056 | 0.500 | 0.011 |
| GaussianNB | 0.688 |0.590 | 0.641 | 0.064 | 0.474 | 0.023 |
| SVM | 0.778 |0.567 | 0.677 | 0.079 | 0.525 | 11.070 |
| DecisionTree | 0.892 |0.403 | 0.659 | 0.095 | 0.592 | 0.804 |
| SGD | 0.890 |0.455 | 0.682 | 0.106 | 0.600 | 0.113 |
| NearestCentroid | 0.694 |0.522 | 0.612 | 0.058 | 0.471 | 0.004 |
| AdaBoost | 0.849 |0.470 | 0.668 | 0.086 | 0.563 | 8.036 |
| GradientBoosting | 0.859 |0.440 | 0.659 | 0.084 | 0.566 | 6.780 |
| HistGradientBoosting | 0.963 |0.410 | 0.699 | 0.268 | 0.733 | 24.232 |
| MLP | 0.879 |0.388 | 0.644 | 0.084 | 0.574 | 106.323 |
| LogisticRegression(Ours) | 0.650 |0.682 | 0.666 | 0.040 | 0.436 | 19.910 |


### Table 4year.arff with balanced training data via SMOTE
| ModelName | Precision | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.795 |0.477 | 0.645 | 0.086 | 0.539 | 0.010 |
| GaussianNB | 0.802 |0.588 | 0.701 | 0.108 | 0.561 | 0.022 |
| SVM | 0.801 |0.641 | 0.725 | 0.119 | 0.568 | 9.120 |
| DecisionTree | 0.889 |0.484 | 0.698 | 0.139 | 0.626 | 0.726 |
| SGD | 0.761 |0.758 | 0.760 | 0.125 | 0.553 | 0.101 |
| NearestCentroid | 0.719 |0.654 | 0.688 | 0.093 | 0.512 | 0.006 |
| AdaBoost | 0.849 |0.608 | 0.735 | 0.139 | 0.606 | 7.405 |
| GradientBoosting | 0.862 |0.647 | 0.761 | 0.161 | 0.626 | 6.209 |
| HistGradientBoosting | 0.965 |0.569 | 0.778 | 0.421 | 0.805 | 30.426 |
| MLP | 0.878 |0.523 | 0.710 | 0.139 | 0.621 | 102.660 |
| LogisticRegression(Ours) | 0.754 |0.643 | 0.700 | 0.064 | 0.501 | 22.184 |


### Table 5year.arff with balanced training data via SMOTE
| ModelName | Precision | Recall | ROC_AUC	| PR_AUC	| F1_score	| Time_Used |
| :--------  | :-----  | :----:  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.799 |0.620 | 0.716 | 0.147 | 0.590 | 0.013 |
| GaussianNB | 0.873 |0.678 | 0.782 | 0.229 | 0.674 | 0.007 |
| SVM | 0.867 |0.711 | 0.795 | 0.233 | 0.674 | 2.447 |
| DecisionTree | 0.896 |0.554 | 0.738 | 0.219 | 0.682 | 0.426 |
| SGD | 0.892 |0.694 | 0.800 | 0.265 | 0.703 | 0.053 |
| NearestCentroid | 0.773 |0.736 | 0.755 | 0.160 | 0.585 | 0.002 |
| AdaBoost | 0.905 |0.702 | 0.811 | 0.294 | 0.724 | 4.232 |
| GradientBoosting | 0.901 |0.678 | 0.797 | 0.276 | 0.714 | 3.388 |
| HistGradientBoosting | 0.953 |0.636 | 0.806 | 0.444 | 0.811 | 25.816 |
| MLP | 0.911 |0.587 | 0.761 | 0.261 | 0.712 | 61.952 |
| LogisticRegression(Ours) | 0.813 |0.789 | 0.802 | 0.112 | 0.561 | 16.805 |
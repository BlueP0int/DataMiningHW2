# DataMiningHW2
10 models for Polish+companies+bankruptcy+data

### Table 1 Final Results of all-5-years data for Comparative Models

| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
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







### Table 1year.arff
| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.987 | 0.500 | 0.013 | 0.497 | 0.008 |
| GaussianNB | 0.645 | 0.512 | 0.013 | 0.404 | 0.003 |
| SVM | 0.987 | 0.500 | 0.013 | 0.497 | 0.030 |
| DecisionTree | 0.972 | 0.492 | 0.013 | 0.493 | 0.069 |
| SGD | 0.987 | 0.500 | 0.013 | 0.497 | 0.008 |
| Nearest_Neighbors | 0.626 | 0.502 | 0.013 | 0.397 | 0.002 |
| AdaBoost | 0.986 | 0.561 | 0.053 | 0.587 | 1.212 |
| GradientBoosting | 0.980 | 0.558 | 0.029 | 0.562 | 0.891 |
| HistGradientBoosting | 0.986 | 0.499 | 0.013 | 0.496 | 285.766 |
| MLP | 0.983 | 0.498 | 0.013 | 0.496 | 31.259 |
|LogisticRegression(Ours) |  0.988 | 0.5 | 0.012 | 0.497 |


### Table 2year.arff
| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.983 | 0.500 | 0.017 | 0.496 | 0.002 |
| GaussianNB | 0.795 | 0.685 | 0.034 | 0.486 | 0.003 |
| SVM | 0.983 | 0.500 | 0.017 | 0.496 | 0.078 |
| DecisionTree | 0.954 | 0.485 | 0.017 | 0.488 | 0.073 |
| SGD | 0.983 | 0.500 | 0.017 | 0.496 | 0.011 |
| Nearest_Neighbors | 0.656 | 0.685 | 0.030 | 0.428 | 0.002 |
| AdaBoost | 0.974 | 0.496 | 0.017 | 0.493 | 1.531 |
| GradientBoosting | 0.980 | 0.499 | 0.017 | 0.495 | 1.146 |
| HistGradientBoosting | 0.982 | 0.499 | 0.017 | 0.495 | 295.337 |
| MLP | 0.979 | 0.639 | 0.116 | 0.655 | 35.009 |
|LogisticRegression(Ours) |  0.979 | 0.5 | 0.021 | 0.495 |


### Table 3year.arff
| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.975 | 0.499 | 0.024 | 0.494 | 0.005 |
| GaussianNB | 0.760 | 0.686 | 0.045 | 0.484 | 0.003 |
| SVM | 0.976 | 0.500 | 0.024 | 0.494 | 0.113 |
| DecisionTree | 0.961 | 0.556 | 0.039 | 0.558 | 0.142 |
| SGD | 0.975 | 0.499 | 0.024 | 0.494 | 0.017 |
| Nearest_Neighbors | 0.651 | 0.609 | 0.032 | 0.428 | 0.003 |
| AdaBoost | 0.975 | 0.542 | 0.056 | 0.565 | 1.786 |
| GradientBoosting | 0.973 | 0.498 | 0.024 | 0.493 | 1.384 |
| HistGradientBoosting | 0.975 | 0.521 | 0.037 | 0.532 | 317.250 |
| MLP | 0.957 | 0.596 | 0.056 | 0.585 | 41.183 |
|LogisticRegression(Ours) |  0.973 | 0.5 | 0.027 | 0.493 |


### Table 4year.arff
| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.970 | 0.499 | 0.029 | 0.492 | 0.003 |
| GaussianNB | 0.848 | 0.679 | 0.063 | 0.539 | 0.003 |
| SVM | 0.971 | 0.500 | 0.029 | 0.493 | 0.121 |
| DecisionTree | 0.948 | 0.627 | 0.081 | 0.608 | 0.108 |
| SGD | 0.972 | 0.518 | 0.064 | 0.527 | 0.018 |
| Nearest_Neighbors | 0.719 | 0.699 | 0.056 | 0.478 | 0.003 |
| AdaBoost | 0.963 | 0.548 | 0.051 | 0.564 | 1.768 |
| GradientBoosting | 0.966 | 0.550 | 0.058 | 0.570 | 1.356 |
| HistGradientBoosting | 0.968 | 0.516 | 0.035 | 0.522 | 311.313 |
| MLP | 0.968 | 0.585 | 0.093 | 0.614 | 33.888 |
|LogisticRegression(Ours) |  0.967 | 0.5 | 0.033 | 0.492 |


### Table 5year.arff
| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB | 0.965 | 0.523 | 0.057 | 0.535 | 0.003 |
| GaussianNB | 0.876 | 0.775 | 0.125 | 0.602 | 0.002 |
| SVM | 0.965 | 0.500 | 0.035 | 0.491 | 0.054 |
| DecisionTree | 0.956 | 0.656 | 0.140 | 0.659 | 0.100 |
| SGD | 0.965 | 0.500 | 0.035 | 0.491 | 0.011 |
| Nearest_Neighbors | 0.769 | 0.789 | 0.097 | 0.530 | 0.007 |
| AdaBoost | 0.970 | 0.686 | 0.256 | 0.728 | 1.154 |
| GradientBoosting | 0.942 | 0.786 | 0.214 | 0.698 | 0.833 |
| HistGradientBoosting | 0.974 | 0.642 | 0.270 | 0.708 | 293.055 |
| MLP | 0.962 | 0.705 | 0.213 | 0.710 | 30.333 |
| LogisticRegression(Ours) |  0.966 | 0.5 | 0.034 | 0.491 |
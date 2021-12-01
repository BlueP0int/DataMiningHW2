# DataMiningHW2
10 models for Polish+companies+bankruptcy+data

### Table 1 Final Results for Comparative Models

| modelName | acc	| ROC_AUC	| PR_AUC	| F1_score	| time_used |
| :--------  | :-----  | :----:  | :--------  | :-----  | :----:  |
| BernoulliNB |	0.897 |	0.627 |	0.046 |	0.542 |	4.152 |
| GaussianNB |	0.863 |	0.66 |	0.049 |	0.531 |	1.777 |
| SVM |	0.976 |	0.5 |	0.024 |	0.494 |	5.1 |
| DecisionTree |	0.965	 |0.657 |	0.114 |	0.647 |	1.107 |
| SGD |	0.876 |	0.642 |	0.046 |	0.533 |	0.066 |
| Nearest_Neighbors |	0.723 |	0.675 |	0.042 |	0.467 |	1.054 |
| AdaBoost |	0.978 |	0.582 |	0.131 |	0.628 |	15.569 |
| GradientBoosting |	0.978 |	0.588 |	0.145 |	0.636 |	8.96 |
| **HistGradientBoosting** |	**0.982** |	**0.64** |	**0.262** |	**0.708** |	**329.543** |
| MLP |	0.977 |	0.623 |	0.158 |	0.667 |	13921.868 |

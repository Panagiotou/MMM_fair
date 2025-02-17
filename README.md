### MMM-Fair is a multi-objective, fairness-aware boosting classifier originally inspired by the paper: "Multi-fairness Under Class-Imbalance"
https://link.springer.com/chapter/10.1007/978-3-031-18840-4_21
#

The original algorithm targeted Equalized Odds (a.k.a. Disparate Mistreatment). This MMM-Fair implementation generalizes to multiple fairness objectives:
	•	Demographic Parity (DP)
	•	Equal Opportunity (EP)
	•	Equalized Odds (EO)

We further improve the approach by:
	1.	Flexible Base Learners: Any scikit-learn estimator (e.g. DecisionTreeClassifier, LogisticRegression, MLP) can be used as the base learner.
	2.	Fairness-Weighted Alpha: The boosting weight (alpha) accounts for fairness metrics alongside classification error.
	3.	Dynamic Handling of Over-Boosted Samples: Reduces excessive emphasis on specific samples once fairness goals are partially met.


## Installation

	•	pip install mmm-fair

Requires Python 3.11+.

Dependencies: numpy, scikit-learn, tqdm, pymoo, pandas, ucimlrepo, skl2onnx, etc.

## Usage Overview

You can import and use MMM-Fair directly:

	•	from mmm_fair import MMM_Fair 
	•	from sklearn.tree import DecisionTreeClassifier

# Suppose you have X (features), y (labels)
### 
mmm = MMM_Fair(
    estimator=DecisionTreeClassifier(max_depth=5),
    constraints="EO",        # or "DP", "EP"
    n_estimators=1000,
    random_state=42,
    # other parameters, e.g. gamma, saIndex, saValue...
)
mmm.fit(X, y)
preds = mmm.predict(X_test)

Fairness Constraints
	•	constraints="DP" → Demographic Parity
	•	constraints="EP" → Equal Opportunity
	•	constraints="EO" → Equalized Odds

Pass the relevant saIndex (sensitive attribute array) and saValue (dictionary of protected vs. non-protected group mappings) to MMM-Fair if you want it to track fairness properly for subgroups.

Train & Deploy Script

This package provides a script, train_and_deploy.py, which:
	1.	Loads data (from a known UCI dataset or a local CSV).
	2.	Specifies fairness constraints, protected attributes, and base learner.
	3.	Trains MMM-Fair with your chosen hyperparameters.
	4.	Deploys the model in ONNX or pickle format.

### Example command:

[using UCI library](https://archive.ics.uci.edu)

python -m mmm_fair.train_and_deploy \\ \
  --dataset Adult \\ \
  --prots race sex \\ \
  --nprotgs White Male \\ \
  --constraint EO \\ \
  --base_learner Logistic\\ \
  --deploy onnx

[using local "csv" data]

python -m mmm_fair.train_and_deploy \\ \
  --dataset mydata.csv \\ \
  --target label_col \\ \
  --prots prot_1 prot_2 prot_3 \\ \
  --nprotgs npg1 npg2 npg3 \\ \
  --constraint EO \\ \
  --base_learner tree \\ \
  --deploy onnx

#### Currently the fairness intervention only implemented for categorical groups. 
#### So if protected attribute is numerical e.g. "age" then for non-protected value i.e. --nprotgs provide a range like 30_60 as argument. 


	•	Result: Multiple ONNX files (one per boosting round) plus a model_params.npy inside a directory. It’s then zipped into a .zip archive for distribution or analysis.

MAMMOth Toolkit Integration

The ONNX output and model_params.npy are designed to integrate with the [MAMMOth](https://github.com/mammoth-eu/mammoth-toolkit-releases) or the demonstrator app from the [mammoth-commons](https://github.com/mammoth-eu/mammoth-toolkit-releases) project.

By providing the .zip archive, you can:
	•	Upload it to MAMMOth,
	•	Examine bias and performance metrics across subgroups,
	•	Compare fairness trade-offs with a user-friendly interface.

Example Workflow
	1.	Choose Fairness Constraint: e.g., DP, EO, or EP.
	2.	Define sensitive attributes in saIndex and the protected-group condition in saValue.
	3.	Pick base learner (e.g., DecisionTreeClassifier(max_depth=5)).
	4.	Train with a large number of estimators (n_estimators=300 or 1000) for best performance.
	5.	Optionally do partial ensemble selection with update_theta(criteria="all") or update_theta(criteria="fairness") .
	6.	Export to ONNX or pickle for downstream usage.

References
	•	Original Paper:
“[Multi-Fairness Under Class-Imbalance](https://link.springer.com/chapter/10.1007/978-3-031-18840-4_21),”  Roy, Arjun, Vasileios Iosifidis, and Eirini Ntoutsi. International Conference on Discovery Science. Cham: Springer Nature Switzerland, 2022.


License & Contributing

This project is released under [Apache License Version 2.0].
Contributions are welcome—please open an issue or pull request on GitHub.

Contact

For questions or collaborations, please contact [arjun.roy@unibw.de](mailto:arjun.roy@unibw.de) 
Check out the source code at: [GITHUB](https://github.com/arjunroyihrpa/MMM_fair).
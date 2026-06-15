# CS771: Introduction to Machine Learning - Assignments Portfolio

This repository contains the reports and analysis for the minor assignments of the course **CS771: Introduction to Machine Learning** at IIT Kanpur (CSE, Summer/Term).

## Team Members: **Melboverse**
* **Anushka Rajora** (CSE / 240163) - [anushkar24@iitk.ac.in](mailto:anushkar24@iitk.ac.in)
* **Anushika** (CSE / 240315) - [anushikad24@iitk.ac.in](mailto:anushikad24@iitk.ac.in)
* **Gowra Akshitha** (CSE / 240409) - [gowrak24@iitk.ac.in](mailto:gowrak24@iitk.ac.in)
* **Chinthala Vinayasri** (CSE / 240310) - [chinthalav24@iitk.ac.in](mailto:chinthalav24@iitk.ac.in)
* **Lingamsetti Rajasa** (CSE / 240596) - [lrajasa24@iitk.ac.in](mailto:lrajasa24@iitk.ac.in)
* **Raparthi Bhavishitha** (CSE / 240852) - [bhavisht24@iitk.ac.in](mailto:bhavisht24@iitk.ac.in)

---

## Directory Structure

The repository is organized as follows:
```
.
├── Assignment_1/
│   └── Minor_Assignment_1_Report.pdf  # SVM analysis & geometric proofs
├── Assignment_2/
│   └── Minor_Assignment_2_Report.pdf  # Generative models & missing pixel imputation
├── Assignment_3/
│   └── Minor_Assignment_3_Report.pdf  # EM mixture models & ozone level prediction
└── README.md                          # Portfolio overview
```

---

## Assignment 1: Testing the Limits (Support Vector Machines)

### Overview
This assignment explores the boundary conditions of linear Support Vector Machines (SVMs) using synthetic 2D datasets. It provides a formal geometric proof of linear separability for synthetic data and conducts an empirical search over the hyperparameter space.

### Key Contributions & Analysis
1. **Geometric Proof of Linear Separability**:
   - Analyzed the data generation function `genChallengeData` for challenge level $L > 0$.
   - Defined a coordinate framework with centers: $C_{P1}(-\alpha, \sqrt{2})$, $C_{P2}(\alpha, -\sqrt{2})$, and $C_N(-\alpha, -\sqrt{2})$ where horizontal displacement is:
     $$\alpha = \sqrt{2} \left(1 + \frac{1}{2L}\right)$$
   - Mathematically proved that the minimum perpendicular distance (margin) between the positive class and the negative class remains strictly positive ($Margin = h - 2 > 0$, where $h$ is the triangle's altitude) for all $L > 0$, proving that a linear separating hyperplane always exists.
2. **Hyperparameter Sweep for Low Challenge Settings ($L = 0.2, n = 50$)**:
   - Swung hyperparameters for `LinearSVC` (using L2 penalty and squared hinge loss).
   - Determined the smallest regularization strength ($myC = 0.01$), minimum iteration limit ($max\_iter = 10$), and largest tolerance ($myTol = 10^{-1}$) that still achieve $100\%$ training accuracy.
3. **Hyperparameter Sweep for High Challenge Settings ($L = 70, n = 50$)**:
   - Found that under extreme challenge levels where classes are packed close together, the SVM requires a much stricter tolerance and higher regularization parameter to find the separating plane ($myC = 200.2576$, $max\_iter = 12$, $myTol = 1.25 \times 10^{-4}$).
4. **Effect of Training Points ($n$)**:
   - Verified that dataset size has minimal impact on final accuracy due to guaranteed linear separability, but increases computational training time non-monotonically.

### Key Learnings
- **Regularization vs. Margin**: Regularization parameter $C$ acts as a penalty weight on misclassifications. When $C$ is too low (e.g., $0.001$), the solver prefers a wider margin and tolerates errors (underfitting). Increasing $C$ forces the boundaries to separate the data perfectly.
- **Optimization Tolerance**: Higher tolerance allows the solver to terminate earlier. In simple settings (low challenge level), a loose tolerance ($10^{-1}$) is sufficient. In high-challenge settings, optimization must be extremely precise (tolerance $\sim 10^{-4}$) to resolve the tight margin.

---

## Assignment 2: Inference & Imputation with Missing Pixels

### Overview
This assignment implements simultaneous classification and reconstruction of censored MNIST handwritten digits using a class-conditional Gaussian generative model. It compares a derived **single-step joint MAP** inference framework against a standard **two-step** inference pipeline.

### Key Contributions & Analysis
1. **Mathematical Derivation of Single-Step Inference**:
   - Formulated the joint Maximum A Posteriori (MAP) objective to compute estimates for both the class label $y$ and missing pixels $x_m$, given observed pixels $x_o$:
     $$(y^*, x_m^*) = \arg\max_{c, v} P(y=c, x_m=v \mid x_o)$$
   - Using properties of multivariate Gaussians, proved the optimal reconstruction of missing pixels is the conditional mean:
     $$\hat{x}_m = \mu_{m,c} + \Sigma_{mo,c} \Sigma_{oo,c}^{-1} (x_o - \mu_{o,c})$$
   - Formulated the single-step class score which incorporates conditional covariance volume:
     $$y^*_{\text{single}} = \arg\max_{c} \frac{P(y=c \mid x_o)}{\sqrt{|\bar{\Sigma}_c|}}$$
   - Demonstrated that because of the division by $\sqrt{|\bar{\Sigma}_c|}$ (conditional uncertainty determinant), single-step and two-step methods yield different decision boundaries.
2. **Implementation Details**:
   - Utilized NumPy's submatrix indexing `np.ix_` to cleanly slice covariances.
   - Employed `scipy.linalg.pinv` (pseudoinverse) to handle singular covariance submatrices.
   - Handled numerical stability issues using `np.linalg.slogdet`.
3. **Performance Benchmarking (Censoring Levels 2% to 51%)**:
   - Benchmarked classification accuracy and execution time across different amounts of missing data.
   - **Key Finding**: Standard two-step inference outperforms single-step under heavy censoring ($51\%$, $32\%$, $18\%$) since it ignores noisy missing-data covariance estimates. Single-step inference performs competitively and slightly outperforms under light censoring ($8\%$, $2\%$) where observed data is plentiful.

### Key Learnings
- **Propagating Uncertainty**: Generative classifiers can mathematically model missing data through conditional distribution properties.
- **Robustness vs. Expressiveness**: While the single-step joint MAP formulation is theoretically more expressive because it penalizes classes with high reconstruction uncertainty, it is highly sensitive to errors in covariance estimation when observed features are extremely scarce.

---

## Assignment 3: Expectation-Maximization & Mixture Models

### Overview
This assignment applies the Expectation-Maximization (EM) algorithm to train a Mixture of Experts (using Ridge regression experts) for predicting atmospheric ozone levels ($O_3$) from low-cost sensor data. It evaluates performance against a tuned Decision Tree Regressor.

### Key Contributions & Analysis
1. **EM Algorithm Iteration Tuning ($n\_iter$)**:
   - Analyzed EM optimization stability by sweeping iterations from 1 to 100 at $K = 4$.
   - Identified a "knee point" at $n\_iter = 88$ where the model successfully overcomes early routing boundary volatility (caused by the helper LinearSVC classifier) and stabilizes at an MAE of 10.17 ppb.
2. **Mixture Capacity Scaling ($K$)**:
   - Swept the number of components $K \in \{1, 2, 4, 8, 16, 32\}$.
   - Demonstrated that scaling components reduces Fair MAE from $15.55$ ppb ($K=1$, single regressor) to $9.24$ ppb ($K=32$).
3. **Feature Ablation Study**:
   - Systematically removed one feature at a time to rank predictor importance:
     $$\text{o3op1 (Ozone sensor channel 1)} > \text{Temp(C)} > \text{o3op2 (Ozone sensor channel 2)} > \text{Time} > \text{RH (\% महत्त्व)}$$
   - Explained that Time has lower standalone importance due to multicollinearity/redundancy with temperature and relative humidity.
4. **Mixture Model vs. Decision Tree Benchmarking**:
   - Compared the EM mixture model against a tuned `DecisionTreeRegressor` (best settings: depth=8, MAE=10.91 ppb).
   - Formulated the Pareto Frontier: The $K=32$ EM mixture model is significantly more accurate ($9.24$ ppb MAE vs. $10.91$ ppb), but the Decision Tree is over 150 times faster in prediction latency (0.107 ms vs 16.02 ms).

### Key Learnings
- **Expectation-Maximization as Soft Routing**: EM routes data points to different regressors, performing a soft partition of the input space. This allows a set of linear experts to model complex, non-linear relationships.
- **Pareto Trade-offs in ML**: Higher-capacity local regression mixtures yield superior predictive accuracy because each leaf models a linear trend (rather than a constant mean). However, the routing overhead and multiple model evaluations make it slower, representing a classic speed-accuracy trade-off.

---

## Repository setup and usage
To clone the repository and explore the reports:
```bash
git clone https://github.com/anushkarajora11/CS771-Assignments.git
```
All details regarding mathematical formulations, experimental settings, and results can be found in the respective report PDFs inside `Assignment_1/`, `Assignment_2/`, and `Assignment_3/`.

# 23CSE301 Machine Learning Capstone — Viva Defense & Algorithm Master Guide

> **Project Title:** Industrial Machine Learning Platform — Continuous Emission Regression, Credit Approval Classification, and Unsupervised Robotic Hazard Discovery  
> **Course:** 23CSE301 Machine Learning Capstone  
> **Target Audience:** Group Viva Examination & Technical Evaluation Panel

---

# Capstone project workload split :
**Everyone in the group should have equal contributions on classification and clustering .**

### Example of Work Division Among 3 Members:

#### Classification Algorithms: 
* **Member 1:** 2 classification algorithms (**Bagging Classifier**, **MLP Classifier**)
* **Member 2:** 2 classification algorithms (**AdaBoost Classifier**, **Gradient Boosting Classifier**)
* **Member 3:** 1 classification algorithm (**Random Forest Classifier**) + other deliverables such as **Section A2** (Consolidated 10-Algorithm Comparison Table) & **Section A3** (Model Selection & Hyperparameter Tuning)

#### Clustering Algorithms: 
* **Member 1:** Other deliverables of clustering such as **Section B2** (Dimensionality Reduction & Visualizations: PCA, t-SNE, Elbow, Dendrogram) & **Section B3** (Quantitative Cluster Validation & Safety Hazard Validation)
* **Member 2:** 1 clustering algorithm (**Agglomerative Hierarchical Clustering**)
* **Member 3:** 1 clustering algorithm (**K-Means Clustering**)

*(Note: Review 1 baseline algorithms — Logistic Regression, Decision Trees, KNN, Naive Bayes, SVM, and the Regression Track — are shared foundations distributed equally for contextual viva defense).*

---

# PART 1: Classification Track (UCI Credit Approval)

## Member 1 Responsibilities

### 1. Bagging Classifier (Bootstrap Aggregating)
* **Intuition in Plain English:**  
  Imagine asking 100 different financial auditors to evaluate a credit applicant. Instead of giving them all the exact same training files, each auditor is given a random sample drawn *with replacement* from the main dataset. Each auditor trains an independent Decision Tree. For a new applicant, all 100 trees vote, and the majority decision wins.
* **Core Mathematical Mechanism:**
  1. **Bootstrap Sampling:** Given dataset $D$ of size $N$, generate $B$ datasets $D_1, D_2, \dots, D_B$ where each sample is drawn uniformly at random with replacement. On average, each bootstrap sample contains $\approx 63.2\%$ of the original instances; the remaining $36.8\%$ are Out-Of-Bag (OOB) samples.
  2. **Ensemble Aggregation:**
     $$\hat{y}_{\text{bagging}}(x) = \text{mode} \left\{ h_1(x), h_2(x), \dots, h_B(x) \right\}$$
  3. **Variance Reduction Proof:**  
     If $B$ independent models each have variance $\sigma^2$ with average pair-wise correlation $\rho$, the ensemble variance is:
     $$\text{Var}(\bar{h}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
     As $B$ increases, the second term approaches 0, reducing model variance without increasing bias.
* **Key Hyperparameters:**
  * `n_estimators=100`: Number of base trees in the ensemble.
  * `max_samples=1.0`: Fraction of instances drawn to train each base estimator.
  * `bootstrap=True`: Enables sampling with replacement.
* **Our Project Result:** Accuracy: **$86.96\%$** | F1-Score: **$0.8690$** | ROC-AUC: **$0.9388$**.
* **Viva Punchline:** *"Bagging trains homogeneous base learners in parallel on bootstrap samples to dramatically reduce variance on high-variance models like unpruned decision trees."*

---

### 2. Multi-Layer Perceptron (MLP) Classifier
* **Intuition in Plain English:**  
  A feedforward artificial neural network inspired by biological neurons. Raw numerical and one-hot encoded applicant features pass through successive hidden layers where linear combinations are transformed by non-linear activation functions. The network learns complex decision boundaries that no single straight line or hyperplane could ever capture.
* **Core Mathematical Mechanism:**
  1. **Forward Propagation:**
     For layer $l \in \{1, \dots, L\}$:
     $$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
     $$a^{[l]} = g^{[l]}(z^{[l]})$$
     Where $W^{[l]}$ is the weight matrix, $b^{[l]}$ is the bias vector, and $g(z)$ is the activation function.
  2. **Activation Functions:**
     * **Hidden Layers (ReLU):** $g(z) = \max(0, z)$ (prevents vanishing gradients during backprop).
     * **Output Layer (Sigmoid for Binary Classification):**
       $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
  3. **Loss Function (Binary Cross-Entropy / Log Loss):**
     $$\mathcal{L}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
  4. **Backpropagation (Gradient Descent via Adam Optimizer):**
     Using the chain rule, compute partial derivatives $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$ and $\frac{\partial \mathcal{L}}{\partial b^{[l]}}$:
     $$\frac{\partial \mathcal{L}}{\partial z^{[l]}} = \frac{\partial \mathcal{L}}{\partial a^{[l]}} \odot g'^{[l]}(z^{[l]})$$
     Weights are updated using adaptive first and second momentum moments:
     $$W^{[l]} \leftarrow W^{[l]} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
* **Key Hyperparameters:**
  * `hidden_layer_sizes=(64, 32)`: Two hidden layers with 64 and 32 neurons.
  * `activation='relu'`: Rectified Linear Unit activation.
  * `solver='adam'`: Adaptive Moment Estimation optimizer.
  * `max_iter=500`, `early_stopping=True`: Halts training when validation loss stops improving to prevent overfitting.
* **Our Project Result:** Accuracy: **$86.23\%$** | F1-Score: **$0.8624$** | ROC-AUC: **$0.9234$**.
* **Viva Punchline:** *"MLP optimizes a non-convex loss surface via backpropagation and Adam, mapping tabular credit features into high-dimensional non-linear representations."*

---

## Member 2 Responsibilities

### 3. AdaBoost Classifier (Adaptive Boosting)
* **Intuition in Plain English:**  
  Unlike Bagging where models work independently in parallel, AdaBoost works *sequentially*. It starts with a simple "weak learner" (a Decision Stump — a tree with depth 1). It tests the stump on all data. Whichever applicants the first stump misclassifies are assigned higher weights (higher penalties). The next stump is forced to focus specifically on the difficult, previously missed applicants. At the end, stumps that had lower error rates are given a louder voice in the final decision.
* **Core Mathematical Mechanism:**
  1. **Initialization:** Assign uniform sample weights to all $N$ training examples:
     $$w_i^{(1)} = \frac{1}{N}, \quad \forall i \in \{1, \dots, N\}$$
  2. **Iterative Boosting (for $t = 1, \dots, T$):**
     * Train weak estimator $h_t(x)$ minimizing weighted error:
       $$\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i^{(t)}$$
     * Compute estimator voting weight (Stage Weight $\alpha_t$):
       $$\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$$
       *(If $\epsilon_t < 0.5$, $\alpha_t > 0$; as error decreases, importance increases exponentially).*
     * Update sample weights for the next iteration:
       $$w_i^{(t+1)} = \frac{w_i^{(t)} \exp\left( - \alpha_t y_i h_t(x_i) \right)}{Z_t}$$
       Where $Z_t$ is a normalization factor ensuring $\sum_i w_i^{(t+1)} = 1$. Misclassified instances get multiplied by $e^{\alpha_t} > 1$.
  3. **Final Ensemble Prediction:**
     $$H(x) = \text{sign}\left( \sum_{t=1}^T \alpha_t h_t(x) \right)$$
* **Key Hyperparameters:**
  * `estimator=DecisionTreeClassifier(max_depth=1)`: Decision stump as weak learner.
  * `n_estimators=100`: Maximum number of sequential boosting stages.
  * `learning_rate=0.8`: Scales the contribution of each stump shrinkage factor.
* **Our Project Result:** Accuracy: **$86.23\%$** | F1-Score: **$0.8617$** | ROC-AUC: **$0.9169$**.
* **Viva Punchline:** *"AdaBoost adaptively re-weights hard-to-classify samples at each sequential stage, converting a committee of weak decision stumps into a robust strong classifier."*

---

### 4. Gradient Boosting Classifier (GBM)
* **Intuition in Plain English:**  
  Rather than adjusting sample weights like AdaBoost, Gradient Boosting trains each new tree to directly predict the **residual error (gradient)** of all previous trees combined. If the current ensemble predicts probability 0.70 for an approved applicant (true label 1.0), the residual error is $+0.30$. The next tree is trained to output that missing $+0.30$.
* **Core Mathematical Mechanism:**
  1. **Base Initialization:** Initialize with a constant log-odds value minimizing cross-entropy:
     $$F_0(x) = \arg\min_\gamma \sum_{i=1}^N L(y_i, \gamma) = \ln\left(\frac{p}{1 - p}\right)$$
  2. **Iterative Gradient Descent in Function Space (for $m = 1, \dots, M$):**
     * Compute the **pseudo-residuals** (negative gradient of the loss function):
       $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)} = y_i - p_{m-1}(x_i)$$
       Where $p(x) = \frac{1}{1 + e^{-F(x)}}$.
     * Fit a regression tree $h_m(x)$ to predict the pseudo-residuals $r_{im}$.
     * Compute terminal leaf optimal gamma values $\gamma_{jm}$ using Newton-Raphson approximation:
       $$\gamma_{jm} = \frac{\sum_{x_i \in R_{jm}} r_{im}}{\sum_{x_i \in R_{jm}} p_{m-1}(x_i)(1 - p_{m-1}(x_i))}$$
     * Update model with learning rate $\nu$ (shrinkage):
       $$F_m(x) = F_{m-1}(x) + \nu \sum_j \gamma_{jm} \mathbb{I}(x \in R_{jm})$$
* **Key Hyperparameters:**
  * `n_estimators=100`: Number of sequential boosting stages.
  * `learning_rate=0.08`: Shrinkage rate scaling tree contributions (prevents overfitting).
  * `max_depth=3`: Shallow depth limiting individual tree capacity.
  * `subsample=0.85`: Stochastic gradient boosting utilizing $85\%$ random rows per tree.
* **Our Project Result:** Accuracy: **$86.96\%$** | F1-Score: **$0.8693$** | ROC-AUC: **$0.9272$**.
* **Viva Punchline:** *"Gradient Boosting performs gradient descent in function space by iteratively fitting shallow regression trees to the negative gradient of cross-entropy loss."*

---

## Member 3 Responsibilities

### 5. Random Forest Classifier (Champion Algorithm)
* **Intuition in Plain English:**  
  Random Forest takes Bagging to the next level. If there is one overwhelmingly dominant feature in a dataset (e.g., credit history), every standard bagged tree would still pick it at the root, making all trees correlate with each other. Random Forest solves this by injecting a second layer of randomness: at every single split, it only permits the tree to choose from a small, random subset of features (typically $\sqrt{p}$). This decorrelates the trees, creating diverse perspectives.
* **Core Mathematical Mechanism:**
  1. **Double Randomization:**
     * Random row selection with replacement ($N$ bootstrap samples).
     * Random feature subspace: at each split candidate, choose $m = \lfloor \sqrt{p} \rfloor$ features from total $p$ features.
  2. **Split Criterion (Gini Impurity):**
     For a node $t$ containing samples from classes $k \in \{0, 1\}$:
     $$I_G(t) = 1 - \sum_{k=0}^1 p(k|t)^2$$
     Select split $s$ maximizing Impurity Reduction:
     $$\Delta I_G(s, t) = I_G(t) - \frac{N_L}{N_t} I_G(t_L) - \frac{N_R}{N_t} I_G(t_R)$$
  3. **Ensemble Majority Voting:**
     $$\hat{P}(y=1|x) = \frac{1}{B} \sum_{b=1}^B P_b(y=1|x)$$
* **Key Hyperparameters:**
  * `n_estimators=200`: Total forest population.
  * `max_depth=12`: Depth ceiling preventing leaf memorization.
  * `max_features='sqrt'`: Subspace size ($\approx \sqrt{15} \approx 4$ features per split).
  * `min_samples_split=5`: Minimum instances required to attempt an internal node split.
* **Our Project Result:** Accuracy: **$91.30\%$** | F1-Score: **$0.9127$** | ROC-AUC: **$0.9668$**.
* **Viva Punchline:** *"Random Forest combines bootstrap row bagging with random feature subspace projection, creating an ensemble of decorrelated, low-bias trees that maximize voting variance reduction."*

---

### Deliverable: Section A2 — Consolidated 10-Algorithm Comparison

Member 3 must defend the **evaluation metrics and selection criteria** that proved Random Forest superior over all other 9 algorithms:

| Metric | Mathematical Formula | What It Measures | Why It Matters for Credit Approval |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall percentage of correct approvals and rejections | High-level sanity check (91.3% in our RF) |
| **Precision** | $\frac{TP}{TP + FP}$ | Of all applicants predicted positive, how many were truly creditworthy? | Prevents **Type I Error** (giving loans to defaulting applicants) |
| **Recall (Sensitivity)**| $\frac{TP}{TP + FN}$ | Of all truly creditworthy applicants, how many did the model find? | Prevents **Type II Error** (rejecting good paying customers) |
| **F1-Score (Weighted)**| $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean balancing Precision and Recall | Prevents misleading accuracy on imbalanced classes |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$ | Area Under the Receiver Operating Characteristic curve | Discriminative ability across **all** possible classification thresholds |

#### Final 10-Model Empirical Ranking Table (Defend this in Viva):
1. **Random Forest (Tuned):** Accuracy **$91.30\%$** | F1 **$0.9127$** | ROC-AUC **$0.9668$** *(Champion)*
2. **Logistic Regression (L2):** Accuracy **$89.13\%$** | F1 **$0.8912$** | ROC-AUC **$0.9600$** *(Best linear model)*
3. **Bagging Classifier:** Accuracy **$86.96\%$** | F1 **$0.8690$** | ROC-AUC **$0.9388$**
4. **Gradient Boosting:** Accuracy **$86.96\%$** | F1 **$0.8693$** | ROC-AUC **$0.9272$**
5. **Support Vector Machine (RBF):** Accuracy **$86.96\%$** | F1 **$0.8687$** | ROC-AUC **$0.9421$**
6. **AdaBoost Classifier:** Accuracy **$86.23\%$** | F1 **$0.8617$** | ROC-AUC **$0.9169$**
7. **MLP Classifier:** Accuracy **$86.23\%$** | F1 **$0.8624$** | ROC-AUC **$0.9234$**
8. **Decision Tree (CART):** Accuracy **$84.78\%$** | F1 **$0.8478$** | ROC-AUC **$0.8459$** *(High variance)*
9. **K-Nearest Neighbors ($k=7$):** Accuracy **$84.78\%$** | F1 **$0.8474$** | ROC-AUC **$0.9213$**
10. **Gaussian Naive Bayes:** Accuracy **$81.16\%$** | F1 **$0.8091$** | ROC-AUC **$0.9142$** *(Violated independence)*

---

### Deliverable: Section A3 — Hyperparameter Tuning & Model Selection

* **Tuning Method:** `RandomizedSearchCV` with 5-Fold Stratified Cross-Validation over 40 parameter iterations.
* **Search Space & Optimal Values:**
  * `n_estimators`: $[100, 150, 200, 250, 300] \longrightarrow$ **Optimal: $200$**
  * `max_depth`: $[6, 8, 10, 12, 15, \text{None}] \longrightarrow$ **Optimal: $12$**
  * `min_samples_split`: $[2, 5, 8, 10] \longrightarrow$ **Optimal: $5$**
  * `max_features`: `['sqrt', 'log2', 0.5]` $\longrightarrow$ **Optimal: `'sqrt'`**
* **The "Before vs After" Evidence of Gain:**
  * Baseline Untuned Random Forest: Accuracy $= 87.68\%$, F1 $= 0.8757$, ROC-AUC $= 0.9452$
  * Tuned Champion Random Forest: Accuracy $= \mathbf{91.30\%}$ ($+3.62\%$), F1 $= \mathbf{0.9127}$ ($+0.037$), ROC-AUC $= \mathbf{0.9668}$ ($+0.0216$).
* **Viva Question:** *"Why RandomizedSearchCV instead of GridSearchCV?"*  
  **Answer:** *"GridSearchCV tests every combinatorial permutation ($5 \times 6 \times 4 \times 3 \times 5 = 1,800$ fits), which is computationally wasteful. RandomizedSearchCV samples the parameter probability distribution uniformly, finding within 40 iterations a configuration within $99\%$ statistical confidence of the true optimum in a fraction of the compute time."*

---

# PART 2: Clustering Track (UR3 Robot Telemetry)

## Member 3 Responsibilities

### 6. K-Means Clustering (Centroid-Based Partitioning)
* **Intuition in Plain English:**  
  We want to organize 7,409 telemetry snapshots from the UR3 robot arm into $K=2$ natural operating states without ever showing the algorithm any labels. K-Means randomly places 2 center pins (centroids) in the 19-dimensional sensor space. Every data point is assigned to whichever center pin is closest. Then, the center pins move to the exact mathematical center of their new members. This two-step dance repeats until the pins stop moving.
* **Core Mathematical Mechanism:**
  1. **Objective Function (Inertia / Within-Cluster Sum of Squares - WCSS):**
     $$J(C) = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|_2^2$$
     Where $\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$ is the mean centroid of cluster $C_k$.
  2. **Expectation-Maximization Steps:**
     * **Assignment Step (E-step):** Assign each point $x_i$ to the nearest centroid:
       $$c_i^{(t)} = \arg\min_{k \in \{1, \dots, K\}} \| x_i - \mu_k^{(t)} \|^2$$
     * **Update Step (M-step):** Recompute centroid coordinates:
       $$\mu_k^{(t+1)} = \frac{\sum_{i=1}^N \mathbb{I}(c_i^{(t)} = k) x_i}{\sum_{i=1}^N \mathbb{I}(c_i^{(t)} = k)}$$
  3. **Convergence Condition:** Stops when $\|\mu_k^{(t+1)} - \mu_k^{(t)}\| < \epsilon$ (tolerance $= 10^{-4}$).
  4. **K-Means++ Initialization:** Rather than uniform random placement, K-Means++ chooses the first centroid at random, then chooses subsequent centroids with probability proportional to the squared Euclidean distance from the nearest existing centroid:
     $$P(x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}$$
     This prevents bad local minima traps.
* **Key Hyperparameters:**
  * `n_clusters=2`: Optimal cluster count determined by Elbow and Silhouette analysis.
  * `init='k-means++'`: Smart centroid seeding.
  * `n_init=10`: Runs 10 distinct initializations and keeps the lowest inertia run.
* **Our Project Result:** Silhouette: **$0.3117$** | Davies-Bouldin: **$1.16$** | Calinski-Harabasz: **$1997.1$**.
* **Viva Punchline:** *"K-Means iteratively partitions the 19-sensor feature space to minimize within-cluster variance using Lloyd's expectation-maximization algorithm with k-means++ seeding."*

---

## Member 2 Responsibilities

### 7. Agglomerative Hierarchical Clustering (Connectivity-Based)
* **Intuition in Plain English:**  
  Instead of picking centroids upfront, Agglomerative Hierarchical Clustering is a **bottom-up** approach. It begins by treating every single sensor reading as its own independent single-point cluster. It then looks for the two closest points in the entire dataset and fuses them into a pair. It keeps merging the two most similar clusters step-by-step until all points are joined into a single giant master tree (the Dendrogram). We then slice the tree horizontally at distance threshold $d=3.8$ to yield 2 distinct clusters.
* **Core Mathematical Mechanism:**
  1. **Proximity Matrix:** Compute pairwise Euclidean distances between all clusters:
     $$D(u, v) = \| u - v \|_2$$
  2. **Ward's Minimum Variance Linkage Criterion:**  
     Instead of measuring the distance between the closest points (single linkage) or furthest points (complete linkage), Ward's linkage merges the pair of clusters $(A, B)$ that causes the **smallest increase in total within-cluster sum of squares ($\Delta ESS$):**
     $$\Delta ESS_{AB} = ESS_{AB} - (ESS_A + ESS_B) = \frac{n_A n_B}{n_A + n_B} \| \mu_A - \mu_B \|_2^2$$
  3. **Lance-Williams Update Formula:**  
     When cluster $A$ and $B$ merge into $A \cup B$, the distance to any other cluster $C$ is updated in $O(1)$ time without re-inspecting individual data points:
     $$D(A \cup B, C) = \frac{n_A + n_C}{n_A + n_B + n_C} D(A, C) + \frac{n_B + n_C}{n_A + n_B + n_C} D(B, C) - \frac{n_C}{n_A + n_B + n_C} D(A, B)$$
* **Key Hyperparameters:**
  * `n_clusters=2`: Number of clusters to extract by cutting the tree.
  * `metric='euclidean'`: Distance metric.
  * `linkage='ward'`: Minimum variance criterion.
* **Our Project Result:** Silhouette: **$0.3082$** | Davies-Bouldin: **$1.18$** | Calinski-Harabasz: **$1942.5$**.
* **Viva Punchline:** *"Agglomerative Hierarchical Clustering executes bottom-up greedy merges using Ward's minimum-variance criterion, preserving natural cluster hierarchy and multi-scale data geometry."*

---

## Member 1 Responsibilities

### Deliverable: Section B2 — Dimensionality Reduction & Cluster Visualizations

Member 1 must explain **how we visualize and determine the cluster count** in 19-dimensional robotic telemetry space:

#### 1. Dimensionality Reduction: PCA vs t-SNE
* **Principal Component Analysis (PCA):**
  * **What it does:** Linear orthogonal transformation projecting 19 sensor signals onto directions of maximum variance.
  * **Mathematical Formula:**  
    Eigen-decomposition of the sample covariance matrix $\mathbf{\Sigma} = \frac{1}{N} \mathbf{X}^T \mathbf{X}$:
    $$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$
    Where $\mathbf{v}_1, \mathbf{v}_2$ are eigenvectors corresponding to the two largest eigenvalues $\lambda_1, \lambda_2$.
    $$\text{Explained Variance Ratio} = \frac{\lambda_1 + \lambda_2}{\sum_{j=1}^{19} \lambda_j}$$
  * **Role in Project:** Preserves global variance structure; PC1 ($28.4\%$) and PC2 ($16.2\%$) explain $\approx 44.6\%$ total variance, showing distinct geometric separation between operating regimes.
* **t-Distributed Stochastic Neighbor Embedding (t-SNE):**
  * **What it does:** Non-linear dimensionality reduction that converts Euclidean distances into conditional probabilities representing similarities, preserving local manifold neighborhoods.
  * **Mathematical Formula:**  
    High-dimensional affinities:
    $$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}, \quad p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}$$
    Low-dimensional Student-t distribution (avoids crowding problem):
    $$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|y_k - y_l\|^2)^{-1}}$$
    Minimized using Kullback-Leibler (KL) divergence:
    $$KL(P || Q) = \sum_{i \neq j} p_{ij} \ln\left( \frac{p_{ij}}{q_{ij}} \right)$$

#### 2. Determining Optimal K: Elbow Method & Dendrogram Cut
* **The Elbow Method (Inertia Curve):**  
  Plotted WCSS across $k \in [2, 10]$. Inertia drops precipitously from $k=1 \to k=2$ ($89,200 \to 78,100$), after which the rate of variance explanation flattens out into an "elbow inflection".
* **The Dendrogram Threshold Cut:**  
  In the Agglomerative Dendrogram, cutting the tree horizontally at Euclidean distance threshold **$d = 3.8$** cleanly bisects the root into exactly 2 dominant super-branches, providing independent confirmation of $K=2$.

---

### Deliverable: Section B3 — Quantitative Validation & Ground-Truth Hazard Validation

Member 1 must defend the **three mathematical cluster validation metrics** and the **post-hoc safety hazard validation**:

#### 1. Intrinsic Cluster Metrics (No Labels Needed):

1. **Silhouette Score ($S \in [-1, +1]$):**
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   * $a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \neq i} \|x_i - x_j\|$ (mean intra-cluster distance to own cluster points).
   * $b(i) = \min_{C_B \neq C_A} \frac{1}{|C_B|} \sum_{j \in C_B} \|x_i - x_j\|$ (mean nearest-cluster distance).
   * **Our Score: $0.3117$** (peaks globally at $k=2$; confirms dense cohesion and healthy separation).
2. **Davies-Bouldin Index ($DB \ge 0$, Lower is Better):**
   $$DB = \frac{1}{K} \sum_{i=1}^K \max_{j \neq i} \left( \frac{S_i + S_j}{M_{ij}} \right)$$
   Where $S_i$ is cluster dispersion (average distance to centroid) and $M_{ij} = \|\mu_i - \mu_j\|$ is centroid separation.
   * **Our Score: $1.16$** (indicates tight clusters relative to inter-cluster separation).
3. **Calinski-Harabasz Index (Variance Ratio Criterion, Higher is Better):**
   $$CH = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - K}{K - 1}$$
   Where $B_k$ is the between-cluster dispersion matrix and $W_k$ is the within-cluster dispersion matrix.
   * **Our Score: $1997.1$** (massive between-group variance compared to internal spread).

#### 2. Post-Hoc Safety Hazard Ratio (Real-World Impact):
* **Zero-Leakage Design:** The safety signal `Robot_ProtectiveStop` was strictly excluded from clustering. It was only cross-tabulated after clusters were finalized:
  * **Cluster 0:** Contained $4.76\%$ protective stops.
  * **Cluster 1:** Contained only $1.95\%$ protective stops.
* **Hazard Disparity Ratio:**
  $$\text{Hazard Ratio} = \frac{\text{Stop Rate in Cluster 0}}{\text{Stop Rate in Cluster 1}} = \frac{4.76\%}{1.95\%} = \mathbf{2.44\times}$$
* **Viva Punchline:** *"Unsupervised clustering autonomously segregated high-strain, emergency-stop-prone telemetry regimes with a 2.44× hazard differential without ever seeing a single supervisory hazard label."*

---

# PART 3: Shared Review 1 Baseline Algorithms (Quick Defense)

Any examiner may ask about earlier baseline algorithms tested in Review 1:

| Algorithm | Core Mathematical Concept | Formula / Objective | In Our Project |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear combination passed through sigmoid activation | $P(y=1\|x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$ | Best linear baseline ($89.13\%$ Acc, $0.960$ AUC) |
| **K-Nearest Neighbors** | Non-parametric majority vote of $k$ Euclidean neighbors | $d(x, x') = \sqrt{\sum_{i=1}^p (x_i - x_i')^2}$ | $84.78\%$ Acc ($k=7$; sensitive to feature scale) |
| **Naive Bayes** | Bayes' Theorem with conditional feature independence assumption | $P(y\|x) \propto P(y) \prod_{j=1}^p P(x_j\|y)$ | $81.16\%$ Acc (Independence assumption violated) |
| **Decision Tree (CART)** | Recursive binary partitioning minimizing Gini impurity | $\Delta I_G = I_G(parent) - \sum \frac{N_c}{N} I_G(child)$ | $84.78\%$ Acc (Prone to high variance/overfitting) |
| **Support Vector Machine** | Maximum margin hyperplane with RBF kernel trick | $\max \frac{2}{\|w\|} \quad \text{s.t.} \quad y_i(w^T \phi(x_i) + b) \ge 1$ | $86.96\%$ Acc ($K(x, x') = \exp(-\gamma\|x-x'\|^2)$) |
| **Ridge Regression ($L_2$)** | Ordinary Least Squares with squared weight penalty | $J(w) = \text{MSE} + \alpha \sum_{j=1}^p w_j^2$ | Prevents multicollinearity in sensor signals |
| **Lasso Regression ($L_1$)** | OLS with absolute weight penalty (induces feature sparsity) | $J(w) = \text{MSE} + \lambda \sum_{j=1}^p \|w_j\|$ | Zeroes out uninformative sensor features |
| **Random Forest Regressor** | Average prediction of 200 variance-reduced trees | $\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(x)$ | **Champion Regressor** ($R^2 = 0.985$, $\text{RMSE} = 0.982$) |

---

# PART 4: Top 10 Anticipated Viva Questions & Killer Answers

### Q1: "Why did Random Forest outperform Gradient Boosting and AdaBoost on Credit Approval?"
> **Answer:** *"Gradient Boosting and AdaBoost build trees sequentially to minimize residual errors, making them vulnerable to label noise or outliers in tabular credit datasets. Random Forest builds fully grown, deep trees independently in parallel with double randomization (bootstrap rows + random feature subspaces), which is significantly more resilient to noisy tabular features and prevents sequential error propagation."*

### Q2: "What is data leakage and how did your team prevent it across all tracks?"
> **Answer:** *"Data leakage occurs when information from outside the training dataset is used to create the model. We prevented it through two strict design rules:  
> 1. In Classification and Regression, `ColumnTransformer` (StandardScaler and OneHotEncoder) was `.fit()` strictly on the training partition ($80\%$) and merely `.transform()`ed on the test partition ($20\%$).  
> 2. In Clustering, the ground-truth safety label `Robot_ProtectiveStop` was completely isolated and never seen during feature scaling or clustering fitting; it was only cross-referenced post-hoc in Section B3."*

### Q3: "What is the difference between Bagging and Boosting?"
> **Answer:** *"Bagging trains homogeneous base learners in parallel on independent bootstrap samples to reduce **variance** (error from sensitivity to sample fluctuations). Boosting trains base learners sequentially, where each learner corrects the errors of its predecessor, primarily reducing **bias** (error from overly simplistic assumptions)."*

### Q4: "Why did you use Ward's Linkage for Agglomerative Hierarchical Clustering?"
> **Answer:** *"Single linkage suffers from 'chaining' (long stringy clusters), and complete linkage is overly sensitive to outliers. Ward's linkage minimizes the increase in total within-cluster variance ($\Delta ESS$), creating compact, spherical, balanced clusters that naturally align with physical operating regimes."*

### Q5: "Why did Gaussian Naive Bayes achieve the lowest accuracy (81.16%)?"
> **Answer:** *"Naive Bayes makes the strong assumption that all input features are conditionally independent given the class label ($P(x_1, x_2|y) = P(x_1|y)P(x_2|y)$). In credit approval datasets, financial attributes like income, debt, and credit score have strong mutual correlations. Violating this independence assumption causes Naive Bayes to overcount correlated evidence."*

### Q6: "Why is ROC-AUC more reliable than raw Accuracy?"
> **Answer:** *"Accuracy depends heavily on the chosen decision threshold (default $0.5$) and is easily distorted by class imbalance. ROC-AUC evaluates the True Positive Rate against the False Positive Rate across every possible classification threshold from $0.0$ to $1.0$, measuring the model's fundamental ability to rank positive instances above negative instances regardless of class priors."*

### Q7: "How did you verify that K=2 is the optimal cluster count?"
> **Answer:** *"We triangulated across three independent methods:  
> 1. The **Elbow Method** showed an inflection point in inertia at $k=2$.  
> 2. The **Silhouette Score** peaked globally at $k=2$ ($S = 0.3117$), dropping at $k=3$ ($0.245$) and higher.  
> 3. The **Hierarchical Dendrogram** exhibited its longest vertical branch merge distance at $d=3.8$, cleanly partitioning the dataset into two distinct subtrees."*

### Q8: "What is the role of the activation function in MLP, and why not use Sigmoid everywhere?"
> **Answer:** *"Activation functions introduce non-linearity, allowing the network to approximate arbitrary functions (Universal Approximation Theorem). We use ReLU in hidden layers because its derivative is $1$ for all positive inputs, preventing the **vanishing gradient problem** that plagues deep Sigmoid networks where derivatives saturate near 0."*

### Q9: "What is Out-Of-Bag (OOB) error in Bagging and Random Forest?"
> **Answer:** *"Because bootstrap sampling draws with replacement, each tree misses approximately $(1 - \frac{1}{N})^N \approx \frac{1}{e} \approx 36.8\%$ of the dataset. These unused samples act as an organic, free validation set. Evaluating each tree on its unseen OOB samples provides an unbiased estimate of generalization error without requiring a separate validation split."*

### Q10: "How does the Hazard Ratio validate unsupervised clustering?"
> **Answer:** *"Since clustering is unsupervised, internal metrics like Silhouette only tell us if mathematical clusters are compact and separated. Cross-referencing our clusters with the post-hoc `Robot_ProtectiveStop` telemetry revealed that Cluster 0 exhibited a $4.76\%$ emergency stop rate while Cluster 1 had only $1.95\%$. This $2.44\times$ hazard ratio proves that the clusters correspond to genuine, physically hazardous electro-mechanical operating states."*

---

# Summary Checklist for Group Members

* **Member 1 Checklist:**
  * [ ] Memorize Bagging variance formula: $\text{Var}(\bar{h}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$.
  * [ ] Explain MLP forward and backward pass, ReLU vs Sigmoid, and Adam optimizer.
  * [ ] Walk through PCA covariance matrix eigen-decomposition vs t-SNE KL-divergence.
  * [ ] Explain the three clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz) and the $2.44\times$ Hazard Ratio.

* **Member 2 Checklist:**
  * [ ] Walk through AdaBoost sample weight update equation: $w_i \leftarrow w_i \exp(-\alpha_t y_i h_t(x_i))$.
  * [ ] Explain Gradient Boosting pseudo-residuals: $r_{im} = y_i - p(x_i)$ and shrinkage learning rate $\nu$.
  * [ ] Explain Agglomerative Hierarchical Clustering, Ward's minimum variance criterion, and Lance-Williams update.
  * [ ] Explain how cutting the dendrogram at $d=3.8$ yields $K=2$.

* **Member 3 Checklist:**
  * [ ] Explain Random Forest double randomization (bootstrap rows + $\sqrt{p}$ random features per split).
  * [ ] Defend Section A2 consolidated comparison table and why RF achieved $91.3\%$ Accuracy / $0.967$ ROC-AUC.
  * [ ] Defend Section A3 hyperparameter tuning gains (RandomizedSearchCV, before vs after).
  * [ ] Explain K-Means objective function (Inertia/WCSS), expectation-maximization updates, and K-Means++ seeding.

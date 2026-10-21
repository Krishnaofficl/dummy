# Comprehensive Guide to `notebooks/01_bayes_theorem.ipynb`
**Title:** *01 — Bayes' Theorem Warm-up: From Medical Screening to Student Success*
**Prepared for Presentation & Viva Voce**

---

## 1. Executive Summary: What is this Notebook?

This notebook serves as the **theoretical and pedagogical bridge** of the entire project. It introduces the student/examiner to:
1. The mathematical definition of **Bayes' Theorem** and the Law of Total Probability.
2. The **Base-Rate Fallacy** through the famous **Medical Diagnosis Paradox** (95% sensitive test yielding only ~16.1% posterior).
3. The **Student Performance application**: computing $P(\text{Fail} \mid \text{LowAttendance})$ where failure risk jumps from **20% to 50%**.
4. A **Sensitivity Analysis** showing how the Likelihood Ratio $\lambda = \frac{P(E \mid H)}{P(E \mid \neg H)}$ controls the strength of belief updates.
5. An **Interactive Widget** allowing real-time exploration of prior and likelihood shifts.
6. The fundamental justification for **Bayesian Networks**: why single-variable Bayes fails when scaling to $N$ interdependent variables ($2^N$ combinatorial explosion).

---

## 2. Cell-by-Cell Content & Code Walkthrough

### Cell 0 & 1: Learning Objectives & Mathematical Foundations
* **Bayes' Theorem Formula**:
  $$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$
* **Law of Total Probability (Denominator / Marginal Evidence)**:
  $$P(E) = P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)$$
* **The 4 Fundamental Quantities**:
  | Term | Notation | Meaning in Context |
  |---|---|---|
  | **Prior** | $P(H)$ | Initial belief in hypothesis $H$ before observing evidence $E$. |
  | **Likelihood** | $P(E \mid H)$ | Probability of observing evidence $E$ given hypothesis $H$ is true. |
  | **Marginal Evidence** | $P(E)$ | Total probability of observing evidence $E$ across all worlds. |
  | **Posterior** | $P(H \mid E)$ | Updated belief in hypothesis $H$ after observing evidence $E$. |

---

### Cells 3 to 5: Example 1 — The Medical Diagnosis Paradox (Base Rate Fallacy)

#### Problem Setup:
* **Prior (Prevalence)**: $P(\text{Disease}) = 0.01$ (Only 1% have the disease).
* **Sensitivity (True Positive Rate)**: $P(+ \mid \text{Disease}) = 0.95$ (95%).
* **False Positive Rate**: $P(+ \mid \text{Healthy}) = 0.05$ (5%).

#### Code Implementation (`Cell 4`):
```python
def bayes_posterior(prior: float, likelihood_pos: float, false_alarm: float) -> tuple[float, float]:
    p_evidence = likelihood_pos * prior + false_alarm * (1.0 - prior)
    posterior = (likelihood_pos * prior) / p_evidence
    return posterior, p_evidence
```

#### Step-by-Step Calculation:
1. **Marginal Evidence (Total Positive Tests)**:
   $$P(+) = (0.95 \times 0.01) + (0.05 \times 0.99) = 0.0095 + 0.0495 = 0.0590 \text{ (5.90\%)}$$
2. **Posterior Probability**:
   $$P(\text{Disease} \mid +) = \frac{0.0095}{0.0590} = 0.161016... \approx \mathbf{16.1\%}$$

#### Viva Intuition Check (10,000 Person Breakdown):
* Out of 10,000 individuals:
  * **100 people are sick**: $95$ test positive, $5$ test negative.
  * **9,900 people are healthy**: $5\%$ of $9,900 = \mathbf{495}$ test positive falsely!
  * **Total positive tests**: $95 + 495 = 590$.
  * **Fraction of positive testers who are actually sick**:
    $$\frac{95}{590} \approx \mathbf{16.1\%}$$
* **Core Takeaway**: When the prior is rare (low base rate), false positives from the large healthy pool vastly outnumber true positives from the small diseased pool!

---

### Cells 6 to 9: Example 2 — Student Performance Prediction

#### Problem Setup:
* An instructor observes a student has **Low Attendance** ($>10$ absences).
* **Hypothesis**: Will the student **Fail** the course?
* **Base Failure Rate (Prior)**:
  $$P(\text{Fail}) = 0.20 \quad (\implies P(\text{Pass}) = 0.80)$$
* **Attendance Patterns (Likelihoods)**:
  $$P(\text{LowAttendance} \mid \text{Fail}) = 0.60 \quad \text{(Failing students frequently skip)}$$
  $$P(\text{LowAttendance} \mid \text{Pass}) = 0.15 \quad \text{(Passing students rarely skip)}$$

#### Step-by-Step Calculation (`Cell 7`):
1. **Marginal Evidence**:
   $$P(\text{LowAttendance}) = P(\text{LowAtt} \mid \text{Fail})P(\text{Fail}) + P(\text{LowAtt} \mid \text{Pass})P(\text{Pass})$$
   $$P(\text{LowAttendance}) = (0.60 \times 0.20) + (0.15 \times 0.80) = 0.12 + 0.12 = \mathbf{0.2400 \text{ (24.0\%)}}$$
2. **Posterior Probability**:
   $$P(\text{Fail} \mid \text{LowAttendance}) = \frac{P(\text{LowAtt} \mid \text{Fail}) \cdot P(\text{Fail})}{P(\text{LowAttendance})}$$
   $$P(\text{Fail} \mid \text{LowAttendance}) = \frac{0.12}{0.24} = \mathbf{0.5000 \text{ (50.0\%)}}$$

#### Visualizations (`Cell 9`):
1. **Belief Update Bar Chart**: Shows the dramatic jump in probability of failure from **20% (Prior)** to **50% (Posterior)** upon observing low attendance.
2. **Low-Attendance Pool Pie Chart**:
   * Total Low-Attendance Pool = 24% of students.
   * Failing & Low-Attendance = $12\%$ of total students.
   * Passing & Low-Attendance = $12\%$ of total students.
   * Slice breakdown = **50% Fail vs. 50% Pass**.

---

### Cells 10 & 11: Section 4 — Sensitivity Analysis & Likelihood Ratio

#### Mathematical Insight:
The sensitivity of the posterior to the prior is dictated by the **Likelihood Ratio ($\lambda$)**:
$$\lambda = \frac{P(E \mid H)}{P(E \mid \neg H)}$$

#### The 3 Scenarios Plotted:
1. **Weak Signal**:
   * $P(E \mid \text{Fail}) = 0.35, P(E \mid \text{Pass}) = 0.25 \implies \lambda = 1.4$
   * Curve stays close to the diagonal line; evidence barely updates belief.
2. **Current Attendance**:
   * $P(E \mid \text{Fail}) = 0.60, P(E \mid \text{Pass}) = 0.15 \implies \lambda = 4.0$
   * Moderate convex curve; shifts Prior 0.20 to Posterior 0.50.
3. **Strong Exam Signal**:
   * $P(E \mid \text{Fail}) = 0.90, P(E \mid \text{Pass}) = 0.05 \implies \lambda = 18.0$
   * Steep curve; even a tiny prior quickly escalates to near 100% posterior.
4. **Uninformative Baseline**:
   * Diagonal dashed line $P(\text{Posterior}) = P(\text{Prior})$ where $\lambda = 1$.

---

### Cells 12 & 13: Section 5 — Interactive Prior & Likelihood Explorer

Implemented via `ipywidgets.interact`:
* **Slider 1**: `prior` (School-wide Failure Rate: 0.01 to 0.99)
* **Slider 2**: `p_e_given_fail` (Absence rate among failing students: 0.01 to 0.99)
* **Slider 3**: `p_e_given_pass` (Absence rate among passing students: 0.01 to 0.99)
* Updates live bar chart comparing Prior vs. Posterior with annotated percentages.

---

### Cells 14 & 15: Section 6 — Why We Need Bayesian Networks

This is the **climax of the notebook** and directly motivates the entire Python codebase:

#### The Problem:
Single-variable Bayes assumes only one hypothesis ($H$) and one evidence ($E$).
In real education, student outcomes depend on a web of interrelated factors:
* Parental education level (`Medu`, `Fedu`)
* Study time (`studytime`)
* Past failures (`failures`)
* School support (`schoolsup`)
* Social/alcohol habits (`Walc`, `Dalc`)
* First/second period grades (`G1`, `G2`)

#### The Exponential Wall:
If we try to construct a naive joint distribution table for just 10 binary variables:
$$\text{Parameters} = 2^{10} = 1024$$
For 30 variables:
$$2^{30} > 1,000,000,000 \text{ probabilities!}$$

#### The Bayesian Network Solution:
By factoring the graph according to conditional independence:
$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))$$
Instead of one massive $2^N$ table, we only store small local Conditional Probability Tables (CPTs) for each node and its direct parents.

---

## 3. High-Yield Viva Questions Specific to This Notebook

### Q1: Can you write down Bayes' formula and define each of the 4 terms?
**Answer**:
$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$
1. **$P(H)$ (Prior)**: Probability of the hypothesis before observing new evidence.
2. **$P(E \mid H)$ (Likelihood)**: Probability that the evidence would be observed if the hypothesis is true.
3. **$P(E)$ (Marginal Evidence)**: Total probability of observing the evidence across all scenarios, calculated via total probability: $P(E \mid H)P(H) + P(E \mid \neg H)P(\neg H)$.
4. **$P(H \mid E)$ (Posterior)**: Probability of the hypothesis revised in light of the evidence.

### Q2: Why did a 95% accurate medical test give only a 16.1% posterior probability?
**Answer**:
Because of the **Base-Rate Fallacy**. The disease has a low base rate ($P(D) = 0.01$). Out of 10,000 people, only 100 have the disease (yielding 95 true positives), while 9,900 are healthy. The 5% false positive rate on 9,900 healthy people generates **495 false positives**. Since total positives equal $95 + 495 = 590$, the probability of actually having the disease given a positive test is $\frac{95}{590} \approx 16.1\%$.

### Q3: In the student performance example, why did the probability of failure jump from 20% to 50%?
**Answer**:
The prior failure rate was 20% ($P(\text{Fail})=0.20, P(\text{Pass})=0.80$). The likelihood of low attendance was 4 times higher in failing students ($P(\text{LowAtt} \mid \text{Fail}) = 0.60$) than in passing students ($P(\text{LowAtt} \mid \text{Pass}) = 0.15$). Total probability of low attendance was $0.60 \times 0.20 + 0.15 \times 0.80 = 0.12 + 0.12 = 0.24$. Thus, $P(\text{Fail} \mid \text{LowAtt}) = \frac{0.12}{0.24} = 50.0\%$.

### Q4: Why are the two slices in the pie chart (Cell 9) exactly equal (12% each)?
**Answer**:
Because $P(\text{LowAtt} \cap \text{Fail}) = P(\text{LowAtt} \mid \text{Fail}) \times P(\text{Fail}) = 0.60 \times 0.20 = 0.12$, and $P(\text{LowAtt} \cap \text{Pass}) = P(\text{LowAtt} \mid \text{Pass}) \times P(\text{Pass}) = 0.15 \times 0.80 = 0.12$. Since both joint probabilities are equal to 0.12, exactly half of all students with low attendance are failing and half are passing.

### Q5: What is the Likelihood Ratio $\lambda$, and what does $\lambda = 1$ mean?
**Answer**:
$$\lambda = \frac{P(E \mid H)}{P(E \mid \neg H)}$$
It quantifies how much more likely the evidence is under the hypothesis than under its negation. If $\lambda = 1$, the evidence is equally likely whether $H$ is true or false, meaning the evidence is completely uninformative: $P(H \mid E) = P(H)$.

### Q6: How does this notebook motivate the rest of the project?
**Answer**:
This notebook proves that Bayes' Theorem works for 1 hypothesis and 1 evidence variable. However, it shows that real student performance involves 10+ interacting variables, which would require $2^{10} = 1024$ numbers if done naively. This motivates the transition to **Bayesian Networks** in `network.py`, where conditional independence reduces exponential joint tables into compact local CPTs.

# Master Study Guide: Student Performance Prediction with Bayesian Inference
**Preparation for Presentation & Viva Voce**

---

## 1. The 30-Second Elevator Pitch

> *"This project builds an end-to-end discrete Bayesian Network inference engine from scratch in pure Python and NumPy. We implement both **Exact Inference** (Brute-force Enumeration and Variable Elimination with Min-Fill heuristic) and **Approximate Inference** (Rejection Sampling, Likelihood Weighting, and Markov Chain Monte Carlo Gibbs Sampling). We apply these to predict student academic outcomes (`FinalGrade`) under partial evidence. Through rigorous empirical benchmarks, we demonstrate the fundamental trade-off: **Exact inference guarantees precision but exhibits exponential runtime growth $O(d^N)$, whereas Approximate inference scales gracefully $O(N)$ with a convergence error bound of $O(1/\sqrt{N})$, and Likelihood Weighting overcomes the catastrophic sample collapse seen in Rejection Sampling under rare evidence.**"*

---

## 2. Core Mathematical Foundations

### 2.1 What is a Bayesian Network?
A Bayesian Network is a **Directed Acyclic Graph (DAG)** $\mathcal{G} = (V, E)$ paired with a set of **Conditional Probability Tables (CPTs)**:
- **Vertices ($V$)**: Discrete random variables $X_1, X_2, \dots, X_n$.
- **Edges ($E$)**: Directed edges encoding direct probabilistic dependencies.
- **Markov Property**: Each variable $X_i$ is conditionally independent of its non-descendants given its direct parents $\text{Parents}(X_i)$.

### 2.2 The Joint Distribution Factorization
Thanks to the conditional independence assumptions encoded in the DAG topology, the full joint probability distribution simplifies from an intractable $O(d^n)$ table to the product of local conditional distributions:
$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^n P(X_i \mid \text{Parents}(X_i))$$

### 2.3 Factor Algebra (The Building Blocks)
Every CPT is represented as a multidimensional array (Factor $\phi(X)$):
1. **Factor Product ($\phi_1 \times \phi_2$)**:
   $$\phi_{new}(X \cup Y) = \phi_1(X) \times \phi_2(Y)$$
   Aligned via NumPy broadcasting across matching and disjoint variable axes.
2. **Marginalization ($\sum_{X} \phi$)**:
   Summing out unwanted hidden variables along their corresponding array axes:
   $$\phi_{reduced}(Y) = \sum_{x \in X} \phi(x, Y)$$
3. **Evidence Reduction / Slicing ($\phi[E = e]$)**:
   Instantiating observed variables to their observed indices, collapsing factor dimensions.
4. **Normalization ($\alpha$)**:
   Scaling factor values such that $\sum \phi = 1$.

---

## 3. The Student Network Topology

```
    [StudyTime]       [Attendance]
         \                 /
          v               /
     [PriorGrade]        /
           \            /
            v          v
           [FinalGrade]
```

### Variables & Cardinalities:
- `StudyTime` $\in \{\text{Low}, \text{High}\}$ (Prior: $P(\text{Low})=0.4, P(\text{High})=0.6$)
- `Attendance` $\in \{\text{Poor}, \text{Good}\}$ (Prior: $P(\text{Poor})=0.3, P(\text{Good})=0.7$)
- `PriorGrade` $\in \{\text{Low}, \text{High}\}$ (Conditioned on `StudyTime`)
  - $P(\text{Low} \mid \text{StudyTime}=\text{Low}) = 0.75$, $P(\text{Low} \mid \text{StudyTime}=\text{High}) = 0.20$
- `FinalGrade` $\in \{\text{Fail}, \text{Pass}\}$ (Conditioned on `PriorGrade` and `Attendance`)
  - E.g., $P(\text{Pass} \mid \text{PriorGrade}=\text{High}, \text{Attendance}=\text{Good}) = 0.90$
  - E.g., $P(\text{Pass} \mid \text{PriorGrade}=\text{Low}, \text{Attendance}=\text{Poor}) = 0.15$

**Standard Benchmark Query**:
Given evidence `StudyTime = Low` and `Attendance = Good`, what is $P(\text{FinalGrade} \mid \text{evidence})$?
- **Ground Truth Result**: $P(\text{Fail}) = 32.50\%$, $P(\text{Pass}) = 67.50\%$.

---

## 4. The 5 Inference Algorithms: Deep Dive

| Algorithm | Type | How It Works | Strengths | Failure Mode / Weakness |
|---|---|---|---|---|
| **1. Enumeration** (`exact.py`) | Exact | Full joint sum over all Cartesian assignments to hidden variables. | Simple, exact ground truth. | Exponential time complexity $O(d^n)$; dies at $N \approx 20$. |
| **2. Variable Elimination** (`exact.py`) | Exact | Pushes summations inward: $\sum \prod \rightarrow \prod \sum$, eliminating one hidden variable at a time. | Avoids materializing full joint table; linear on polytrees/chains. | NP-hard in worst case. Dense graphs (high treewidth) produce massive intermediate factors. |
| **3. Rejection Sampling** (`sampling.py`) | Approximate (Prior Sampling) | Ancestral forward sampling from prior; discards worlds that conflict with evidence. | Unbiased, trivial to implement, no graph manipulation. | **Catastrophic sample collapse** on rare evidence ($O(1/P(e))$ rejection rate). |
| **4. Likelihood Weighting** (`sampling.py`) | Approximate (Importance Sampling) | Fixes evidence nodes; samples non-evidence nodes forward; weights sample by $w = \prod P(e_i \mid \text{parents}(e_i))$. | 100% sample retention; resilient to rare evidence. | High variance if evidence is late in the graph (downstream leaves). |
| **5. Gibbs Sampling (MCMC)** (`sampling.py`) | Approximate (Markov Chain Monte Carlo) | Keeps evidence clamped; resamples each non-evidence node from its Markov Blanket conditional distribution. | Samples stay in high-probability regions; efficient for complex dependencies. | Successive samples are autocorrelated; requires **burn-in** and **thinning**. |

---

## 5. Walkthrough of the Three Presentation Benchmarks

### Figure 1: Convergence Error vs. Number of Samples (`01_convergence_error.png`)
- **Metric**: Total Variation Distance (TVD) against Exact VE:
  $$\text{TVD}(P_{approx}, P_{exact}) = \frac{1}{2} \sum_{s} |P_{approx}(s) - P_{exact}(s)|$$
- **Finding**: Log-log plot shows all three approximate algorithms (Rejection, LW, Gibbs) converge to 0 error along the theoretical Monte Carlo rate line of $O(1/\sqrt{N})$.
- **Key Takeaway**: To cut error by half, you must quadruple the number of samples ($4\times N$).

### Figure 2: Rare Evidence Stress Test (`02_rare_evidence.png`)
- **Scenarios Tested**:
  - *Common*: `Attendance = Good` ($P \approx 70\%$) $\rightarrow$ Rejection keeps **~70.0%**.
  - *Moderate*: `StudyTime = Low`, `Attendance = Poor` $\rightarrow$ Rejection keeps **~12.0%**.
  - *Rare*: `StudyTime = High`, `PriorGrade = Low`, `Attendance = Poor` $\rightarrow$ Rejection keeps **< 1.0%** (rejects 99.2% of compute!).
- **Likelihood Weighting**: Keeps **100.0%** of samples across all scenarios because evidence variables are never sampled randomly—they are clamped, and likelihood is accumulated as weights.

### Figure 3: Computational Scalability (`03_runtime_scaling.png`)
- **Benchmark Graph**: Linear Markov chain $X_0 \rightarrow X_1 \rightarrow \dots \rightarrow X_{N-1}$.
- **Enumeration**: Runtime curve curves exponentially upwards ($O(2^N)$), becoming unusable past $N=15$.
- **Variable Elimination**: Stays completely flat/linear ($O(N)$) because on a chain, eliminating $X_k$ only merges two 2-variable factors into a single 2-variable factor.
- **Likelihood Weighting**: Strictly linear $O(N \times S)$, independent of network treewidth.

---

## 6. Top 20 Viva Voce Questions & High-Scoring Answers

### Category A: Core Bayesian Theory
1. **Q: What is the primary advantage of a Bayesian Network over a full joint distribution table?**
   - *Answer*: Scalability and interpretability. A full joint distribution of $N$ binary variables requires $2^N - 1$ parameters. A Bayesian Network exploits conditional independencies ($P(X \mid \text{Parents}(X))$), drastically reducing parameters to $\sum 2^{|\text{Parents}(X_i)|}$, enabling tractable storage and inference.

2. **Q: What is a Markov Blanket, and what is its mathematical significance?**
   - *Answer*: The Markov Blanket of node $X$ consists of its **parents**, its **children**, and its **children's other parents (spouses)**. Node $X$ is conditionally independent of all other nodes in the entire network given its Markov Blanket: $P(X \mid \text{all others}) = P(X \mid \text{MB}(X))$. This local property makes Gibbs sampling computationally efficient.

3. **Q: What is D-separation (Directed Separation)?**
   - *Answer*: D-separation is a graphical criterion that determines whether variable set $A$ is conditionally independent of set $B$ given set $C$. It analyzes three structural motifs:
     1. *Causal chains* ($X \rightarrow Y \rightarrow Z$): Blocked if $Y$ is observed.
     2. *Common cause / Fork* ($X \leftarrow Y \rightarrow Z$): Blocked if $Y$ is observed.
     3. *Common effect / Collider / V-structure* ($X \rightarrow Y \leftarrow Z$): Active/unblocked ONLY if $Y$ or its descendant is observed ("explaining away").

4. **Q: What is the "Base Rate Fallacy", and how does Bayes' Rule resolve it?**
   - *Answer*: It occurs when people judge the probability of an event based solely on specific evidence (e.g., test sensitivity) while ignoring the background prior probability (base rate). In `01_bayes_theorem.ipynb`, a 99% accurate test for a rare disease (0.1% incidence) yields only a ~16.1% posterior probability because false positives from the large healthy population outnumber true positives.

---

### Category B: Exact Inference (Enumeration vs. VE)
5. **Q: Why does Enumeration have exponential time complexity?**
   - *Answer*: To compute $P(Q \mid e) = \alpha \sum_{H} P(Q, e, H)$, Enumeration computes the joint probability for every combination of states across all hidden variables $H$. If there are $k$ hidden variables each with $d$ states, there are $d^k$ terms to evaluate.

6. **Q: How does Variable Elimination achieve efficiency over Enumeration?**
   - *Answer*: By exploiting the distributive law of multiplication over addition: $a \cdot b + a \cdot c = a(b + c)$. VE pushes summations past factors that do not depend on the variable being summed out. Instead of multiplying everything and summing once at the end, it sums out variables incrementally, preventing the creation of an exponential-sized joint factor.

7. **Q: What determines the time complexity of Variable Elimination?**
   - *Answer*: The size of the largest intermediate factor created during the elimination process, which is bounded by the **induced treewidth** $w^*$ of the graph under the chosen elimination ordering ($O(d^{w^*+1})$). Finding the optimal elimination ordering is NP-hard.

8. **Q: What is the Min-Fill heuristic implemented in `exact.py`?**
   - *Answer*: At each step of elimination, Min-Fill selects the variable whose elimination adds the minimum number of fill edges (or connects the smallest set of active neighbors) in the moralized graph. This greedily minimizes the size of intermediate factors.

---

### Category C: Approximate Inference (Sampling Methods)
9. **Q: Why does Rejection Sampling collapse when evidence is rare?**
   - *Answer*: Rejection sampling draws samples unconditionally from the joint prior distribution $P(X_1, \dots, X_n)$. The probability that a sample matches evidence $e$ is exactly equal to the marginal prior probability $P(e)$. If $P(e) = 0.001$, on average 999 out of 1000 generated samples are discarded, leading to massive computational waste.

10. **Q: How does Likelihood Weighting solve the rejection problem?**
    - *Answer*: It never rejects a sample. Whenever it encounters an evidence variable $E_j$, instead of randomly sampling its value, it forces $E_j = e_j$ and multiplies the sample weight $w$ by the conditional probability of that evidence given its sampled parents: $w \gets w \times P(E_j = e_j \mid \text{Parents}(E_j))$.

11. **Q: What is a key limitation of Likelihood Weighting?**
    - *Answer*: If evidence variables are downstream leaves, the upstream ancestral samples are generated blindly from their priors without knowledge of the evidence. When the evidence is finally weighted at the leaves, most samples receive near-zero weights, leading to a low **Effective Sample Size (ESS)** and high variance.

12. **Q: How does Gibbs Sampling differ from Rejection and Likelihood Weighting?**
    - *Answer*: Rejection and Likelihood Weighting generate independent forward samples from the prior. Gibbs Sampling is a **Markov Chain Monte Carlo (MCMC)** method that transitions from state to state by conditioning on the current state of the rest of the network. It always conditions on evidence and spends time exploring the true posterior distribution.

13. **Q: Why do we need "burn-in" and "thinning" in Gibbs Sampling?**
    - *Answer*:
      - **Burn-in**: The Markov chain begins from an arbitrary initialization. It requires a number of initial steps before reaching its stationary distribution (equilibrium). These initial samples are biased and must be discarded.
      - **Thinning**: Successive states in a Markov chain are autocorrelated. By only recording every $k$-th sample (thinning), we reduce autocorrelation between recorded samples.

14. **Q: How does Gibbs Sampling compute the transition probability for variable $X_i$?**
    - *Answer*: $P(X_i = x \mid \text{all other variables}) = P(X_i = x \mid \text{MB}(X_i)) \propto P(x \mid \text{Parents}(X_i)) \prod_{Y \in \text{Children}(X_i)} P(Y \mid \text{Parents}(Y))$. All other variables cancel out as normalization constants.

---

### Category D: Implementation & Metrics
15. **Q: How did you implement Factor Product in `factor.py` without external libraries?**
    - *Answer*: Using NumPy axis manipulation. We determine the union of variables from both factors, expand the dimensions of each factor array with `np.expand_dims` for missing variables, transpose them with `np.transpose` to align variable axes, and perform pointwise NumPy multiplication `v1_aligned * v2_aligned`.

16. **Q: What is Total Variation Distance (TVD) and why use it instead of Mean Squared Error (MSE)?**
    - *Answer*: $\text{TVD}(P, Q) = \frac{1}{2}\sum_x |P(x) - Q(x)|$. It is a standard statistical metric bounded strictly between 0 and 1, representing the maximum difference in probability assigned to any event by the two distributions: $\text{TVD}(P, Q) = \max_A |P(A) - Q(A)|$.

17. **Q: How does `network.py` ensure that the graph is a valid DAG?**
    - *Answer*: `BayesNet.topological_sort()` implements Kahn's algorithm using in-degrees. If the count of topologically sorted nodes is less than the total number of nodes, a cycle exists, and the method raises a `ValueError("The graph contains a cycle! Not a valid DAG.")`.

18. **Q: What is the Effective Sample Size (ESS) in Likelihood Weighting?**
    - *Answer*: $\text{ESS} = \frac{(\sum_{i=1}^N w_i)^2}{\sum_{i=1}^N w_i^2}$. It measures the number of equivalent unweighted independent samples. If all weights are equal, $\text{ESS} = N$; if one sample dominates all weight, $\text{ESS} = 1$.

19. **Q: What happens if an approximate inference query receives zero matching samples?**
    - *Answer*: As a safeguard in `sampling.py`, if accepted samples count is zero or total weight is zero, the code falls back to returning a uniform distribution $1 / |\text{states}|$ to prevent division-by-zero crashes.

20. **Q: What is the core takeaway message for your presentation conclusion?**
    - *Answer*: There is no universally superior inference algorithm. For small, tree-structured networks or when exact numerical guarantees are mandatory, **Variable Elimination** is ideal. For large, dense, continuous, or high-treewidth networks where exact inference is intractable, **MCMC Gibbs Sampling** or **Likelihood Weighting** is essential, balancing sample count against error tolerance.

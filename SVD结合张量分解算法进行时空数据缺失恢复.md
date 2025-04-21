# SVD结合张量分解（STD）算法实现与分析

本文基于SVD结合张量分解（STD）算法，对时序数据的缺失值进行恢复。以下将通过对每个步骤的详细解释和示例来说明该算法的实现原理。

---

## 步骤1：构建张量并处理缺失数据

首先，我们使用一个简单的三阶张量来表示交通速度数据：

$$
X = 
\begin{bmatrix}
\begin{bmatrix} 60 & 62 & 63 \\ 55 & \textbf{?} & 58 \\ 70 & 72 & 68 \end{bmatrix}, 
\begin{bmatrix} 45 & 47 & 48 \\ 50 & 52 & 53 \\ 60 & 65 & \textbf{?} \end{bmatrix},
\begin{bmatrix} 30 & 32 & 35 \\ 40 & 42 & 45 \\ 50 & \textbf{?} & 55 \end{bmatrix}
\end{bmatrix}
$$
其中，**?** 表示缺失值。

---

## 步骤2：多模式偏差初始化

首先，我们计算所有观测到的交通速度数据的平均值：

$$
\mu = \frac{60 + 62 + 63 + 55 + 58 + 70 + 72 + 68 + 45 + 47 + 48 + 50 + 52 + 53 + 60 + 65 + 30 + 32 + 35 + 40 + 42 + 45 + 50 + 55}{24} = 53.08
$$
接着，我们对每个维度（道路、天数和时间窗口）分别计算偏差。以道路偏差为例：

$$
b_1^{(1)} = \frac{60 + 62 + 63 + 55 + 58 + 70 + 72 + 68}{8} - 53.08 = 63.5 - 53.08 = 10.42
$$
类似地，可以计算出其他道路的偏差。

- **道路偏差**：  
  $$
  b_1^{(1)} = 10.42, \quad b_2^{(1)} = 0, \quad b_3^{(1)} = -10.42
  $$

- **天数偏差**：  
  $$
  b_1^{(2)} = -5.42, \quad b_2^{(2)} = 1.92, \quad b_3^{(2)} = 3.5
  $$

- **时间窗口偏差**：  
  $$
  b_1^{(3)} = -5.75, \quad b_2^{(3)} = 0.67, \quad b_3^{(3)} = 5.08
  $$


根据这些偏差，缺失值的初步估算为：
$$
\hat{x}_{2,2,2} = 53.08 + 0 + 1.92 + 0.67 = 55.67\\
\hat{x}_{2,3,3} = 53.08 + 0 + 3.5 + 5.08 = 61.67
$$


---

## 步骤3：截断SVD（Truncated SVD）

张量的模式展开是将张量转换为矩阵的过程。我们以模式-1展开为例，将张量沿第一个维度展开成一个 \(3 \times 9\) 的矩阵。

**模式展开**（mode unfolding）是将张量沿某一特定维度展开为矩阵的过程。在三阶张量的情形中，模式的编号从1到3，分别对应不同的维度。

### 模式-1展开：
将张量沿第一个维度（即"道路"）展开。展开后的矩阵的行对应第一个维度的元素，列则是第二个和第三个维度的组合。举例来说，假设张量$X$ 的维度为 $3×3×3 $，模式-1展开后的矩阵$X(1)$ 将是$ 3×(3×3)$的，即 $3×6$ 的矩阵。

假设张量 \(X\) 如下：

$$
X = 
\begin{bmatrix}
\begin{bmatrix} x_{111} & x_{112} & x_{113} \\ x_{121} & x_{122} & x_{123} \\ x_{131} & x_{132} & x_{133} \end{bmatrix}, 
\begin{bmatrix} x_{211} & x_{212} & x_{213} \\ x_{221} & x_{222} & x_{223} \\ x_{231} & x_{232} & x_{233} \end{bmatrix},
\begin{bmatrix} x_{311} & x_{312} & x_{313} \\ x_{321} & x_{322} & x_{323} \\ x_{331} & x_{332} & x_{333} \end{bmatrix}
\end{bmatrix}
$$
模式-1展开将张量按第一个维度（"道路"）展开成一个矩阵：

$$
X(1) = 
\begin{bmatrix}
x_{111} & x_{112} & x_{113} & x_{121} & x_{122} & x_{123} & x_{131} & x_{132} & x_{133} \\
x_{211} & x_{212} & x_{213} & x_{221} & x_{222} & x_{223} & x_{231} & x_{232} & x_{233} \\
x_{311} & x_{312} & x_{313} & x_{321} & x_{322} & x_{323} & x_{331} & x_{332} & x_{333}
\end{bmatrix}
$$
对该矩阵进行SVD分解，并通过截断SVD保留主要特征。

---

## 步骤4：SVD结合张量分解（STD）

### Tucker分解回顾

Tucker分解是一种将高阶张量分解为一个核心张量和多个因子矩阵的过程。假设我们有一个三阶张量 $X \in \mathbb{R}^{n_1 \times n_2 \times n_3}$，那么Tucker分解可以表示为：

$X \approx G \times_1 U \times_2 V \times_3 W$

其中：

- $G∈Rr1×r2×r3G \in \mathbb{R}^{r_1 \times r_2 \times r_3}G∈Rr1×r2×r3$是核心张量；
- $U \in \mathbb{R}^{n_1 \times r_1}、V \in \mathbb{R}^{n_2 \times r_2}、W \in \mathbb{R}^{n_3 \times r_3}$ 分别是三个模式下的因子矩阵；
- $\times_i $表示沿第 iii 模式上的乘法。

SVD结合张量分解通过优化过程（如梯度下降法）更新因子矩阵 \(U, V, W\) 和核心张量 \(G\)，以最小化目标函数：
$$
\min \frac{1}{2} \| S \ast (X - G \times_1 U \times_2 V \times_3 W) \|_F^2 + \frac{\lambda}{2} (\|U\|_F^2 + \|V\|_F^2 + \|W\|_F^2)
$$

- S 是指示张量，表示哪些元素是观测到的，
- $∗$ 是Hadamard乘积（逐元素乘积）；
- $\| \cdot \|_F$ 表示Frobenius范数；
- $\lambda$ 是正则化参数，用于控制过拟合。

### 梯度计算

- **因子矩阵 \(U\) 的更新**：
  $$
  \frac{\partial L}{\partial U} = - S \ast (X - G \times_1 U \times_2 V \times_3 W) \times_1 G \times_2 V \times_3 W + \lambda U
  $$

- **因子矩阵 \(V\) 的更新**：
  $$
  \frac{\partial L}{\partial V} = - S \ast (X - G \times_1 U \times_2 V \times_3 W) \times_2 G \times_1 U \times_3 W + \lambda V
  $$

- **因子矩阵 \(W\) 的更新**：
  $$
  \frac{\partial L}{\partial W} = - S \ast (X - G \times_1 U \times_2 V \times_3 W) \times_3 G \times_1 U \times_2 V + \lambda W
  $$

- **核心张量 \(G\) 的更新**：
  $$
  \frac{\partial L}{\partial G} = - S \ast (X - G \times_1 U \times_2 V \times_3 W) \times_1 U^T \times_2 V^T \times_3 W^T
  $$


### 梯度更新

使用梯度下降法更新参数：

$$
U \leftarrow U - \alpha \frac{\partial L}{\partial U}\\

V \leftarrow V - \alpha \frac{\partial L}{\partial V}\\

W \leftarrow W - \alpha \frac{\partial L}{\partial W}\\

G \leftarrow G - \alpha \frac{\partial L}{\partial G}
$$
通过迭代，逐步优化因子矩阵和核心张量，直到目标函数收敛。

---

## SVD结合的作用

### 1. 更好的初始化
SVD提取了张量的主特征作为初始因子矩阵，使得张量分解从更好的起点开始，提高了优化的效率和精度。

### 2. 降维处理
SVD帮助确定张量展开矩阵的秩，减少计算复杂度，同时保留最重要的信息。

### 3. 去噪
通过截断SVD，去除噪声和次要成分，进一步提高了缺失数据的恢复效果。

---

## 总结

SVD结合张量分解（STD）通过将SVD用于张量展开的初始化，极大提高了数据恢复的效果。STD利用SVD的降维和去噪能力，为张量分解提供了良好的初始值，并通过梯度下降法优化因子矩阵和核心张量，最终恢复缺失的时空数据。
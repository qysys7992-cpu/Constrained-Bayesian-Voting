# Constrained Bayesian Voting
# 约束贝叶斯投票机制建模与隐性偏好推断

> A reproducible research project for latent preference inference and hybrid expert-public voting mechanism analysis.  
> 一个面向“专家评价 + 公众投票”混合决策系统的隐性偏好推断与投票机制分析项目。

---

## 📌 Project Overview | 项目简介

本项目研究这样一类典型问题：

在一个同时包含**专家评价（Expert Evaluation）**与**公众投票（Public Preference）**的混合决策系统中，如果公众投票数据并不公开，我们能否仅根据已知的专家评分、淘汰结果以及参与者特征，反向推断潜在的公众支持程度，并进一步分析不同投票机制的公平性、稳定性与偏差？

本项目最初来源于一次数学建模研究工作，研究对象为美国电视节目 *Dancing with the Stars (DWTS)* 的专家评分与公众投票机制。

在原始研究中，我们将未知公众投票的推断建模为一个 **Inverse Problem（逆问题）**，并构建了一个：

**Constrained Bayesian Hierarchical Model（约束贝叶斯层次模型）**

将历史淘汰结果作为约束条件嵌入后验推断过程，再通过自适应 MCMC 方法对隐藏的公众投票偏好进行估计。

目前，该项目正在进一步整理为一个：

**可复现、可公开、可持续扩展的研究项目（Reproducible Research Project）**

包括模型实现、实验结果、敏感性分析、投票机制比较以及动态权重机制设计等内容。

---

## 🌐 English Summary

This project studies hybrid decision-making systems that combine expert evaluations with public preferences.

The central challenge is that public voting information may be partially or completely unobserved, while expert scores and final elimination outcomes are observable.

We formulate latent public preference estimation as a constrained inverse problem and develop a **Constrained Bayesian Hierarchical Model**.

Historical elimination outcomes are incorporated into the inference process as constraints, while adaptive Markov Chain Monte Carlo (MCMC) methods are used to explore the posterior distribution of latent public support.

The inferred latent preferences are then used to compare alternative voting mechanisms and to design a **Dynamic Weighted Ranking (DWR)** mechanism that adaptively balances expert and public influence.

This repository is being developed as a reproducible implementation of the original mathematical modeling study.

---

## 🔗 Research Software DOI | 科研软件 DOI

**Version 1.0.0**

Zenodo DOI: **10.5281/zenodo.22334107**

This research software has been publicly archived on Zenodo with a persistent DOI.

本项目首个公开科研软件版本已通过 Zenodo 完成永久归档，并获得可引用 DOI。


# 🔍 Research Question | 研究问题

本项目主要围绕以下几个问题展开：

### 1. 隐性公众偏好能否被反向推断？

节目公开的信息包括：

- 专家评分；
- 选手特征；
- 每周淘汰结果；
- 最终排名。

但真正的公众投票数量并不公开。

因此核心任务是：

> 如何利用“已知结果”反向推断“未知投票”？

我们将其建模为一个受约束的潜变量推断问题。

---

### 2. 不同投票制度是否存在系统性偏差？

研究比较了多种专家—公众混合决策机制，包括：

- Rank-based System  
  排名制

- Percentage-based System  
  百分比制

- Rank + Judge Save  
  排名制 + 专家干预机制

研究重点分析：

- 不同机制是否更偏向专家；
- 是否更偏向公众；
- 是否会放大极端公众偏好；
- 是否会造成结果不稳定。

---

### 3. 专家和公众的偏好是否存在系统差异？

我们进一步分析：

- 年龄；
- 性别；
- 职业类别；
- 是否为运动员；
- 专业舞伴；

等因素对：

- 专家评分；
- 公众支持；

的不同影响。

---

### 4. 能否设计一种更加自适应的投票机制？

基于前述分析，我们提出：

**Dynamic Weighted Ranking (DWR)**

即：

**动态加权排名机制**

其核心思想是：

> 当专家意见高度一致时，提高专家评价权重；  
> 当专家意见分歧较大时，提高公众投票权重。

从而在：

- 专业性；
- 公众参与；
- 公平性；
- 稳定性；

之间取得动态平衡。

---

# 🧠 Methodology | 方法体系

## 1. Data Preprocessing | 数据预处理

原始数据需要进行：

- 异常值检测；
- 缺失值处理；
- 特征标准化；
- 类别变量编码；
- 选手、赛季与专业舞伴信息整理。

原研究中考虑的预处理步骤包括：

- 3σ异常值判断；
- 缺失值补全；
- Z-score标准化；
- One-hot / Target Encoding；
- 高基数类别处理。

---

## 2. Constrained Bayesian Hierarchical Model
## 约束贝叶斯层次模型

这是项目的核心部分。

未知公众投票被视为潜变量：

\[
F_{i,t}
\]

其中：

- \(i\)：选手；
- \(t\)：比赛周次。

模型假设公众投票服从右偏、非负分布，并通过层次结构引入：

- 固定效应；
- 赛季随机效应；
- 选手随机效应。

整体思想为：

```text
Contestant Features
        +
Season Effects
        +
Individual Effects
        ↓
Latent Fan Preference Distribution
        ↓
Elimination Constraints
        ↓
Posterior Distribution

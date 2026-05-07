# MOIVY++ - 多目标常春藤优化算法

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

HMOIVY（Hierarchical Multi-objective Ivy）是一种新型多目标进化优化算法，灵感来源于常春藤植物的生长机制。该算法通过模拟常春藤的攀爬、分枝和扩展行为，实现了在多目标优化问题中的高效搜索。

本项目是毕业设计的源代码实现，包含算法核心代码和基准测试框架。

## 主要特性

- **三层分层更新机制**：精英层、次精英层、被支配层采用不同更新策略
- **混合领导选择**：结合分解式（Tchebycheff）和拥挤距离抽样
- **自适应变异**：多项式变异参数随迭代代数自适应调整
- **外部存档维护**：基于HV贡献的智能截断策略
- **Growth Vector记忆**：保留搜索历史信息，提升搜索平滑性
- **多种边界处理**：支持反射式边界处理，避免个体堆积在边界

## 安装说明

### 环境要求

- Python 3.8 或更高版本
- pip 包管理器

### 安装步骤

```bash
# 克隆仓库
git clone <repository-url>
cd GradProj-Source

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 项目结构

```
GradProj-Source/
├── moivy_v2.py              # MOIVY++ 算法核心实现
├── benchmark_moivy_v3.py    # 基准测试与比较框架
├── requirements.txt         # Python 依赖包列表
└── README.md                # 项目说明文档
```

## 使用方法

### 1. 作为优化器使用

```python
from pymoo.problems import get_problem
from moivy_v2 import moivy_optimize

# 定义问题
problem = get_problem("zdt1")

# 运行优化
result = moivy_optimize(
    problem,
    pop_size=100,
    n_gen=200,
    seed=42,
    archive_size=200,
    verbose=True
)

# 获取结果
print("最终种群目标值形状:", result.F.shape)
print("存档非支配解形状:", result.archive_F.shape)
print("存档决策变量形状:", result.archive_X.shape)
```

### 2. 运行基准测试

```bash
# 运行简单测试套件（ZDT1, ZDT2, DTLZ2）
python benchmark_moivy_v3.py --suite easy --pop 100 --gen 1000 --seeds 1 2 3 4 5 --archive 300 --outdir results_easy

# 运行困难双目标测试（ZDT3/4/6, DTLZ3/4, WFG4/5/9）
python benchmark_moivy_v3.py --suite hard2 --pop 120 --gen 1500 --seeds 1 2 3 --archive 400 --outdir results_hard2

# 运行困难三目标测试（DTLZ3/4, WFG4/9）
python benchmark_moivy_v3.py --suite hard3 --pop 150 --gen 2000 --seeds 1 2 3 --archive 400 --outdir results_hard3

# 自定义问题列表
python benchmark_moivy_v3.py --problems zdt1 zdt4 dtlz3:2:30 wfg4:2:24 --pop 120 --gen 400

# 指定输出目录
python benchmark_moivy_v3.py --suite hard2 --outdir my_results --pop 100 --gen 300
```

### 3. 命令行参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--suite` | 预定义测试套件 (easy/hard2/hard3) | None |
| `--problems` | 自定义问题列表，格式: name[:n_obj:n_var[:k[:l]]] | zdt1 zdt2 dtlz2:2:12 |
| `--pop` | 种群大小 | 80 |
| `--gen` | 迭代代数 | 200 |
| `--seeds` | 随机种子列表 | 1 2 3 4 5 |
| `--archive` | 外部存档大小 | 250 |
| `--outdir` | 输出目录 | results_moivy |
| `--pf_points` | Pareto前沿采样点数 | 4000 |

## 算法原理

### 核心思想

HMOIVY 算法将常春藤的生长行为抽象为以下机制：

1. **生长向量（Growth Vector）**：记录个体的历史移动方向，具有记忆性
2. **邻居机制**：循环邻居与随机邻居混合，兼顾局部结构和随机扰动
3. **三层分层更新**：
   - **精英层（m1）**：第一非支配前沿，偏局部开发与细化
   - **次精英层（m2）**：第二前沿，开发与探索平衡
   - **被支配层（m3）**：偏全局探索与跳出停滞
4. **领导选择**：70% 使用 Tchebycheff 分解选择，30% 使用拥挤距离抽样
5. **SBX嫁接**：部分个体与领导进行模拟二进制交叉，快速形成新解

### 变异策略

- **多项式变异（PM）**：变异参数 eta 从 12 线性增加到 40，前期粗探索、后期细开发
- **SBX交叉**：交叉概率从 0.1 线性增加到 0.4，逐步增强开发能力

### 存档维护

- 合并当前种群与历史存档，保留非支配解
- 二维目标时使用 HV 贡献截断，保护前沿两端极值点
- 高维目标时使用拥挤距离截断

## 性能指标

基准测试框架计算以下性能指标：

| 指标 | 全称 | 说明 |
|------|------|------|
| GD | Generational Distance | 代际距离，衡量收敛性 |
| GD+ | GD Plus | 改进的代际距离 |
| IGD | Inverted Generational Distance | 反向代际距离，综合评价 |
| IGD+ | IGD Plus | 改进的反向代际距离 |
| HV | Hypervolume | 超体积，综合评价指标 |
| Epsilon | Additive Epsilon | 加性 epsilon 指标 |
| Spacing | Spacing | 解集分布均匀性 |
| Delta | Deb's Spread | 前沿覆盖与均匀性（仅2目标） |
| Coverage | C(A,B) | 覆盖率，算法间支配关系 |

## 输出结果

运行基准测试后，输出目录结构如下：

```
results_moivy/
├── details_all.csv          # 所有问题的详细结果
├── summary.csv              # 汇总统计（均值±标准差）
├── problem_id/
│   ├── details.csv          # 当前问题详细结果
│   ├── convergence.csv      # 收敛过程记录
│   ├── coverage.csv         # 覆盖率矩阵
│   ├── coverage_mean.csv    # 平均覆盖率
│   ├── fronts/              # 最终前沿数据（.npz格式）
│   └── plots/               # 可视化图表
│       ├── pareto_overlay_*.png/pdf    # Pareto前沿对比图
│       ├── conv_*.png/pdf              # 收敛曲线
│       └── coverage_heatmap.png/pdf    # 覆盖率热力图
```

## 测试问题

### 预定义测试套件

| 套件 | 问题 | 特点 |
|------|------|------|
| easy | ZDT1, ZDT2, DTLZ2(2obj) | 简单问题，验证算法基本功能 |
| hard2 | ZDT3, ZDT4, ZDT6, DTLZ3/4, WFG4/5/9 | 困难双目标问题 |
| hard3 | DTLZ3/4, WFG4/9 | 困难三目标问题 |

### 支持的测试问题

- **ZDT系列**：ZDT1-ZDT6
- **DTLZ系列**：DTLZ1-DTLZ7
- **WFG系列**：WFG1-WFG9

### 问题参数化格式

```
name[:n_obj:n_var[:k[:l]]]
```

示例：
- `zdt1` - 默认参数
- `dtlz3:2:30` - 2目标，30维
- `wfg4:3:24` - 3目标，24维
- `wfg4:2:24:4:20` - 2目标，24维，k=4，l=20

## 比较算法

基准测试框架包含以下基线算法：

- **NSGA-II**：非支配排序遗传算法II
- **MOEA/D**：基于分解的多目标进化算法
- **SPEA2**：强度Pareto进化算法2


## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

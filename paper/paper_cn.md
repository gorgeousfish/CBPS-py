<!--
  JOSS论文中文版 — cbps
  元信息（对应英文版YAML frontmatter字段）:
    - 软件名: cbps
    - 标题: cbps: Python协变量平衡倾向得分
    - 语言: Python 3.10+
    - 关键词: Python, 因果推断, 倾向得分, 协变量平衡, 观察性研究, 广义矩方法
    - 作者: Cai Xuanyu (ORCID待补充, 澳门城市大学), Xu Wenli (ORCID待补充, 通讯作者, 澳门城市大学)
    - 机构: 澳门城市大学数据科学学院, 中国澳门特别行政区
    - 日期: 2026年2月16日
    - 参考文献: paper.bib
-->

# cbps: Python协变量平衡倾向得分

## Summary

在观察性研究中，由于无法进行随机分组，研究者常借助倾向得分（propensity score）——即个体接受某种处理的条件概率——来调整混杂偏差、估计因果效应。然而，传统的倾向得分估计方法对模型误设（model misspecification）高度敏感：即使模型仅存在轻微的设定错误，也可能导致因果效应估计产生显著偏差。为解决这一问题，Imai and Ratkovic [-@imai2014covariate] 提出了协变量平衡倾向得分（Covariate Balancing Propensity Score, CBPS）方法。该方法基于广义矩方法（Generalized Method of Moments, GMM）框架，在估计倾向得分的同时直接优化处理组与对照组之间的协变量平衡，从而显著提升了对模型误设的稳健性。此后，CBPS方法族不断扩展，已涵盖连续处理变量、纵向数据的边际结构模型、高维协变量设定以及最优平衡条件等多种场景。

`cbps`是CBPS方法族的完整Python实现，覆盖了五篇核心论文提出的全部方法变体，支持二值、多值、连续、纵向和高维处理变量，并提供非参数估计、双稳健（doubly robust）估计、工具变量扩展（CBIV）、诊断工具和可视化功能。在接口设计上，`cbps`同时提供R风格的formula接口和NumPy array接口，前者便于R用户迁移，后者便于与Python机器学习生态集成。此外，该包内置了经典因果推断数据集加载器，便于教学演示和结果复现。

## Statement of Need

自 Rosenbaum and Rubin [-@rosenbaum1983central] 的开创性工作以来，倾向得分——即个体在给定协变量条件下接受处理的概率——已成为观察性研究中因果推断的基石。通过对倾向得分的条件化或加权，研究者可以在非随机化数据中近似随机实验的条件，从而估计处理效应。然而，传统倾向得分方法存在一个根本性矛盾——"倾向得分悖论"（propensity score tautology）[@imai2014covariate]：倾向得分的目的是平衡处理组与对照组之间的协变量分布，但其估计本身依赖于模型的正确设定。一旦模型形式有误，所得到的倾向得分不仅无法实现协变量平衡，反而可能加剧偏差。CBPS方法通过在GMM框架中同时施加得分条件（score condition）和平衡条件（balance condition），从根本上缓解了这一悖论。

目前，CBPS方法族的唯一完整实现存在于R生态中。然而，Python已成为因果推断研究的主流语言之一，DoWhy（https://github.com/py-why/dowhy）、EconML和CausalML等活跃项目构成了蓬勃发展的因果推断生态。尽管如此，现有Python工具均未完整实现CBPS方法族：cbpys [@lal2024cbpys] 基于指数倾斜方法实现了二值处理ATT估计，但未覆盖完整的GMM框架和其他变体。这一缺口迫使需要CBPS的研究者在R和Python之间频繁切换，增加了工作流的复杂性，也限制了CBPS方法在Python社区中的推广。

`cbps`正是为填补这一缺口而开发。该包面向因果推断研究者、计量经济学家、流行病学家和政治学家，通过统一的API覆盖CBPS方法族的全部变体——从基础的二值处理CBPS到纵向数据的边际结构模型CBMSM [@robins2000marginal]，再到高维hdCBPS和最优平衡oCBPS——使研究者无需离开Python生态即可使用完整的CBPS方法论。

## State of the Field

下表总结了现有工具对CBPS方法族各变体的支持情况：

| 工具                             | 语言             |     CBPS     |    CBGPS    |    CBMSM    |    hdCBPS    |    oCBPS    |     CBIV     |
| -------------------------------- | ---------------- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| CBPS [@fong2025cbps]             | R                |      ✓      |      ✓      |      ✓      |      ✓      |      ✓      |      ✓      |
| WeightIt [@greifer2025weightit]  | R                |      ✓      |      ✓      |     部分     |      ✗      |      ✗      |      ✗      |
| CBPS [@premik2017cbps]           | Stata            |      ✓      |      ✗      |      ✗      |      ✗      |      ✗      |      ✗      |
| psweight [@kranker2021improving] | Stata            |      ✓      |      ✗      |      ✗      |      ✗      |      ✗      |      ✗      |
| cbpys [@lal2024cbpys]            | Python           |     部分     |      ✗      |      ✗      |      ✗      |      ✗      |      ✗      |
| balance [@sarig2023balance]      | Python           |     部分     |      ✗      |      ✗      |      ✗      |      ✗      |      ✗      |
| **cbps**            | **Python** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

在R生态中，CBPS包 [@fong2025cbps] 是该方法族的参考实现，覆盖全部六种变体。WeightIt [@greifer2025weightit] 作为通用加权框架支持基础CBPS和CBGPS，但不包含hdCBPS、oCBPS和完整的CBMSM支持。在Stata生态中，Premik [-@premik2017cbps] 基于GMM框架实现了二值处理CBPS，psweight [@kranker2021improving] 提供了带权重稳定化的倾向得分加权方法；两者均仅支持二值处理变量。在Python生态中，cbpys [@lal2024cbpys] 基于指数倾斜方法实现了二值处理ATT估计，但未采用完整的GMM框架；balance包 [@sarig2023balance] 仅覆盖面向调查非响应校正的二值处理CBPS。熵平衡 [@hainmueller2012entropy] 和广义倾向得分 [@hirano2003efficient] 等相关方法与CBPS在理论基础和估计策略上存在本质差异。

我们选择将 `cbps`作为独立包发布，而非贡献至现有项目，基于以下考量：（1）CBPS方法族基于独特的GMM优化框架，其矩条件构造和权重矩阵选择与通用因果推断库的架构不兼容；（2）完整实现五篇核心论文的全部方法变体——包括诊断、推断和可视化——需要专门的模块化设计；（3）与R参考实现的数值对齐要求独立的、针对性的测试基础设施；（4）现有Python因果推断库围绕因果图识别（DoWhy）或异质性效应估计（EconML）设计，其架构不适合嵌入基于GMM的倾向得分方法。

## Software Design

`cbps`的设计围绕三个核心决策展开，每个决策均服务于学术软件的可复现性和易用性目标。

**GMM统一框架。** 所有CBPS变体共享广义矩方法的理论基础 [@hansen1982large]。以二值处理为例，CBPS的核心思想是同时求解以下矩条件：

$$
\frac{1}{N}\sum_{i=1}^{N} g(T_i, X_i; \beta) = 0
$$

其中矩函数 $g$ 同时包含确保似然最大化的得分条件（score condition）和确保加权后协变量平衡的平衡条件（balance condition）。当两类条件联合使用时形成过度识别估计，可通过Hansen's J检验评估模型设定 [@hansen1982large]。我们选择以此GMM框架作为统一基础——而非为每种方法独立实现优化——因为它确保了从基础CBPS [@imai2014covariate] 到连续处理CBGPS [@fong2018covariate]、纵向数据CBMSM [@imai2015robust]、高维hdCBPS [@ning2020robust]、最优平衡oCBPS [@fan2022optimal] 以及双稳健估计 [@tsiatis2007comment; @robins1999association] 和经验似然扩展 [@owen2001empirical] 之间的方法一致性。

**双接口设计。** 该包同时提供R风格的formula接口和NumPy array接口，而非仅支持单一接口。formula接口降低了R用户的迁移成本，array接口则便于与scikit-learn等Python机器学习生态集成。

**模块化方法族架构。** 各CBPS变体的数学基础和优化策略差异显著，因此我们按方法类型将代码组织为独立模块（core、msm、highdim、nonparametric、iv等），而非采用单一类继承体系。这种设计允许各模块独立开发、测试和维护。

## Research Impact Statement

作为新发布的软件包，`cbps`通过以下三方面证据展示其近期可信的研究影响。

**数值验证。** 该包的蒙特卡洛验证测试覆盖了全部五篇核心论文的数据生成过程（DGP），使用与原始论文相同的模拟设定，验证偏差（Bias）、均方根误差（RMSE）和收敛率等指标与论文报告结果一致，容差基于蒙特卡洛标准误（3×MC SE）。这些测试在持续集成流水线中自动运行，确保各版本发布间的数值稳定性。

**软件质量。** 该包建立了多层测试体系，包含超过2,000个测试函数（分布在125个测试文件中），覆盖单元测试、集成测试、蒙特卡洛模拟验证和论文结果复现。项目提供完整的Sphinx API文档并托管于ReadTheDocs，附带3个Python复现脚本分别对应Imai and Ratkovic (2014, 2015) 和Fong, Hazlett and Imai (2018) 的核心分析。持续集成通过GitHub Actions实现，覆盖多操作系统和多Python版本。

**可复现性贡献。** `cbps`使原本仅在R中可用的CBPS方法族可在Python生态中直接使用，降低了方法应用的语言门槛。包内内置四个数据集加载器——LaLonde就业培训数据 [@lalonde1986evaluating]、Blackwell纵向政治竞选数据 [@blackwell2013framework]、连续处理模拟数据和npCBPS验证数据——便于教学演示和方法比较。双接口设计和scikit-learn兼容接口进一步降低了不同背景研究者的使用门槛。

## Acknowledgements

我们感谢Kosuke Imai、Marc Ratkovic和Christian Fong等学者在CBPS方法论和R包参考实现方面的开创性贡献。`cbps`的开发依赖于NumPy [@harris2020array]、SciPy [@virtanen2020scipy]、statsmodels [@seabold2010statsmodels] 和pandas [@mckinney2010data] 等开源项目。

# References

# r11：有界离线类别提升树（性能评分前固定）

r10已按原+0.005/4of6门止损，不调旧参数/种子，不改变r09下一正式候选或原r05最好。
新的r11是非线性类别提升树，每目标独立拟合完整可见标签与全部GIVEN。框架使用
CatBoost1.2.10，遵守Apache2许可并明确上游来源，不把CatBoost本身称作原创算法。
参与者自有部分是信息边界、逐目标拟合/支持映射/平滑、规范输出与可复现试验。

官方类顺序文档：https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier ，
参数说明：https://catboost.ai/docs/en/references/training-parameters/common 。
本机依赖准入bf846fe0已完成，只证明import/小拟合；不重跑它当性能或原生证据。

所有GIVEN声明选项编码为稳定类别字符串，NaN是独立缺失码，未知选项拒绝；不用
respondent_id/role/EXCLUDE/行号/index/题名/国家特例。每目标只用该目标可见标签，
全块隐藏查询不参加任何拟合或边际，其他PREDICT值不是特征。未缺失目标不额外
训练。GIVEN缺失输出取至少一PREDICT可见行的0.5平滑边际，查询分布不作训练统计。

不足90个标签、单一已见类别、无GIVEN或训练GIVEN全常量时用0.5平滑目标边际。
其余固定：96轮、深度6、
learning_rate0.08、l2_leaf_reg5、Ordered boosting、Bayesian bootstrap、temperature1、
one_hot_max_size16、max_ctr_complexity1、random_strength1、random_seed20261012。
CPU/thread_count2、loss MultiClass、use_best_model=False、allow_writing_files=False。
没有验证集/调参/early-stop/外部权重/模型API/日志写盘。全部GIVEN、目标/样本保留；
限制CTR组合复杂度是预定计算/正则选择，不是删问题或观察分数后改参数。

按model.classes_映射概率到完整schema选项；未见类别也保留。若目标有n可见标签、
K完整选项，用 (n*p + 0.5)/(n+0.5*K) 做固定支持平滑；缺失类别p为0。不执行打印
observed_if或硬gate，NA_GATED是普通可学类别。非法类映射/概率必须失败，不静默
用另一模型补造。输入frame不变、缺失向量按schema顺序/行主序排列。

先8项新增宿主小检查，60秒上限：一个真实非线性信号、真实训练行/部分标签边界、
query/ID/index不影响拟合、非连续/逆序类别支持映射、稀疏/单类/空训练边际、软gate、
非法输出拒绝与完整规范顺序。多数边界检查用自有可观测假后端，不是问卷性能。
Linux首次对应检查另验证跨平台执行，不重跑旧r10/r09/准入套件。

性能用全新seed202610121与202610122，完整UNHCR→UNICEF→WorldBank，与冻结r05
配对。官方公开工具固定0d2332d8ae19a8ce171031142bdc97134910e7ec，源/schema/config
均不改。最多12格/6对，4/6差>=.005且没有<-.015；明显退步或剩余胜场不可达立即
止损，剩格不补、未知不记零、不调参数/seed/质量门。本批是独立方法，不从既有
不同seed结果直接断言比r09更好；正式最佳只能由真实官方反馈更新。

本机约11GiB逼近10GiB门，完整评分首次放在自有PUBLIC仓库的免费标准Ubuntu24
运行器；不创建付费大runner/GPU/模型API/artifact/cache/周期任务，最多两深线。
使用Python3.12、NumPy2.5.3/pandas3.0.6/CatBoost1.2.10/SciPy1.18.1，依赖固定。
候选预测预算按完整类别工作量预分配480/120/240=840秒，留60秒给启动等；正式
Development900共享总预算不扩大。对照各300；父步骤候选限600/240/360秒，批次
1800秒/CI35分钟、输出50MiB、每步可用盘>=10GiB。候选一轮父步骤之和需<=900，
更保守地包含依赖准备和本地评分；失败则停止，不扩大预算/裁样本让它通过。

新ZIP与原r05 ZIP必须实际核对并通过合同门；每对输入/人数/评分格数相同，前后
source/data SHA不变。输出仅公开虚构数据/聚合成绩，不传真实数据/CV/邮件/凭据。
方法门过后才另做实际ZIP最终phase2、原生资源/当时账号阶段当前已接受条款额度
与精确SHA门，才可能正式一次提交。不自动选Test或称获奖/H100/真实泛化。
已知新增服务实付/承诺0，综合月2000上限、通用软件/电费/invoice覆盖仍未全核实。

# R15：单次联合训练的共享分类网络（品质评分前固定）

R14在完整UNHCR首题由真实子日志确认predict600秒超时，固定双fit配置关闭。
不复活该源/grid/seed或调旧预算；官方最好R11开发显示.62保留。R15是新标准
共享MLP/Adam方案的独立NumPy实现，不把MLP、Adam或分段softmax称作发明。
方法背景：[Adam原论文](https://arxiv.org/abs/1412.6980)；分段归约按
[NumPy官方reduceat语义](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduceat.html)。
这些来源不证明本比赛性能，也不替代完整实验。

仅GIVEN分类one-hot（含独立missing类）为特征，按至少一个PREDICT可见标签的
eligible行均值中心化，原全隐藏查询排除拟合/中心/先验。ID/role/索引/其他PREDICT
标签不作输入特征；PREDICT标签可共同用于多任务损失，未知标签不贡献损失。
完全没有GIVEN变化/需要预测的有效头时只用完整支持的半计数先验。

先验每类+.5；活动头至少90个可见标签且至少2个已见类，可共同训练没有当前
查询的活动辅助头；至少一个需预测活动头才拟合。64维tanh共享隐藏层，各头
完整选项softmax（普通NA_GATED，不硬gate）。输出bias初值logprior；W1 std=1/sqrt(D)，
W2 std=.05/sqrt(64)，隐藏bias N(0,.01)，固定模型seed20261016。

固定192步full-batch Adam，lr=.02、beta1=.9、beta2=.999、eps=1e-8、全局梯度范数
上限5；L2=.001。目标为各活动头已知样本平均CE再对头平均，加
.001/(2*头数)*(sum(W1^2)+sum(W2^2))，bias不惩罚。稳定分段log-softmax，无
数值下溢的log(trueprob)损失；最终预测.95*网络概率+.05*正先验并归一化。
无holdout再拟合/后校准/模型API/隐藏真值/训练权重文件；每次predict仅一次联合fit。

所有原schema项先完整校验，原frame/schema不改。GIVEN与不活动PREDICT缺失返回
先验，其余返回对应头完整向量；原全query/全部目标/行序canonical不缩减。
float64，估算密集工作区上限768MiB（超限失败、不下采样），全量训练/查询不变。

12项新原生控制：分段log-softmax/极端稳定、解析梯度全参数有限差分、缺失标签
不计损失、先验完整支持、GIVEN设计与中心不受query/ID影响、只拟合可见标签行
且一次、缺少GIVEN/少标签/单类回退、全canonical与输入不变、非法schema/向量
拒绝、模型确定性、真实小型多头优化与学习信号。不是旧模型微测重跑或品质证明。
不做本机拟合，本机6.6GiB低于10GiB；新原生控制仅一次成功执行后进入完整比较。

在任何R15品质前固定freshseed202610161与202610162、完整UNHCR→UNICEF→WorldBank，
冻结R11同输入配对、官方0d2332d8/config/schema，不删正式行/目标/坏例。最多
12评分/6配对，原4of6 delta>=.005且无<-.015。质量不可达/明显退步/执行或资源
失败即停，不改本模型宽/步数/混合/正则/seed/阈值追分；C2须真实C1结果允许。

预测caps600/90/150=840＋60余量，候选每cohort父步骤和<=900；cohort1800秒/CI
35min/50MiB/>=10GiB空盘，最多两组条件顺序PUBLIC标准免费CPU；有界失败日志保留。
完整质量过门后实际最终入口另为900/300/300=1500<官方3600。未来真实账号/阶段/
当前已接受实质条款/实际可用额度/同ZIP SHA门后才一次正式提交，Test仍未选。

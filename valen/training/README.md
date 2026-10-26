# Valen：SFT 与 RLCD

`runner.py` 共用数据分片、token 预算、梯度同步和 checkpoint 流程。`sft.py` 实现标签监督，`rlcd.py` 实现基于 GRPO 的优化。stage 由各架构的 builder 解释，method 决定训练目标，两者独立。

runner 从架构 backend 取得训练执行单元，再调用 `loss_from_logits`。Qwen shared_state 与双编码器按 state 共享计算图，question 模式按题及分支执行。`state_mean` 先在 state 内平均，`question_mean` 按全局有标签 QA 平均，需与 backend 执行单元保持一致。

这里的 RLCD 是 Valen 的实验实现，奖励公式以本仓库的 [rlcd.py](rlcd.py) 为准。

## SFT loss

[sft.py](sft.py) 对候选 logits 使用分布交叉熵。设预测概率为 p、标签分布为 y，候选数为 K：

$$
\mathrm{CE}=-\sum_{k=1}^{K}y_k\log p_k.
$$

$$
\mathrm{RPS}=\frac{1}{K-1}\sum_{j=1}^{K-1}
\left(\sum_{k=1}^{j}(p_k-y_k)\right)^2.
$$

$$
L_{\mathrm{SFT}}=\mathrm{CE}+w_{\mathrm{RPS}}\mathrm{RPS}+w_{\mathrm{Brier}}\sum_k(p_k-y_k)^2.
$$

硬标签和软标签使用同一公式。`rps_weight` 和 SFT 的 `brier_weight` 缺省为 0，此时三种任务都只优化交叉熵。Score 的 RPS 区分等级间的远近，使用前 K−1 个累积分布差；Choice 和 Noul 的 RPS 项为 0。RLCD 的 Brier 权重使用单独的 `rlcd.brier_weight`。

`state_mean` 在每个 state 内先对有标签的问题平均，再对全局有效 state 平均，一条三题记录与一条单题记录权重相同；`question_mean` 按全局有标签 QA 平均。Score 的多个分支先拼成同一道题的 logits，不各自增加损失权重。

## RLCD reward与loss

每道题从旧policy的候选分布中有放回采样 `group_size` 个action，不生成文本。对采样答案 $a_i$，记其旧策略概率为 $p_i$，标签为 $y_i=y_{a_i}$：

$$
r_i = w_{\mathrm{correct}}y_i
-w_{\mathrm{confidence}}\left[y_i(1-p_i)^2+(1-y_i)p_i^2\right].
$$

硬标签下，选中答案的标签 $y_i$ 表示答对与否，第二项就是 $(p_i-y_i)^2$。默认权重均为 1：以 0.9 的概率答对，奖励 0.99；以 0.9 的概率答错，奖励 −0.81。软标签保留原分布，使用期望正确性与期望二元平方误差，不先取 argmax。

这里的“置信度”是所选候选的概率。Score 的 RL 动作是一个离散action，推理仍输出期望值。

组内优势为：

$$
A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)+\epsilon}.
$$

标准差使用总体定义；奖励完全相同的组，优势明确置零。采样动作、奖励、优势、旧策略 log-prob 在一次 rollout 的所有更新中固定。

$$
\rho_i=\frac{\pi_\theta(a_i\mid x)}{\pi_{\mathrm{old}}(a_i\mid x)}.
$$

$$
L_{\mathrm{policy}}=-\mathrm{mean}_i\min\left(
\rho_i A_i,\mathrm{clip}(\rho_i,1-\delta,1+\delta)A_i
\right).
$$

$$
L=L_{\mathrm{policy}}+\beta D_{\mathrm{KL}}(\pi_\theta\Vert\pi_{\mathrm{ref}})
+w_{\mathrm{Brier}}\sum_k(\pi_\theta(k)-y_k)^2.
$$

参考策略固定为 RL 开始时的模型，KL 用full-distribution精确计算，约束策略更新幅度。rollout 和策略更新时关闭 dropout；训练前向仍支持 gradient checkpointing。

Brier 辅助项默认开启，设 `brier_weight=0` 可关闭该辅助项。

## Implementation ownership / 实现维护边界

This file retains objective formulas and internal invariants beside the implementation. User training methods, commands and configuration are maintained only in [website training docs](https://valen-team.github.io/docs/valen/training/) and the [complete domain Cookbook](https://valen-team.github.io/cookbook/domain-finetuning/).

开发者应核对 `runner.py` 的归一化、执行单元及显式梯度同步，`checkpoint.py` 的原子权重写入、配置写入顺序和每 rank RNG/游标恢复，以及 `sft.py` / `rlcd.py` 的 loss/state_dict。共享 state 的 token 计数不包含 padding、反向或参考策略额外前向。Checkpoint 保存可训练参数的完整值，冻结参数依赖原基座；恢复参数集合与 world_size 是加载器不变量。

[保存与恢复参考](https://valen-team.github.io/zh-Hans/docs/valen/checkpoints/) · [配置参考](https://valen-team.github.io/zh-Hans/docs/valen/configuration/)；本页不再维护用户教程副本。

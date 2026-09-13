## Experiment 01 — Action Scale

控制变量：
- Robot: Franka Panda
- Task: Reach
- Controller: Differential IK, relative, DLS
- Raw action: [0.2, 0, 0, 0, 0, 0]
- Number of envs: 4
- Steps: 30
- Target range: 相同

自变量：
- A: action scale = 0.5
- B: action scale = 0.2

因变量：
- Mean EE displacement

结果：
- Scale 0.5: Mean EE displacement: 0.4741019308567047
- Scale 0.2: Mean EE displacement: 0.18418797850608826

观察：
Scale=0.5条件下Mean EE displacement近似Scale=0.2条件下的2.5倍

解释：
由于两组实验仅scale不同，0.5是0.2的2.5倍，但是由于IK过程无法完全精确的计算出joint变化，因此近似2.5倍，而非精确的2.5倍


## Experiment 02 -Official Franka Reach Baseline
- Task:
Isaac-Reach-Franka-v0

- Algorithm:
PPO / RSL-RL

- num_envs:
official

- seed:
official

- max_iterations:
official default

- Reward:
official

- Action:
official

- 最终 checkpoint:
model_999.pt

- 最终 Mean Reward:
0.36

- 训练是否稳定:
是，训练总体稳定并基本收敛。

- Play 行为:
机械臂能够稳定追踪随机生成的末端目标位姿，运动过程较平滑，无明显抖动或发散，目标更新后能够重新调整并逐渐靠近新目标
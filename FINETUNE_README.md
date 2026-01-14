# Wan2.1 Fine-tuning 项目文档

## 项目概述

本项目是基于Wan2.1视频生成模型的机器人动作控制微调框架。通过VACE（Video-Action Conditional Encoding）架构和LoRA（Low-Rank Adaptation）技术，实现了从多相机视觉观测预测机器人动作序列的功能。

### 核心特性

- **VACE架构**：将视觉观测和动作信息融合到视频生成模型中
- **LoRA微调**：仅训练少量参数，保持预训练模型性能
- **多相机支持**：支持4相机视角的同时处理
- **时序预测**：基于历史观测预测未来动作序列
- **分布式训练**：支持FSDP（Fully Sharded Data Parallel）多GPU训练

---

## 项目结构

```
finetune/
├── __init__.py                 # 包初始化文件
├── config.yaml                 # 训练配置文件
├── dataset.py                  # 数据加载器
├── robotwin_bbh.json          # 动作归一化统计信息
├── vace_action.py             # VACE推理接口
└── vace_model_action.py       # 动作条件模型定义

train_vace_lora.py             # LoRA训练主脚本
```

---

## 模型架构

### 1. VaceWanActionModel

核心模型类，继承自`WanModel`，增加了动作条件支持。

#### 主要组件

```python
VaceWanActionModel
├── ActionEncoder           # 动作编码器
│   ├── action_embedding    # 动作特征提取 (action_dim → hidden_dim)
│   ├── temporal_conv       # 时序下采样卷积 (可选)
│   └── output_proj         # 输出投影 (hidden_dim → 6*dim)
├── vace_patch_embedding    # VACE上下文补丁嵌入
├── vace_blocks             # VACE注意力块 (双分支架构)
└── blocks                  # 基础Transformer块
```

#### 关键特性

1. **零初始化策略**：动作编码器输出层采用零初始化，确保训练初期保持预训练模型行为
2. **双分支架构**：
   - VACE分支：处理视觉上下文信息
   - 基础分支：主Transformer处理流
3. **全局动作调制**：动作特征经过时间聚合后，作为全局调制信号影响所有层

### 2. ActionEncoder

动作序列编码器，采用时序建模策略。

```python
ActionEncoder(
    action_dim=14,           # 机器人动作维度
    hidden_dim=128,          # 隐藏层维度
    output_dim=6*2048,       # 输出维度 (6个AdaLN参数 × 模型维度)
    temporal_downsample=False  # 是否使用时序下采样
)
```

**工作流程**：
```
actions (B, Ta, 14) 
    → action_embedding → (B, Ta, 128)
    → temporal_conv (废弃方案，不开启) → (B, T', 128)  # 4倍下采样
    → output_proj → (B, T', 12288)  # 6 * dim
    → mean(dim=1) → (B, 12288)  # 时间聚合
    → reshape → (B, 6, 2048)  # AdaLN参数
```

### 3. VACE条件生成流程

```python
# 1. 创建像素级掩码
pixel_mask[B, 1, T, H, W]  # 1=隐藏(目标), 0=可见(上下文)
pixel_mask[:, :, :To, :, :] = 0  # 前To帧为上下文

# 2. 编码带掩码的视频
z0 = vace_encode_frames(video, mask, vae)  # 潜在特征
m0 = vace_encode_masks(mask, vae_stride)   # 掩码特征

# 3. 拼接为VACE上下文
y = vace_latent(z0, m0)  # 每个样本: [latent; mask]

# 4. 前向传播
hints = forward_vace(x, y, seq_len, kwargs)  # VACE分支
output = forward_main(x, hints, actions, kwargs)  # 主分支
```

---

## 数据处理

### ActionDataset

机器人轨迹数据集加载器，支持滑动窗口采样。

#### 参数配置

| 参数 | 类型 | 说明 |
|------|------|------|
| `data_list` | dict | 包含`results`列表的轨迹数据 |
| `stats_path` | str | 动作归一化统计文件路径 |
| `To` | int | 历史观测帧数 (默认1) |
| `Ta` | int | 未来动作预测步数 (默认16) |
| `stride` | int | 滑动窗口步长 (默认1) |
| `camera_names` | list | 相机名称列表 (4个) |
| `image_size` | tuple | 单相机图像尺寸 (H, W) |

#### 数据流水线

```python
# 1. 轨迹数据格式
trajectory = [
    {
        'observation': {
            'images': {
                'cam_high': np.array(...),      # (H, W, 3) 或 (3, H, W)
                'cam_left_wrist': np.array(...),
                'cam_right_wrist': np.array(...),
                'cam_low': np.array(...)
            },
            'prompt': str  # 任务描述
        },
        'action': [14维数组]  # 机器人动作
    },
    ...
]

# 2. 滑动窗口采样
# 对于To=1, Ta=16的配置：
# - 视频帧: [t-To+1, t+Ta) = [t, t+16)  共16帧
# - 动作: [t, t+Ta) = [t, t+16)  共16步

# 3. 多相机图像拼接 (4相机配置)
#   [cam_high     | cam_left_wrist   ]
#   [cam_low      | cam_right_wrist  ]
# 输出尺寸: (3, 2*H, 2*W) = (3, 480, 640)

# 4. 动作归一化
actions_norm = 2 * ((actions - q01) / (q99 - q01)) - 1  # 归一化到 [-1, 1]
```

#### 输出格式

```python
{
    'video': torch.Tensor,    # (Ta, C, 2*H, 2*W) = (16, 3, 480, 640)
    'actions': torch.Tensor,  # (Ta, action_dim) = (16, 14)
    'prompt': str             # 任务描述文本
}
```

---

## 训练配置

### config.yaml 详解

```yaml
# ========== 模型路径 ==========
checkpoint_dir: "/path/to/Wan2.1-VACE-14B-720P"  # 预训练模型目录
output_dir: "./output_lora"                      # LoRA权重输出目录
resume_checkpoint: ""                            # 恢复训练的checkpoint路径

# ========== 数据配置 ==========
data_path: "/path/to/trajectory_data.pkl"        # 轨迹数据文件
stats_path: "./finetune/robotwin_bbh.json"      # 动作统计信息
To: 1                                            # 历史观测帧数
Ta: 16                                           # 未来动作预测步数
camera_names: ['cam_high', 'cam_left_wrist', 'cam_right_wrist', 'cam_low']
image_size: [240, 320]                           # 单相机分辨率 [H, W]
action_dim: 14                                   # 动作空间维度
action_hidden_dim: 128                           # ActionEncoder隐藏层维度
vace_in_dim: 96                                  # VACE输入通道数
num_workers: 4                                   # DataLoader工作进程数

# ========== 训练超参数 ==========
batch_size: 1                                    # 每GPU批大小
lr: 0.0001                                       # 学习率
epochs: 10                                       # 训练轮数
lora_rank: 16                                    # LoRA秩

# ========== 硬件配置 ==========
mixed_precision: "bf16"                          # 混合精度类型
```



## 训练流程

### 1. 启动训练

```bash
# 单机多卡训练 (推荐)
torchrun --nproc_per_node=4 --master_port=29500 train_vace_lora.py

# 多机多卡训练
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train_vace_lora.py
```

### 2. 训练管线

```python
# 主训练循环
for epoch in range(epochs):
    for batch in dataloader:
        # ========== 数据准备 ==========
        video: (B, C, T, H, W)      # 视频帧
        actions: (B, Ta, 14)        # 动作序列
        prompts: List[str]          # 任务描述
        
        # ========== 编码 (冻结) ==========
        latents = vae.encode(video)              # VAE编码
        context = t5_encoder(prompts)            # 文本编码
        y = vace_encode(video, mask, vae)        # VACE上下文
        
        # ========== Flow Matching 损失 ==========
        t ~ Uniform(0, 1)                        # 随机时间步
        noise ~ N(0, I)                          # 高斯噪声
        x_t = (1-t) * noise + t * latents        # 插值
        target = latents - noise                  # 速度目标
        
        # ========== 前向预测 ==========
        pred = model(
            x=x_t, 
            t=t, 
            vace_context=y,
            context=context,
            actions=actions
        )
        
        # ========== 损失计算 ==========
        loss = MSE(pred, target)
        
        # ========== 反向传播 ==========
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. LoRA配置

```python
lora_config = LoraConfig(
    r=16,                    # LoRA秩
    lora_alpha=32,           # 缩放因子 (通常为2*r)
    target_modules=[
        "q", "k", "v", "o",  # 注意力投影
        "ffn.0", "ffn.2"     # FFN层
    ],
    lora_dropout=0.05,       # Dropout率
    bias="none"              # 不训练bias
)
```

**可训练参数**：
- LoRA矩阵: ~1-5% 总参数量
- ActionEncoder: ~0.5M 参数
- 总计: ~100-200M 参数 (相比14B基础模型)

### 4. FSDP配置

```python
model = FSDP(
    module=model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    auto_wrap_policy=auto_wrap_policy,              # 自动包装策略
    device_id=local_rank,
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,      # 参数精度
        reduce_dtype=torch.bfloat16,     # 梯度规约精度
        buffer_dtype=torch.float32       # 缓冲区精度
    ),
    sync_module_states=True,             # 同步模块状态
    use_orig_params=True                 # 使用原始参数名
)
```

### 5. Checkpoint管理

```python
# 保存 (仅Rank 0)
save_lora_checkpoint(
    model, 
    f"{output_dir}/lora_epoch_{epoch}.pth",
    rank
)

# 保存内容
{
    'base_model.model.q.lora_A': ...,
    'base_model.model.q.lora_B': ...,
    'base_model.model.action_encoder.action_embedding.0.weight': ...,
    'base_model.model.action_encoder.action_embedding.0.bias': ...,
    ...
}

# 加载
load_lora_checkpoint(model, checkpoint_path, rank)
```

---

## 推理使用

### WanActionVace 类

继承自`WanT2V`，增加了动作条件生成能力。

#### 初始化

```python
from finetune.vace_action import WanActionVace

model = WanActionVace(
    config=wan_shared_cfg,
    checkpoint_dir="/path/to/Wan2.1-VACE-14B-720P",
    device_id=0,
    rank=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_usp=False,
    t5_cpu=False
)

# 加载LoRA权重
model.model.load_state_dict(
    torch.load("output_lora/lora_epoch_9.pth"),
    strict=False
)
```

#### 生成流程

```python
# 1. 准备输入
video = prepare_video_frames(...)  # (To, C, H, W) 历史帧
prompt = "Pick up the red block"   # 任务描述
actions = None                     # 推理时可为None或提供引导

# 2. 生成
output = model.generate(
    video=video,
    prompt=prompt,
    actions=actions,
    num_inference_steps=50,
    guidance_scale=7.5
)

# 3. 从生成的视频中提取动作
# (需要额外的动作解码器或逆向求解器)
```

---

## 性能优化

### 内存优化

| 技术 | 说明 | 内存节省 |
|------|------|----------|
| **FSDP** | 完全分片数据并行 | ~75% |
| **LoRA** | 低秩适应 | ~95%可训练参数 |
| **Mixed Precision** | BF16训练 | ~50% |
| **Gradient Checkpointing** | 重计算换内存 | ~30-50% |
| **冻结VAE/T5** | 仅推理模式 | ~40% |

### 训练监控

```bash
# TensorBoard
tensorboard --logdir=output_lora/logs --port=6006

# 监控指标
- Loss/train: 训练损失
- Learning_rate: 学习率曲线
```



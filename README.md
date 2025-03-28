# 陆家嘴人群行为轨迹模拟：Qwen2.5-7B-Instruct模型微调

## 项目概述
本项目对Qwen2.5-7B-Instruct模型进行微调，使其能够根据指令生成上海市陆家嘴金融区人群的日常行为轨迹。通过结构化的训练数据，模型可以输出包含时空信息、活动内容和主观评价的完整轨迹描述。

## 1. 实验环境

### 1.1 硬件环境
* 开发环境：Apple Silicon M1
* 训练环境：autodl服务器，RTX 4090 (24GB)，Ubuntu 22.04，PyTorch 2.1.0，CUDA 12.1

### 1.2 环境配置
```bash
# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct

# 部署LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

## 2. 数据特点
训练数据集采用指令微调格式，以Alpaca格式为准，包含人群画像、出行活动链和主观评价三部分：

数据集结构符合标准JSON格式：
```json
{
  "instruction": "请基于真实世界信息，生成一个在上海市陆家嘴区域内进行活动的人，在某一典型工作日内的完整活动轨迹信息...",
  "input": "请生成陆家嘴内一位人士的一天活动轨迹。",
  "output": "# 此人个体基本信息\n\t[陆家嘴活动人群画像]:商务人群\n\t[年龄]: 30岁\n..."
}
```

## 3. 微调参数设置
采用LoRA方法进行高效参数微调：

```yaml
# 核心参数
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
learning_rate: 5.0e-05
num_train_epochs: 10.0
per_device_train_batch_size: 1
gradient_accumulation_steps: 5
max_grad_norm: 1.0
```

## 4. 推理应用
训练完成后使用简单的推理脚本进行测试：

```python
def generate_response(instruction, input_text=""):
    prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(generation_output[0], skip_special_tokens=False)
    assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return assistant_response.strip()
```

## 5. 评估指标
模型评估主要从以下几个维度进行：

* **行为连贯性**：生成轨迹的时空逻辑是否合理
* **格式规范性**：输出是否符合预期的格式结构
* **坐标准确性**：生成的坐标点是否落在陆家嘴区域内
* **人物一致性**：人物画像与行为模式是否匹配

## 6. 后续改进
* 增加更多样化的人群类型数据
* 优化坐标点精确度
* 融合多模态信息以提升真实感
* 开发API接口便于应用集成

## 7. 小结
本项目通过对Qwen2.5-7B-Instruct模型的微调，实现了针对陆家嘴地区人群行为轨迹的模拟生成。基于LoRA技术，在有限的计算资源条件下，模型可以生成符合现实逻辑的人群活动轨迹描述，为城市研究、商业分析等场景提供数据支持。

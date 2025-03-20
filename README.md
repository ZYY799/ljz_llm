# 微调[`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

## 实验环境

### 硬件环境

- Apple Silicon M1
- Ubuntu 服务器
    PyTorch 2.1.0
    Python 3.10 (Ubuntu 22.04)
    CUDA 12.1
    RTX 4090 (24GB) × 1
    CPU: 12 vCPU Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz




```bas
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com  # （可选）配置 hf 国内镜像站
huggingface-cli download --resume-download Qwen/Qwen2.5-7B-Instruct --local-dir /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

  ```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
  ```

```bash
cd LLaMA-Factory
llamafactory-cli webui
```


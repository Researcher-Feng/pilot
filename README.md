# 项目：DEMO



## 安装和使用

核心依赖库：

```
hydra-core==1.4.0.dev1
omegaconf==2.4.0.dev3
openai==1.109.1
langchain==1.0.3
langchain-core==1.0.2
langchain-deepseek==1.0.0
langchain-ollama==1.0.0
langchain-openai==1.0.1
```

配置环境：

```bash
conda create -n pilot python=3.12
conda activate pilot
pip install -r requirements.txt
```

执行测试：

```bash
# 先配置run.py的路径和选项
cd v5
python run.py
```



## 版本历史

- v1: 初始版本
- v2: 核心增强：补充agent的多轮对话逻辑
- v3: 核心增强：支持本地Ollama模型调用
- v4: 核心增强：补充更多的功能选择，如对话摘要
- v5: 核心增强：可通过多轮对话实现初步效果
- v6
- v7
- v8
- v9: 当前（2025.11.13）稳定版本

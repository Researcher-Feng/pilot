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
- v5: 核心增强：可通过多轮对话实现初步效果，当前（2025.11.6）稳定版本

./v5
├── function
│   ├── agent_IO.py
│   ├── contex_fun.py
│   ├── format_fun.py
│   ├── __init__.py
│   ├── memory_fun.py
│   ├── __pycache__
│   │   ├── agent_IO.cpython-310.pyc
│   │   ├── agent_IO.cpython-312.pyc
│   │   ├── contex_fun.cpython-310.pyc
│   │   ├── contex_fun.cpython-312.pyc
│   │   ├── format_fun.cpython-310.pyc
│   │   ├── format_fun.cpython-312.pyc
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-312.pyc
│   │   ├── memory_fun.cpython-310.pyc
│   │   ├── memory_fun.cpython-312.pyc
│   │   ├── tools_fun.cpython-310.pyc
│   │   └── tools_fun.cpython-312.pyc
│   └── tools_fun.py
├── prompt
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-310.pyc
│   │   ├── __init__.cpython-312.pyc
│   │   ├── system.cpython-310.pyc
│   │   └── system.cpython-312.pyc
│   └── system.py
├── __pycache__
│   └── run.cpython-312-pytest-8.4.2.pyc
├── README.md
├── results
│   ├── math_tutoring_explicit_interaction_20251105_124638_plots.png
│   └── math_tutoring_explicit_interaction_20251105_131550_plots.png
├── run1.py
├── run.py
├── tests
│   └── test_modules.py
└── utils
    ├── api_config.py
    ├── dataset
    │   ├── fs.py
    │   ├── hdfs_io.py
    │   ├── __init__.py
    │   ├── parallel_thinking_sft_dataset.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── __init__.cpython-312.pyc
    │   │   ├── parallel_thinking_sft_dataset.cpython-310.pyc
    │   │   └── parallel_thinking_sft_dataset.cpython-312.pyc
    │   └── tokenizer.py
    ├── evaluator.py
    ├── __init__.py
    ├── MARIO_EVAL
    │   ├── data
    │   │   └── math_testset_annotation.json
    │   ├── demo.py
    │   ├── eval_server
    │   │   ├── batch_eval.py
    │   │   ├── eval_gunicorn.py
    │   │   ├── gunicorn_config.py
    │   │   ├── __init__.py
    │   │   ├── launch_server.sh
    │   │   ├── ray_utils.py
    │   │   └── README.md
    │   ├── example_tora_eval.py
    │   ├── __init__.py
    │   ├── latex2sympy
    │   │   ├── antlr-4.11.1-complete.jar
    │   │   ├── asciimath_printer.py
    │   │   ├── build
    │   │   │   ├── bdist.win-amd64
    │   │   │   ├── __init__.py
    │   │   │   └── lib
    │   │   │       ├── asciimath_printer.py
    │   │   │       ├── gen
    │   │   │       │   ├── __init__.py
    │   │   │       │   ├── PSLexer.py
    │   │   │       │   ├── PSListener.py
    │   │   │       │   └── PSParser.py
    │   │   │       ├── __init__.py
    │   │   │       ├── latex2sympy2.py
    │   │   │       └── tests
    │   │   │           ├── abs_test.py
    │   │   │           ├── all_bad_test.py
    │   │   │           ├── all_good_test.py
    │   │   │           ├── atom_expr_test.py
    │   │   │           ├── binomial_test.py
    │   │   │           ├── ceil_test.py
    │   │   │           ├── complex_test.py
    │   │   │           ├── context.py
    │   │   │           ├── exp_test.py
    │   │   │           ├── floor_test.py
    │   │   │           ├── gcd_test.py
    │   │   │           ├── greek_test.py
    │   │   │           ├── grouping_test.py
    │   │   │           ├── __init__.py
    │   │   │           ├── lcm_test.py
    │   │   │           ├── left_right_cdot_test.py
    │   │   │           ├── linalg_test.py
    │   │   │           ├── max_test.py
    │   │   │           ├── min_test.py
    │   │   │           ├── mod_test.py
    │   │   │           ├── overline_test.py
    │   │   │           ├── pi_test.py
    │   │   │           ├── trig_test.py
    │   │   │           └── variable_test.py
    │   │   ├── description.txt
    │   │   ├── dev-requirements.in
    │   │   ├── dev-requirements.txt
    │   │   ├── gen
    │   │   │   ├── __init__.py
    │   │   │   ├── PS.interp
    │   │   │   ├── PSLexer.interp
    │   │   │   ├── PSLexer.py
    │   │   │   ├── PSLexer.tokens
    │   │   │   ├── PSListener.py
    │   │   │   ├── PSParser.py
    │   │   │   ├── PS.tokens
    │   │   │   └── __pycache__
    │   │   │       ├── __init__.cpython-310.pyc
    │   │   │       ├── __init__.cpython-312.pyc
    │   │   │       ├── PSLexer.cpython-310.pyc
    │   │   │       ├── PSLexer.cpython-312.pyc
    │   │   │       ├── PSListener.cpython-310.pyc
    │   │   │       ├── PSListener.cpython-312.pyc
    │   │   │       ├── PSParser.cpython-310.pyc
    │   │   │       └── PSParser.cpython-312.pyc
    │   │   ├── gen_orig
    │   │   │   ├── __init__.py
    │   │   │   ├── PSLexer.interp
    │   │   │   ├── PSLexer.py
    │   │   │   ├── PSLexer.tokens
    │   │   │   ├── PS_orig.interp
    │   │   │   ├── PS_origLexer.interp
    │   │   │   ├── PS_origLexer.py
    │   │   │   ├── PS_origLexer.tokens
    │   │   │   ├── PS_origListener.py
    │   │   │   ├── PS_origParser.py
    │   │   │   └── PS_orig.tokens
    │   │   ├── icon.png
    │   │   ├── __init__.py
    │   │   ├── latex2sympy2.egg-info
    │   │   │   ├── dependency_links.txt
    │   │   │   ├── PKG-INFO
    │   │   │   ├── requires.txt
    │   │   │   ├── SOURCES.txt
    │   │   │   └── top_level.txt
    │   │   ├── latex2sympy2_orig.py
    │   │   ├── latex2sympy2.py
    │   │   ├── LICENSE.txt
    │   │   ├── PS.g4
    │   │   ├── PS_orig.g4
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-310.pyc
    │   │   │   ├── __init__.cpython-312.pyc
    │   │   │   ├── latex2sympy2.cpython-310.pyc
    │   │   │   └── latex2sympy2.cpython-312.pyc
    │   │   ├── README.md
    │   │   ├── README_orig.md
    │   │   ├── requirements.in
    │   │   ├── requirements.txt
    │   │   ├── sandbox
    │   │   │   ├── __init__.py
    │   │   │   ├── linalg_equations.py
    │   │   │   ├── linalg_span.py
    │   │   │   ├── matrix_placeholders.py
    │   │   │   ├── matrix.py
    │   │   │   ├── sandbox_equality.py
    │   │   │   ├── sandbox.py
    │   │   │   ├── sectan.py
    │   │   │   └── vector.py
    │   │   ├── scripts
    │   │   │   ├── compile_orig.sh
    │   │   │   ├── compile.sh
    │   │   │   ├── coverage-ci.sh
    │   │   │   ├── coverage.sh
    │   │   │   ├── pre-commit
    │   │   │   ├── pre-push
    │   │   │   ├── publish.sh
    │   │   │   ├── setup-hooks.sh
    │   │   │   ├── setup.sh
    │   │   │   └── test.sh
    │   │   ├── setup.cfg
    │   │   ├── setup.py
    │   │   └── tests
    │   │       ├── abs_test.py
    │   │       ├── all_bad_test.py
    │   │       ├── all_good_test.py
    │   │       ├── atom_expr_test.py
    │   │       ├── binomial_test.py
    │   │       ├── ceil_test.py
    │   │       ├── complex_test.py
    │   │       ├── context.py
    │   │       ├── exp_test.py
    │   │       ├── floor_test.py
    │   │       ├── gcd_test.py
    │   │       ├── greek_test.py
    │   │       ├── grouping_test.py
    │   │       ├── __init__.py
    │   │       ├── lcm_test.py
    │   │       ├── left_right_cdot_test.py
    │   │       ├── linalg_test.py
    │   │       ├── max_test.py
    │   │       ├── min_test.py
    │   │       ├── mod_test.py
    │   │       ├── overline_test.py
    │   │       ├── pi_test.py
    │   │       ├── trig_test.py
    │   │       └── variable_test.py
    │   ├── llm_type.py
    │   ├── math_evaluation
    │   │   ├── core
    │   │   │   ├── constants.py
    │   │   │   ├── evaluations.py
    │   │   │   ├── __init__.py
    │   │   │   ├── latex_normalize.py
    │   │   │   ├── latex_parser.py
    │   │   │   ├── __pycache__
    │   │   │   │   ├── constants.cpython-310.pyc
    │   │   │   │   ├── constants.cpython-312.pyc
    │   │   │   │   ├── evaluations.cpython-310.pyc
    │   │   │   │   ├── evaluations.cpython-312.pyc
    │   │   │   │   ├── __init__.cpython-310.pyc
    │   │   │   │   ├── __init__.cpython-312.pyc
    │   │   │   │   ├── latex_normalize.cpython-310.pyc
    │   │   │   │   ├── latex_normalize.cpython-312.pyc
    │   │   │   │   ├── latex_parser.cpython-310.pyc
    │   │   │   │   ├── latex_parser.cpython-312.pyc
    │   │   │   │   ├── type_evaluations.cpython-310.pyc
    │   │   │   │   └── type_evaluations.cpython-312.pyc
    │   │   │   └── type_evaluations.py
    │   │   ├── __init__.py
    │   │   ├── llms
    │   │   │   ├── __init__.py
    │   │   │   ├── llm_models.py
    │   │   │   └── prompts.py
    │   │   ├── __pycache__
    │   │   │   ├── __init__.cpython-310.pyc
    │   │   │   └── __init__.cpython-312.pyc
    │   │   └── tests
    │   │       ├── __init__.py
    │   │       └── test_is_equiv.py
    │   ├── math_evaluation.egg-info
    │   │   ├── dependency_links.txt
    │   │   ├── PKG-INFO
    │   │   ├── requires.txt
    │   │   ├── SOURCES.txt
    │   │   └── top_level.txt
    │   ├── __pycache__
    │   │   ├── demo.cpython-310.pyc
    │   │   ├── demo.cpython-312.pyc
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── __init__.cpython-312.pyc
    │   ├── README.md
    │   ├── requirements.txt
    │   └── setup.py
    ├── __pycache__
    │   ├── api_config.cpython-310.pyc
    │   ├── api_config.cpython-312.pyc
    │   ├── evaluator.cpython-310.pyc
    │   ├── evaluator.cpython-312.pyc
    │   ├── __init__.cpython-310.pyc
    │   └── __init__.cpython-312.pyc
    ├── sft_trainer.yaml
    └── visualization_tools.py
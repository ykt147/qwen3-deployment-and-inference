# SA25006112_杨开泰_lab1


## Requirements

```
Python: 3.13.5
PyTorch: 2.8.0（CUDA 12.8 支持）
Transformers: 4.57.0
```

## Usage

使用modelscope下载你需要的模型
```
```


Step 1: 
使用提供的 environment.yml 文件创建并配置 Conda 环境：
```
conda env create -f environment.yml
```

Step 2: 
安装完成后，激活名为 mls1 的虚拟环境：

```
conda activate mls1
```

Step 3: 
修改模型路径

```
# 在 infer.py 和 benchmark.py 中修改

model_path = "/path/to/your/local/model/folder"
```

Step 4: 
运行脚本
```
python infer.py
python benchmark.py
```




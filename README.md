# L3C_Baselines
Implementations of baselines for [L3C](https://github.com/FutureAGI/L3C)

Transformers are used as the backbone to modeling Meta-Langauge, Decision Models, World Models.

# Training and Evaluating
To train a model run
```bash
cd L3C_Baselines/demo/xxx
python train.py configuration.yaml --configs key1=value1 key2=value2 ...
```

To evaluate a model run
```bash
cd L3C_Baselines/demo/xxx
python evaluate.py configuration.yaml --configs key1=value1 key2=value2 ...
```

Go to L3C_Baselines/data to generate the datasets directly by specifying some configurations

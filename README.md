### Environment

Require Python 3.12

To run this project, please first install all required packages:

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```
pip install -r requirements.txt
```



### Preprocessing

Then, we need to preprocess the datasets,

for standard KG completion:

```
python preprocess.py -c config/fb15k237.yaml

python preprocess.py -c config/wn18rr.yaml
```

for inductive setting:

```
python preprocess.py -c config/fb15k237_ind.yaml --version v1

python preprocess.py -c config/wn18rr_ind.yaml --version v1
```

### Run with Single GPU

If you only has one GPU (better has 80GB memory under the default setting), please run the model with the following command:

```
python main.py -c config/fb15k237.yaml
```

### Run with Multiple GPU

If you can access multiple GPUs, please run the model with the following command:

```
accelerate launch --gpu_ids 'all' --num_processes 8 --mixed_precision bf16 main.py -c config/fb15k237.yaml
```

### Run with script

Please kindly use the provide scripts to run the model:

```
sh scripts/fb15k237.sh
```

### Cite

Please condiser citing our paper if it is helpful to your work!

```bigquery
@inproceedings{MKGL,
  author       = {Lingbing Guo and
                  Zhongpu Bo and 
                  Zhuo Chen and 
                  Yichi Zhang and
                  Jiaoyan Chen and
                  Lan Yarong and
                  Mengshu Sun and
                  Zhiqiang Zhang and
                  Yangyifei Luo and
                  Qian Li and
                  Qiang Zhang and
                  Wen Zhang and
                  Huajun Chen},
  title        = {MGKL: Mastery of a Three-Word Language},
  booktitle    = {{NeurIPS}},
  year         = {2024}
}
```

### Thanks

We appreciate [LLaMA](https://github.com/facebookresearch/llama), [Huggingface Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora), and many other related works for their open-source contributions.


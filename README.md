# Fact embedding through diffusion model
Implementation code for the knowledge graph embedding diffusion model based on fact triple modeling, [Fact embedding through diffusion model for knowledge graph completion].

<img src="framework.png" width = "800" />

## Environment Configuration
1、Hardware Requirements: NVIDIA GPU environment required. CUDA version ≥ 12.4, driver version ≥ 550. VRAM requirement ≥ 20GB.

2、Install dependencies: Execute the following command:
``` 
pip install -r requirements.txt
```


## Model Training
The algorithm has encapsulated positive and negative sample sampling processes. To train the embeddings, run: sh scripts/run.sh
```bash
 CUDA_VISIBLE_DEVICES=7 python train_fdm.py --cuda --dataset FB15K-237 --do_train --do_valid --do_test \
  --data_path ../data/FB15K-237  -b 512 -d 400 -g 28 \
  -a 0.5 -adv --modelconfig "../model_configs/Tnet.yaml" \
  -lr 0.00008 --max_steps 250000 --dataset_neg -n 256 \
  -save '../models/FB15K-237-Tnet' --test_batch_size 20 --use_ensemble --pretrain_emb \
  --exp_info "xxxxx" 
```

## Model Inference
For trained embeddings, run: sh scripts/eval.sh for inference
```bash
 CUDA_VISIBLE_DEVICES=7 python train_fdm.py --cuda --dataset FB15K-237 --do_test \
  --data_path ../data/FB15K-237  -b 512 -d 400 -g 28 \
  -a 0.5 -adv --modelconfig "../model_configs/Tnet.yaml" \
  -lr 0.00008 --max_steps 250000 --dataset_neg -n 256 \
  -save '../models/FB15K-237-Tnet' --test_batch_size 20 --use_ensemble --pretrain_emb \
  --exp_info "xxxxx" 

```


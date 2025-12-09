#  CUDA_VISIBLE_DEVICES=0 python train_fdm.py --cuda --dataset FB15K-237 --do_train --do_valid --do_test \
#   --data_path ./data/FB15K-237  -b 1024 -d 400 -g 9.0 \
#   -a 0.5 -adv \
#   -lr 0.00008 --max_steps 500000 \
#   -save models/FB15K-237-Tnet --test_batch_size 8 \
#   --exp_info "固定初始的嵌入, 观察能否还原transe的效果, 并输出测试阶段的还原效果, 并输出训练阶段的分类效果, multihot, 采样步数10 eta1., loss加权, 分类器加condition"
# #   --dataset_onehot



#  CUDA_VISIBLE_DEVICES=0 python train_fdm.py --cuda --dataset FB15K-237 --do_train --do_valid --do_test \
#   --data_path ./data/FB15K-237  -b 1024 -d 400 -g 9.0 \
#   -a 0.5 -adv --modelconfig "./model_configs/Conv.yaml"\
#   -lr 0.0008 --max_steps 500000 \
#   -save models/FB15K-237-Conv --test_batch_size 8 \
#   --exp_info "固定初始的嵌入, 观察能否还原transe的效果, 并输出测试阶段的还原效果, 并输出训练阶段的分类效果, multihot, 采样步数10 eta1., loss加权, 分类器加condition, conv做编码器"
# #   --dataset_onehot



 CUDA_VISIBLE_DEVICES=7 python train_fdm.py --cuda --dataset FB15K-237 --do_test \
  --data_path ../data/FB15K-237  -b 512 -d 400 -g 28 \
  -a 0.5 -adv --modelconfig "../model_configs/Tnet.yaml" \
  -lr 0.00008 --max_steps 250000 --dataset_neg -n 256 \
  -save '../models/FB15K-237-Tnet' --test_batch_size 20 --use_ensemble --pretrain_emb \
  --exp_info "--no pretrain_emb grad f dit(2) conv(400) in channel leaky relu(time) ddpm 40ensemble mlp4" 
  # --dataset_onehot --pretrain_emb --use_ensemble --dataset_onehot --use_ensemble
# 
# --init '/workspace/longxiao/diffusion-code/Conditional_DM_KGE/models/FB15K-237-Tnet/2023-04-09 13:44:51/best/best_checkpoint'
# --init '/workspace/longxiao/diffusion-code/Conditional_DM_KGE/models/FB15K-237-Tnet/2023-04-09 13:44:51/best/best_checkpoint'\

#  CUDA_VISIBLE_DEVICES=1 python train_fdm.py --cuda --dataset YAGO3-10 --do_train --do_valid --do_test \
#   --data_path ./data/YAGO3-10  -b 1024 -d 400 -g 9.0 \
#   -a 0.5 -adv --modelconfig "./model_configs/Tnet.yaml"\
#   -lr 0.0008 --max_steps 500000 \
#   -save models/YAGO3-10-Tnet --test_batch_size 8 --dataset_onehot --pretrain_emb\
#   --exp_info " ddim Tnet eta 0 无分类器编码器 --dataset_onehot --pretrain_emb required grad False mask [0, 0, 1]"


#    CUDA_VISIBLE_DEVICES=1 python train_fdm.py --cuda --dataset wn18rr --do_train --do_valid --do_test \
#   --data_path ./data/wn18rr  -b 1024 -d 400 -g 9.0 \
#   -a 0.5 -adv --modelconfig "./model_configs/Tnet.yaml"\
#   -lr 0.0008 --max_steps 500000 \
#   -save models/wn18rr-Tnet --test_batch_size 8 --dataset_onehot --pretrain_emb\
#   --exp_info " ddim Tnet eta 0 无分类器编码器 --dataset_onehot --pretrain_emb required grad False mask [0, 0, 1]"
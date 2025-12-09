import torch
import numpy as np
from functools import partial
from torch import optim
from torch.utils.data import DataLoader
from datasets import *
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import logging
from utils import *
import pdb
from models import *
from multiprocessing import Pool
from torch.autograd import Variable
# import multiprocessing
# multiprocessing.set_start_method('spawn')

class DDPMKge(nn.Module):
    """
    扩散KG embedding模型
    """
    def __init__(self, 
            dataset,
            denoise_model,
            # device,
            nentity,
            nrelation,
            hidden_dim,
            gamma,
            pretrain_emb,
            objective='pred_xtart',
            double_entity_embedding=False,
            double_relation_embedding=False,
            timesteps=200,
            beta_schedule="linear",
            linear_start=1e-4,
            linear_end=2e-2,
            max_seq_len=3,
            loss_type="l2",
            lr=1e-5,
            ddim_sampling_eta=1.,
            use_ensemble = False,
            ddim_sampling_timesteps=10,
            dataset_onehot=False,
            dataset_neg=False,
        ):
        super().__init__()
        # self.device = device
        ########## kge init para ############
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.pretrain_emb = pretrain_emb
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        if self.pretrain_emb:
            if dataset == 'NELL':
                self.entity_embedding = nn.Parameter(torch.from_numpy(np.load( \
                    f'/workspace/longxiao/diffusion-code/Conditional_DM_KGE/data/{dataset}/entity_embedding_400.npy')).float(), requires_grad=False)
                
                self.relation_embedding = nn.Parameter(torch.from_numpy(np.load( \
                    f'/workspace/longxiao/diffusion-code/Conditional_DM_KGE/data/{dataset}/relation_embedding_400.npy')).float(), requires_grad=False)
            
            if dataset == 'wn18rr':
                self.entity_embedding = nn.Parameter(torch.from_numpy(np.load( \
                    f'/workspace/longxiao/diffusion-code/Conditional_DM_KGE/data/{dataset}/entity_embedding_400.npy')).float(), requires_grad=False)
                
                self.relation_embedding = nn.Parameter(torch.from_numpy(np.load( \
                    f'/workspace/longxiao/diffusion-code/Conditional_DM_KGE/data/{dataset}/relation_embedding_400.npy')).float(), requires_grad=False)
            
            if dataset == 'FB15K-237':
                self.entity_embedding = nn.Parameter(torch.from_numpy(np.load( \
                    f'/workspace/longxiao/diffusion-code/Conditional_DM_KGE/data/{dataset}/entity_embedding_400.npy')).float(), requires_grad=True)
                
                self.relation_embedding = nn.Parameter(torch.from_numpy(np.load( \
                    f'/workspace/longxiao/diffusion-code/Conditional_DM_KGE/data/{dataset}/relation_embedding_400.npy')).float(), requires_grad=True)
        # init随机初始化
        else: 
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.entity_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        
            nn.init.uniform_(
                tensor=self.relation_embedding, 
                a=-self.embedding_range.item(), 
                b=self.embedding_range.item()
            )
        ########## kge init para ############

        ########## ddpm init para ############
        self.denoise_model = denoise_model.cuda()
        # pdb.set_trace()
        self.timesteps = timesteps
        self.objective = objective
        self.ddim_sampling_eta = ddim_sampling_eta
        self.register_schedule(beta_schedule=beta_schedule, linear_start=linear_start, linear_end=linear_end)
        self.loss_type = loss_type
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.use_ensemble = use_ensemble
        self.sampling_timesteps = ddim_sampling_timesteps
        self.dataset_onehot = dataset_onehot
        self.dataset_neg = dataset_neg
        ########## ddpm init para ############

    def register_schedule(self, beta_schedule, linear_start, linear_end):
        # define beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(linear_start, linear_end, timesteps=self.timesteps)
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.timesteps)
        else:
            raise NotImplementedError("Not supported beta_schedule.")

        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0) beta * (1 - alphas_cumprod_prev) / 
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others  
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # 这个用来计算ddim里的噪声

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', posterior_variance)

    def similarity(self, emb_list):
        for i in range(len(emb_list)-1):
            j = i + 1
            pdb.set_trace()
            cos_sim_i_j = F.cosine_similarity(emb_list[i], emb_list[j], dim=-1)
            print("cos_sim_i_j", cos_sim_i_j)
            l2 = ((emb_list[i] - emb_list[j])**2).sum()
            print("square_sim", l2)


    def forward(self, sample, mode, if_train=False):
        '''
        要解决train 和 test两个过程
        train: data mode: tail, head; pos: (h, r, t) batch * 3,
        
        test:

        Forward function that calculate the score of a batch of triples.

        :param: mode: head-batch, tail-batch
        :param: sample: (positive_sample, positive_label) positive_sample: batch * 3 [[h, r, t], ... ]
        'single': mode, sample is a batch of triple. This is the positive sample mode ([[252, 1, 5843], ..., ])
        'head-batch' or 'tail-batch': sample consists two part. This is the negative sample mode
        
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        ''' 
        #  
        if mode == 'head-batch':
            # 通过 r t => h
            # tail_part: batch * 3 - > 1d tail: batch * 3 正样本 (h, r, t)
            # head_part: batch * neg_num 都是负样本的头实体
            # pdb.set_trace()
            positive_sample, negative_sample, subsampling_weight, mode = sample
            negative_sample_size = negative_sample.size(1)
            embedding_dim = self.entity_embedding.size(-1)
            batch_size= positive_sample.size(0)
            
            head = torch.index_select(self.entity_embedding, 
                                      dim=0, 
                                      index=positive_sample[:, 0]) # batch * dim
            relation = torch.index_select(self.relation_embedding,
                                          dim=0, index=positive_sample[:, 1]) # batch * dim 
            # 这是一种path不止一跳的情况, 下同: b * len * dim
            # relation = torch.index_select(self.relation_embedding,
            #                               dim=0, index=positive_sample[:, 1]).unsqueeze(1) # batch * 1 * dim     
            
            tail = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 2]) # batch * dim
            # pdb.set_trace()
            t = torch.randint(0, self.timesteps, (batch_size,)).cuda()
            # pdb.set_trace()


            y0_head = tail - relation
            
            triplet_emb = torch.cat((head.unsqueeze(1), relation.unsqueeze(1), tail.unsqueeze(1)), dim=1) # b * 3 * dim
            positive_labels_emb = triplet_emb

            if if_train == True:
                negative_sample_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                negative_sample_emb = torch.cat((negative_sample_emb.unsqueeze(2), relation.unsqueeze(1).repeat(1, negative_sample_size, 1).unsqueeze(2),
                                                 tail.unsqueeze(1).repeat(1, negative_sample_size, 1).unsqueeze(2)), dim=2) # b * 256 * 3 * dim
                # pdb.set_trace()
                loss, positive_sample_loss, negative_sample_loss, metrics = self.cal_losses(denoise_model=self.denoise_model,
                                       input = (triplet_emb, positive_labels_emb, negative_sample_emb, subsampling_weight),
                                       condition=(relation, tail, y0_head), 
                                       mode=mode,  
                                       t=t, 
                                       denoise_loss_type=self.loss_type)
                # pdb.set_trace()
                if metrics is not None:
                    return loss, positive_sample_loss, negative_sample_loss, metrics
                else:
                    return loss, positive_sample_loss, negative_sample_loss
                # return loss, denoise_loss, classifier_loss, metrics

            if if_train == False:
                "The test period"
                # pre_embed = self.p_sample_loop(condition=(relation, tail), 
                #                                time_steps=self.timesteps,
                #                                noise_shape=(batch_size, self.max_seq_len, head.shape[-1]))
                negative_sample_head_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                if self.use_ensemble:
                    pre_head_embed_list = []
                    score_list = []
                    iter = 20
                    # negative_sample_emb = torch.index_select(self.entity_embedding,
                    #                                          dim=0,
                    #                                          index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                    # arg_lists = [{'condition':(relation, tail, y0_head), 
                    #                'shape':(batch_size, embedding_dim),
                    #                'mode':mode} for i in range(iter)]
                    
                    
                    # pool = Pool(processes=10)
                    # pre_embed_list = pool.map(self.ddim_sample, arg_lists)
                    # pool.close()
                    # pool.join()
                    
                    # for pre_embed in pre_embed_list:
                    #     score = torch.norm(negative_sample_emb - pre_embed.unsqueeze(1), p=1, dim=-1)
                    #     score_list.append(score)

                    for i in range(iter):
                        pre_embed = self.ddim_sample({'condition':(relation, tail, y0_head), 
                                                      'shape':(batch_size, 3, embedding_dim),
                                                      'mode':mode})
                        # pdb.set_trace()
                        pred_head_emb = pre_embed[:, 0, :]
                        pre_head_embed_list.append(pred_head_emb)
                        score = torch.norm(negative_sample_head_emb - pred_head_emb.unsqueeze(1), p=1, dim=-1)
                        score_list.append(score)

                    # pdb.set_trace()
                    
                    # self.similarity(pre_embed_list)
                    mean_pre_embed = sum(pre_head_embed_list)/len(pre_head_embed_list)
                    score2 = torch.norm(negative_sample_head_emb - mean_pre_embed.unsqueeze(1), p=1, dim=-1) # b * 14541 cheak

                    return score_list, score2
            
                else:
                    # pdb.set_trace()

                    pre_embed = self.ddim_sample({'condition':(relation, tail, y0_head), 
                                                  'shape':(batch_size, 3, embedding_dim),
                                                  'mode':mode})

                    # pre_embed = self.p_sample_loop(condition=(relation, tail, y0_head), 
                    #                                shape=(batch_size, embedding_dim),
                    #                                mode=mode)                    
                    
                    # pdb.set_trace()
                    # print('1')
                # pdb.set_trace()
                score = torch.norm(negative_sample_head_emb - pre_embed.unsqueeze(1), p=1, dim=-1) # b * 14541
                score2 = torch.norm(negative_sample_head_emb - y0_head.unsqueeze(1), p=1, dim=-1) # b * 14541 cheak
                denoise_loss = F.smooth_l1_loss(positive_labels_emb, pre_embed)
                return score, score2, denoise_loss.mean(), (relation, tail)
                # 计算相似度和负样本


        elif mode == 'tail-batch':
            # 通过 h r => t 预测tail
            positive_sample, negative_sample, subsampling_weight, mode = sample
            negative_sample_size = negative_sample.size(1)
            embedding_dim = self.entity_embedding.size(-1)
            batch_size= positive_sample.size(0)
            
            head = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 0])# batch * dim
            relation = torch.index_select(self.relation_embedding,
                                          dim=0, index=positive_sample[:, 1]) # batch * dim
            tail = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 2])# batch * dim
            t = torch.randint(0, self.timesteps, (batch_size,)).cuda()
            
            # positive_labels_emb = tail
            # pdb.set_trace()
            y0_tail = head + relation   
            triplet_emb = torch.cat((head.unsqueeze(1), relation.unsqueeze(1), tail.unsqueeze(1)), dim=1)
            positive_labels_emb = triplet_emb        
            
            if if_train == True:
                negative_sample_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                # pdb.set_trace()
                negative_sample_emb = torch.cat((head.unsqueeze(1).repeat(1, negative_sample_size, 1).unsqueeze(2), 
                                                 relation.unsqueeze(1).repeat(1, negative_sample_size, 1).unsqueeze(2),
                                                 negative_sample_emb.unsqueeze(2)), dim=2) # [512, 256, 3, 400]

                # pdb.set_trace()
                loss, positive_sample_loss, negative_sample_loss, metrics = self.cal_losses(denoise_model=self.denoise_model, 
                                       input=(triplet_emb, positive_labels_emb, negative_sample_emb, subsampling_weight),
                                       condition=(head, relation, y0_tail), 
                                       mode=mode,
                                       t=t, 
                                       denoise_loss_type=self.loss_type)
                
                if metrics is not None:
                    return loss, positive_sample_loss, negative_sample_loss, metrics
                else:
                    return loss, positive_sample_loss, negative_sample_loss

            else:
                # pre_embed = self.p_sample_loop(condition=(head, relation), 
                #                                time_steps=self.timesteps,
                #                                noise_shape=(batch_size, self.max_seq_len, head.shape[-1]))
                negative_sample_tail_emb = torch.index_select(self.entity_embedding,
                                                         dim=0,
                                                         index=negative_sample.view(-1)).view(batch_size, negative_sample_size, -1)
                if self.use_ensemble:
                    pre_tail_embed_list = []
                    score_list = []
                    iter = 20
                    # arg_lists = [{'condition':(head, relation, y0_tail), 
                    #                'shape':(batch_size, embedding_dim),
                    #                'mode':mode} for i in range(iter)]
                    
                    # pool = Pool(processes=10)
                    # pre_embed_list = pool.map(self.ddim_sample, arg_lists)
                    # pool.close()
                    # pool.join()
                    
                    # for pre_embed in pre_embed_list:
                    #     score = torch.norm(negative_sample_emb - pre_embed.unsqueeze(1), p=1, dim=-1)
                    #     score_list.append(score)

                    for i in range(iter):
                        pre_embed = self.ddim_sample({'condition':(head, relation, y0_tail), 
                                                      'shape':(batch_size, 3, embedding_dim),
                                                      'mode':mode})
                        # pdb.set_trace()
                        pred_tail_emb = pre_embed[:, 2, :]
                        pre_tail_embed_list.append(pred_tail_emb)
                        score = torch.norm(negative_sample_tail_emb - pred_tail_emb.unsqueeze(1), p=1, dim=-1)
                        score_list.append(score)


                    mean_pre_embed = sum(pre_tail_embed_list)/len(pre_tail_embed_list)
                    score2 = torch.norm(negative_sample_tail_emb - mean_pre_embed.unsqueeze(1), p=1, dim=-1) # b * 14541 cheak

                    return score_list, score2

                else:

                    pre_embed = self.ddim_sample({'condition':(head, relation, y0_tail), 
                                                  'shape':(batch_size, 3, embedding_dim),
                                                  'mode':mode})
                
                    # pre_embed = self.p_sample_loop(condition=(head, relation, y0_tail), 
                    #                              shape=(batch_size, embedding_dim),
                    #                              mode=mode)
                # pdb.set_trace()
                score = torch.norm(negative_sample_tail_emb - pre_embed.unsqueeze(1), p=1, dim=-1) 
                score2 = torch.norm(negative_sample_tail_emb - y0_tail.unsqueeze(1), p=1, dim=-1) 
                denoise_loss = F.smooth_l1_loss(positive_labels_emb, pre_embed)
                return score, score2, denoise_loss.mean(), (head, relation)
        else:
            raise ValueError('mode %s not supported' % mode)


    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )        

    def predict_start_from_noise(self, x_t, t, noise):
        return(
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_sample(self, 
                 input_start, 
                 t, 
                 noise=None):
        """
        在输入数据中进行加噪, 返回任意时刻的噪声数据 x(t) = sqrt(alphas_cumprod_t)*x_0 + sqrt(1-alphas_cumprod_t)*noise
        """
        # pdb.set_trace()
        if noise is None:
            noise = torch.randn_like(input_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod,
                                        t, 
                                        input_start.shape) # sqrt_alphas_cumprod: 200, t: 10, x_start: [10, 1, 32, 32]
        
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                                                  t, 
                                                  input_start.shape)
        # pdb.set_trace()
        noise_input_t = sqrt_alphas_cumprod_t * input_start + sqrt_one_minus_alphas_cumprod_t * noise

        return noise_input_t
    

    def cal_losses(self, 
                   denoise_model,
                   input, 
                   condition, 
                   t,
                   mode, 
                   noise=None, 
                   denoise_loss_type="l1"):
        """
        进行去噪的计算 计算恢复的结果及损失
        :param: denoise_model: nn.module
        :param: input_start: tensor: batch of triple_emb, size: batch * hidden_dim or b * len * dim
        :param: condition: tensor: batch of condition (h, r) or (r, t) (batch * hidden_dim, batch * hidden_dim)
        :param: t: batch, 每一个样本一个时间戳
        :param: y0_pre: pre_head, pre_tail
        """
        triplet_emb, positive_labels_emb, negative_sample_emb, subsampling_weight = input # positive_labels_emb: b*dim; neg: b*neg*dim
        batch_size = positive_labels_emb.size(0)
        negative_sample_size = negative_sample_emb.size(0)

        if noise is None:
            noise = torch.randn_like(triplet_emb)
            # noise = nn.init.uniform_(
            #     tensor=self.y0_pre, 
            #     a=-self.embedding_range.item(), 
            #     b=self.embedding_range.item()
            # )


        # pdb.set_trace()
        noisy_input_t = self.q_sample(input_start=triplet_emb, 
                                      t=t, 
                                      noise=noise) # batch * 3dim 
        # pdb.set_trace()
        denoise_triplet_emb = denoise_model(noisy_input_t, condition, t, mode=mode) # batch * 3 * dim
        # pdb.set_trace()

        # logits = triple_classifier(predicted)
        #*********************************分类时加入*********************************************
        # if denoise_loss_type == 'l1':
        #     denoise_loss = F.l1_loss(positive_labels_emb, denoise_y0)
        # elif denoise_loss_type == 'l2':
        #     denoise_loss = F.mse_loss(positive_labels_emb, denoise_y0)
        # elif denoise_loss_type == "huber":
        #     denoise_loss = F.smooth_l1_loss(positive_labels_emb, denoise_y0)
        # else:
        #     raise NotImplementedError()
        
        # pdb.set_trace()
        # denoise_loss = - F.logsigmoid(denoise_loss)

        # if denoise_loss_type == 'l1':
        #     positive_score = self.gamma.item() - F.l1_loss(positive_labels_emb, denoise_y0, reduction='none')
        #     negative_score = self.gamma.item() - F.l1_loss(positive_labels_emb, denoise_y0.unsqueeze(1), reduction='none')
        # elif denoise_loss_type == 'l2':
        #     denoise_loss = F.mse_loss(positive_labels_emb, denoise_y0)
        # elif denoise_loss_type == "huber":
        #     denoise_loss = F.smooth_l1_loss(positive_labels_emb, denoise_y0)
        # else:
        #     raise NotImplementedError()
        
        # 重新定义loss
        # pdb.set_trace()
        
        positive_score = self.gamma - torch.norm(denoise_triplet_emb - positive_labels_emb, p=1, dim=-1).mean(dim=1) # 2是10左右 1是170左右

        negative_score = self.gamma - torch.norm(denoise_triplet_emb.unsqueeze(1) - negative_sample_emb, p=1, dim=-1).mean(dim=-1) # b * 1 * dim

        positive_score = F.logsigmoid(positive_score) # b
        negative_score = F.logsigmoid(-negative_score).mean(dim=1) # b
        positive_sample_loss = - (subsampling_weight * positive_score).sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum()
        positive_sample_loss /= subsampling_weight.sum()
        negative_sample_loss /= subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        metrics = None
#****************************************************************************        
        return loss, positive_sample_loss, negative_sample_loss, metrics
   
   
    # @torch.no_grad()
    def ddim_sample(self,
                    args):
        # pdb.set_trace()
        condition = args['condition']     
        shape = args['shape']
        mode = args['mode']

        batch_size = shape[0]
        total_timesteps, sampling_timesteps, eta, objective = self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps-1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        triplet_emb0 = torch.randn(shape).cuda()

        ensemble_triple_embs = []
        x_start = None
        # pdb.set_trace()
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time).cuda()
            x_start = self.denoise_model(triplet_emb0, condition, time_cond, mode) # 这个denoise生成的是x_0 ; triplet_emb0

            pred_noise = self.predict_noise_from_start(triplet_emb0, time_cond, x_start)
            
            if time_next < 0:
                triplet_emb0 = x_start
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(triplet_emb0)

            triplet_emb0 = x_start * alpha_next.sqrt() + \
                         c * pred_noise + \
                         sigma * noise
            ##############################
            if mode == 'head-batch':
                relation, tail, _ = condition
                relation, tail = Variable(relation, requires_grad=True), Variable(tail, requires_grad=True)  
                gt_emb = torch.cat((relation.unsqueeze(1), tail.unsqueeze(1)), dim=1)
                gt_emb = Variable(gt_emb, requires_grad=True) 
            if mode == 'tail-batch':
                head, relation, _ = condition
                head, relation = Variable(head, requires_grad=True), Variable(relation, requires_grad=True)
                gt_emb = torch.cat((head.unsqueeze(1), relation.unsqueeze(1)), dim=1)
                gt_emb = Variable(gt_emb, requires_grad=True)
            # pdb.set_trace()
            triplet_emb0 = Variable(triplet_emb0, requires_grad=True)
            opt = torch.optim.Adam([triplet_emb0], lr=0.0001) 
            # pdb.set_trace()
            for i in range(50):
                if mode == 'head-batch':
                    _, relation, tail = condition 
                    loss = torch.norm(triplet_emb0[:, 1:, :] - gt_emb, p=2)
                    # loss = Variable(loss, requires_grad=True)
                if mode == 'tail-batch':
                    head, relation, _ = condition
                    loss = torch.norm(triplet_emb0[:, :2, :] - gt_emb, p=2)
                    # loss = Variable(loss, requires_grad=True)
                # pdb.set_trace()
                opt.zero_grad()
                loss.backward() 
                if mode == 'head-batch':
                    mean_grad = torch.mean(triplet_emb0.grad[:, 1:, :], dim=1)
                    triplet_emb0.grad[:, 0, :] = mean_grad
                if mode == 'tail-batch':
                    mean_grad = torch.mean(triplet_emb0.grad[:, :2, :], dim=1)
                    triplet_emb0.grad[:, 2, :] = mean_grad
                opt.step()
            # pdb.set_trace()
            triplet_emb0 = triplet_emb0.data
            ##############################
        # pdb.set_trace()
        # if self.sampling_timesteps > 1:
        #         pass
        # else:
        return triplet_emb0

    
    def p_sample(self, 
                 input_rand, 
                 condition, 
                 t, 
                 t_index, 
                 mode):
        """
        生成数据 在test/valid的时候 从噪声中返回sample的候选实体embed
        param: input_rand: x_t 在每一步骤下的输入
        param: condition ()
        output: x_t-1
        """
        betas_t = extract(self.betas, t, input_rand.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, input_rand.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, input_rand.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 这个是去预测噪声的
        pred_start = self.denoise_model(input_rand, condition, t, mode) # x_t0
        
        # 由x0来计算当前的噪声
        pred_noise = self.predict_noise_from_start(input_rand, t, pred_start)

        model_mean = sqrt_recip_alphas_t * (input_rand - betas_t * pred_noise 
                                            / sqrt_one_minus_alphas_cumprod_t)

        # pdb.set_trace()
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, input_rand.shape)
            noise = torch.randn_like(input_rand)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
    
  
    def p_sample_loop(self, 
                      condition, 
                      shape, 
                      mode):
        """
        shape: batch * dim 

        """
        time_steps = self.timesteps
        noise = torch.randn(shape).cuda()
        batch_size = shape[0]
        out = {}
        with torch.no_grad():
            # for i in tqdm(reversed(range(0, time_steps))):
            for i in reversed(range(0, time_steps)):
                t = torch.full((batch_size,), i).cuda()
                noise = self.p_sample(input_rand=noise,
                                      condition=condition,
                                      t=t,
                                      t_index=i,
                                      mode=mode)
                out[str(i)] = noise
            recon_embed = out[str(0)]
            return recon_embed


            
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        param: model: ddpmkge
        """
        model.train()
        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator) # pos: batch * 3, neg: batch * neg_num 
        # pdb.set_trace()
        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda() 

        # score, denoise_loss, classifier_loss, metrics = model((positive_sample, positive_labels), mode=mode, if_train=True)
        loss, positive_sample_loss, negative_sample_loss, = model((positive_sample, negative_sample, subsampling_weight, mode), mode=mode, if_train=True)
        # pdb.set_trace()

        # if args.uni_weight:
        #     loss = score.mean()

        # else:
        #     loss = (subsampling_weight * score).sum()/subsampling_weight.sum()
       
        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        


        # for name, parms in model.named_parameters():	
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     # print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:',parms.grad)
        #     print("===")
       
        loss.backward()
        # pdb.set_trace()
        # for name, parms in model.named_parameters():	
        #     print('-->name:', name)
        #     # print('-->para:', parms)
        #     # print('-->grad_requirs:',parms.requires_grad)
        #     print('-->grad_value:', parms.grad)
        #     print("===")
        
        optimizer.step()
        # print("=============更新之后===========")

        # print(optimizer)


        positive_sample_loss = positive_sample_loss.mean()
        negative_sample_loss = negative_sample_loss.mean()

        log = {
            **regularization_log,
            'loss': loss.item(),
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            # **metrics
        }

        return log


    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, nentity, nrelation):
        '''
        Evaluate the model on test or valid datasets
        '''        
        model.eval()
        
        test_dataloader_head = DataLoader(
                Test_Ddpmkge_Neg_Dataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=Test_Ddpmkge_Neg_Dataset.collate_fn
            )

        test_dataloader_tail = DataLoader(
                Test_Ddpmkge_Neg_Dataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=Test_Ddpmkge_Neg_Dataset.collate_fn
            )
            
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

    # with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                # pdb.set_trace()
                # positive_sample: batch * 3, postive_labels: batch*nentity
                if args.cuda:
                    positive_sample = positive_sample.cuda() # batch * 3
                    negative_sample = negative_sample.cuda()
                    filter_bias = filter_bias.cuda()

                batch_size = positive_sample.size(0)
                # pdb.set_trace()
                if model.use_ensemble is not True:
                    score, score2, denoise_loss, condition = model((positive_sample, negative_sample, filter_bias, mode), mode=mode, if_train=False) # batch * dim
                    # score = model.triple_classifier(pre_embed) # 进行打分

                    # ********************** 这里要用编码的方式transe i.e. *********************************
                    # pdb.set_trace()
                    score = model.gamma.item() - score
                    score2 = model.gamma.item() - score2
                    score += filter_bias

                    score2 += filter_bias

                    # pdb.set_trace()
                    # positive_labels = positive_labels.to(score.device)
                    # classifier_loss = F.binary_cross_entropy_with_logits(score, positive_labels, reduction="none")
                    # is_positive = positive_labels > 0.5
                    # is_negative = positive_labels <= 0.5
                    # num_positive = is_positive.sum(dim=-1) # B
                    # num_negative = is_negative.sum(dim=-1) # B
                    # neg_weight = torch.zeros_like(positive_labels)
                    # neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)
                    # # 这里是否还需要进行归一化？
                    # neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
                    # classifier_loss = (classifier_loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)

                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim = 1, descending=True) # 

                    argsort_direct = torch.argsort(score2, dim = 1, descending=True) # 

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # pdb.set_trace()
                        ranking2 = (argsort_direct[i, :] == positive_arg[i]).nonzero()
                        assert ranking2.size(0) == 1
                        ranking2 = 1 + ranking2.item()

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            # 'classifier_loss': float(classifier_loss.item()),
                            'Denoise_loss': float(denoise_loss),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            # 'HITS@30': 1.0 if ranking <= 30 else 0.0,
                            # 'HITS@50': 1.0 if ranking <= 50 else 0.0,
                            # 'HITS@80': 1.0 if ranking <= 80 else 0.0,
                            # 'HITS@100': 1.0 if ranking <= 100 else 0.0,
                            # 'HITS@200': 1.0 if ranking <= 200 else 0.0,
                            # 'HITS@300': 1.0 if ranking <= 300 else 0.0,
                            # 'HITS@500': 1.0 if ranking <= 500 else 0.0,
                            'dir_MRR': 1.0/ranking2,
                            'dir_MR': float(ranking2),
                            # 'classifier_loss': float(classifier_loss.item()),
                            'dir_HITS@1': 1.0 if ranking2 <= 1 else 0.0,
                            'dir_HITS@3': 1.0 if ranking2<= 3 else 0.0,
                            'dir_HITS@10': 1.0 if ranking2 <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1
                
                if model.use_ensemble is True:
                    score_list, score2= model((positive_sample, negative_sample, filter_bias, mode), mode=mode, if_train=False) # batch * dim
                    # score = model.triple_classifier(pre_embed) # 进行打分

                    # ********************** 这里要用编码的方式transe i.e. *********************************
                    # pdb.set_trace()
                    argsort_list = []
                    for score_i in score_list:
                        score_i = model.gamma.item() - score_i
                        score_i += filter_bias
                        argsort_i = torch.argsort(score_i, dim = 1, descending=True)
                        argsort_list.append(argsort_i) 

                    score2 = model.gamma.item() - score2
                    score2 += filter_bias

                    argsort2 = torch.argsort(score2, dim = 1, descending=True) # 

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        ranking_min = 999999
                        ranking_max = -1
                        ranking_first = (argsort_list[0][i, :] == positive_arg[i]).nonzero()
                        #Notice that argsort is not ranking
                        ranking2 = (argsort2[i, :] == positive_arg[i]).nonzero()
                        for argsort_i in argsort_list:
                            # pdb.set_trace()

                            ranking_i = (argsort_i[i, :] == positive_arg[i]).nonzero()
                            if ranking_i < ranking_min:
                                ranking_min = ranking_i
                            if ranking_i > ranking_max:
                                ranking_max = ranking_i

                        # pdb.set_trace()
                        assert ranking2.size(0) == 1
                        ranking2 = 1 + ranking2.item()

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking_min = 1 + ranking_min.item()
                        ranking_max = 1 + ranking_max.item()
                        ranking_first = 1 + ranking_first.item()

                        # pdb.set_trace()
                        logs.append({
                            'MRR': 1.0/ranking_min,
                            'MR': float(ranking_min),
                            # 'classifier_loss': float(classifier_loss.item()),
                            # 'Denoise_loss': float(denoise_loss),
                            'HITS@1': 1.0 if ranking_min <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking_min <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking_min <= 10 else 0.0,
                            'HITS@100': 1.0 if ranking_min <= 100 else 0.0,
                            # 'HITS@30': 1.0 if ranking <= 30 else 0.0,
                            # 'HITS@50': 1.0 if ranking <= 50 else 0.0,
                            # 'HITS@80': 1.0 if ranking <= 80 else 0.0,
                            'max_MRR': 1.0/ranking_max,
                            'max_MR': float(ranking_max),
                            # 'classifier_loss': float(classifier_loss.item()),
                            'max_HITS@1': 1.0 if ranking_max <= 1 else 0.0,
                            'max_HITS@3': 1.0 if ranking_max<= 3 else 0.0,
                            'max_HITS@10': 1.0 if ranking_max <= 10 else 0.0,
                            'fir_MRR': 1.0/ranking_first,
                            'fir_MR': float(ranking_first),
                            # 'classifier_loss': float(classifier_loss.item()),
                            'fir_HITS@1': 1.0 if ranking_first <= 1 else 0.0,
                            'fir_HITS@3': 1.0 if ranking_first<= 3 else 0.0,
                            'fir_HITS@10': 1.0 if ranking_first <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1


        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics


if __name__ == "__main__":
    pass
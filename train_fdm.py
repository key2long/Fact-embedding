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
from rounding import *
from models import *
from functools import partial


class DDPMKge(nn.Module):
    """
    扩散KG embedding模型
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """
    def __init__(self, 
            dataset,
            denoise_model,
            triple_encoder,
            triple_classifier,
            # device,
            nentity,
            nrelation,
            hidden_dim,
            gamma,
            pretrain_emb,
            predict_xstart=True,
            double_entity_embedding=False,
            double_relation_embedding=False,
            timesteps=200,
            beta_schedule="linear",
            linear_start=1e-4,
            linear_end=2e-2,
            max_seq_len=3,
            loss_type="l2",
            lr=1e-5,
            dataset_onehot=True,
            use_ensemble = False,
            ddim_sampling_timesteps=10,
            ddim_sampling_eta=1.,
            rescale_timesteps=False,
            use_triple_encoder=False,
            clip_denoised=False,
            top_p=0,
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
            
            if dataset == 'YAGO3-10':
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

            # self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim), requires_grad=False)
            # self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim), requires_grad=False)
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
        # self.triple_classifier = triple_classifier.cuda()
        # self.triple_encoder = triple_encoder.cuda()
        self.triple_classifier = triple_classifier
        # pdb.set_trace()
        self.timesteps = timesteps
        self.dataset_onehot = dataset_onehot
        self.predict_xstart = predict_xstart
        self.rescale_timesteps = rescale_timesteps
        self.register_schedule(beta_schedule=beta_schedule, linear_start=linear_start, linear_end=linear_end)
        self.loss_type = loss_type
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.use_ensemble = use_ensemble
        self.use_triple_encoder = use_triple_encoder
        ########## ddpm init para ############

        ########## sample para ############  
        self.clip_denoised = clip_denoised   
        self.ddim_sampling_eta = ddim_sampling_eta
        self.ddim_sampling_timesteps = ddim_sampling_timesteps
        self.top_p = top_p
        ########## sample para ############  


    def register_schedule(self, beta_schedule, linear_start, linear_end):
        # define beta schedule
        # pdb.set_trace()
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
        log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0) beta * (1 - alphas_cumprod_prev) / 
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others  
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1)) # 这个用来计算ddim里的噪声

        # 分别是x_0前的系数
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 和x_t前的系数
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', posterior_variance)


    def forward(self, ddpmkge_model, sample, mode, if_train=False):
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
            positive_sample, positive_labels = sample
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
            t = torch.randint(0, self.timesteps, (batch_size,)).cuda()
            
            # 这里如果使用encoder来编码 否则直接拼接在一起
            if self.use_triple_encoder:
                triple_emb = self.triple_encoder(h_emb=head,
                                                 path_emb=relation,
                                                 t_emb=tail) # b * 3 * dim
            else:
                triple_emb = torch.stack((head, relation, tail), dim=1)


            if if_train == True:
                # loss, denoise_loss, classifier_loss, metrics = self.cal_losses(denoise_model=self.denoise_model,
                #                        triple_classifier=self.triple_classifier, 
                #                        input = (triple_emb, positive_labels),
                #                        condition=(relation, tail), 
                #                        t=t, 
                #                        denoise_loss_type=self.loss_type)
                # if metrics is not None:
                #     return loss, denoise_loss, classifier_loss, metrics
                # else:
                #     return loss, denoise_loss, classifier_loss
                # return loss, denoise_loss, classifier_loss, metrics

                # input_ids_mask = torch.tensor([1., 1., 1.]).cuda()
                input_ids_mask = torch.tensor([1., 0., 0.]).cuda()
                losses = self.training_ddpmkge_losses(denoise_model=self.denoise_model,
                                                      input_ids_mask=input_ids_mask,
                                                      input=(triple_emb, positive_labels),
                                                      condition=(relation, tail),
                                                      triple_classifier=self.triple_classifier,
                                                      t=t,
                                                      mode=mode)
                return losses


            if if_train == False:
                "The test period"
                # pre_embed = self.p_sample_loop(condition=(relation, tail), 
                #                                time_steps=self.timesteps,
                #                                noise_shape=(batch_size, self.max_seq_len, head.shape[-1]))
                input_ids_mask = torch.tensor([1., 0., 0.]).cuda()
                # input_ids_mask = torch.tensor([1., 1., 1.]).cuda()
                input_ids_mask_ori = input_ids_mask
                input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), triple_emb.shape).cuda()
                noise = torch.randn_like(triple_emb)
                x_noised = torch.where(input_ids_mask==0, triple_emb, noise) # 部分加噪后的输入

                if self.ddim_sampling_timesteps == self.timesteps:
                    self.use_ddim = False
                    step_gap = 1
                else:
                    self.use_ddim = True
                    step_gap = self.timesteps // self.ddim_sampling_timesteps
                
                sample_fn = (
                    self.p_sample_loop if not self.use_ddim else self.ddim_sample_loop
                )

                samples = sample_fn(denoise_model=self.denoise_model,
                                    shape=(batch_size, self.max_seq_len, head.shape[-1]),
                                    noise=x_noised,
                                    clip_denoised=self.clip_denoised,
                                    condition=(relation, tail),
                                    denoised_fn=partial(denoised_fn_round, ddpmkge_model),
                                    top_p=self.top_p,
                                    clamp_step=0,
                                    clamp_first=True,
                                    mask=input_ids_mask,
                                    x_start=triple_emb,
                                    gap=step_gap
                                    )

                sample = samples[-1]
                denoise_loss = F.smooth_l1_loss(triple_emb, sample)
                return sample, denoise_loss.mean(), (relation, tail)
                # 计算相似度和负样本


        elif mode == 'tail-batch':
            # 通过 h r => t
            positive_sample, positive_labels = sample
            batch_size = positive_sample.size(0)
            
            head = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 0])# batch * dim
            relation = torch.index_select(self.relation_embedding,
                                          dim=0, index=positive_sample[:, 1]) # batch * dim
            tail = torch.index_select(self.entity_embedding, 
                                      dim=0, index=positive_sample[:, 2])# batch * dim
            t = torch.randint(0, self.timesteps, (batch_size,)).cuda()
            
            if self.use_triple_encoder:
                triple_emb = self.triple_encoder(h_emb=head,
                                                 path_emb=relation,
                                                 t_emb=tail) # b * 3 * dim
            else:
                triple_emb = torch.stack((head, relation, tail), dim=1)  

            if if_train == True:
                # loss, denoise_loss, classifier_loss, metrics = self.cal_losses(denoise_model=self.denoise_model, 
                #                        triple_classifier=self.triple_classifier, 
                #                        input=(triple_emb, positive_labels),
                #                        condition=(head, relation), 
                #                        t=t, 
                #                        denoise_loss_type=self.loss_type)
                # if metrics is not None:
                #     return loss, denoise_loss, classifier_loss, metrics
                # else:
                #     return loss, denoise_loss, classifier_loss

                input_ids_mask = torch.tensor([0., 0., 1.]).cuda()
                # input_ids_mask = torch.tensor([1., 1., 1.]).cuda()
                losses = self.training_ddpmkge_losses(denoise_model=self.denoise_model,
                                                      input_ids_mask=input_ids_mask,
                                                      input=(triple_emb, positive_labels),
                                                      condition=(head, relation),
                                                      triple_classifier=self.triple_classifier,
                                                      t=t,
                                                      mode=mode)
                return losses


            else:
                # "The test period"
                # pre_embed = self.p_sample_loop(condition=(head, relation), 
                #                                time_steps=self.timesteps,
                #                                noise_shape=(batch_size, self.max_seq_len, head.shape[-1]))
                input_ids_mask = torch.tensor([0., 0., 1.]).cuda()
                # input_ids_mask = torch.tensor([1., 1., 1.]).cuda()
                input_ids_mask_ori = input_ids_mask
                input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), triple_emb.shape).cuda()
                noise = torch.randn_like(triple_emb)
                x_noised = torch.where(input_ids_mask==0, triple_emb, noise) # 部分加噪后的输入

                if self.ddim_sampling_timesteps == self.timesteps:
                    self.use_ddim = False
                    step_gap = 1
                else:
                    self.use_ddim = True
                    step_gap = self.timesteps // self.ddim_sampling_timesteps

                sample_fn = (
                    self.p_sample_loop if not self.use_ddim else self.ddim_sample_loop
                )
                
                samples = sample_fn(denoise_model=self.denoise_model,
                                    shape=(batch_size, self.max_seq_len, head.shape[-1]),
                                    noise=x_noised,
                                    clip_denoised=self.clip_denoised,
                                    condition=(head, relation),
                                    denoised_fn=partial(denoised_fn_round, ddpmkge_model),
                                    top_p=self.top_p,
                                    clamp_step=0,
                                    clamp_first=True,
                                    mask=input_ids_mask,
                                    x_start=triple_emb,
                                    gap=step_gap
                                    )

                sample = samples[-1]
                denoise_loss = F.smooth_l1_loss(triple_emb, sample)
                return sample, denoise_loss.mean(), (relation, tail)
        else:
            raise ValueError('mode %s not supported' % mode)


    def predict_noise_from_xstart(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )        


    def predict_xstart_from_noise(self, x_t, t, noise):
        return(
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


    def scale_timesteps(self, t):
        """
        把时间t映射到0-T的范围内(T是1000)
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.timesteps)
        return t


    def get_x_start(self, x_start_mean, std):
        """
        triple_embedding projection from {Emb(triple)} => {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        """
        noise = torch.rand_like(x_start_mean)
        return(
            x_start_mean + std * noise
        )


    def _x0_helper(self, model_output, x, t):
        # pdb.set_trace()
        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else: # predict eps
            pred_xstart = self.predict_xstart_from_noise(x_t=x, t=t, eps=model_output)
        
            pred_prev, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}


    def q_sample(self, 
                 x_start, 
                 t, 
                 noise=None,
                 mask=None):
        """
        在输入数据中进行加噪, 返回任意时刻的噪声数据 x(t) = sqrt(alphas_cumprod_t)*x_0 + sqrt(1-alphas_cumprod_t)*noise
        """
        # pdb.set_trace()
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod,
                                        t, 
                                        x_start.shape) # sqrt_alphas_cumprod: 200, t: 10, x_start: [10, 1, 32, 32]
        
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                                                  t, 
                                                  x_start.shape)

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        # 这里加入mask的机制把不想加噪的位置去掉 还是原来的初始值
        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape) # mask: [0, 0, 0, 0, 1, 1, 1] mask 掉前四个位置的加噪emb => 变为不加噪声
            return torch.where(mask==0, x_start, x_t)


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0). 获得前向的每一步的均值方差....

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance


    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0) 获得均值方差 这个是通过x_0 和 x_t算出来的

        """
        assert x_start.shape == x_t.shape
        # pdb.set_trace()
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # posterior_log_variance_clipped = extract(
        #     self.posterior_log_variance_clipped, t, x_t.shape
        # )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            # == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        # return posterior_mean, posterior_variance, posterior_log_variance_clipped
        return posterior_mean, posterior_variance


    def p_mean_variance(self, 
                        denoise_model, 
                        x, 
                        t, 
                        clip_denoised=True,
                        condition=None, 
                        denoised_fn=None, 

        ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0. 获得x_[t-1]均值方差 由x_t和x_0计算出来,目标是计算x0, 无论模型预测的是x0还是eps

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps. N * 1
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        B = x.size(0)
        assert t.shape == (B,)
        # print(x.shape)
        if condition is not None:
            model_output = denoise_model(x, condition, self.scale_timesteps(t)) #有两种可能 一种是预测噪声eps 一种是预测x0=>这个都是算x_{t-1}均值条件
        else:
            model_output = denoise_model(x, self.scale_timesteps(t))

        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        # pdb.set_trace()
        # init
        model_variance = torch.empty_like(self.betas).to(self.betas.device)
        model_log_variance = torch.empty_like(self.betas).to(self.betas.device)

        model_variance[0], model_variance[1:] = self.posterior_variance[1], self.betas[1:] # 认为beta固定
        model_log_variance[0], model_log_variance[1:] = self.posterior_variance[1], self.betas[1:]
        
        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape) # shape ([8, 1, 1])

        # 
        def process_xstart(x):
            if denoised_fn is not None:
                # print(denoised_fn)
                # pdb.set_trace()
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        # pdb.set_trace()
        if self.predict_xstart:
            pred_xstart = process_xstart(model_output) # model_output是x0 也是直接返回
        else:
            ### model is used to predict eps
            pred_xstart = process_xstart(
                self.predict_xstart_from_noise(x_t=x, t=t, eps=model_output) #这个时候model_output 是 eps 则用公式x_0 = (...)x_t + (...)eps_t
            )
        # (18)
        model_mean, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        # pdb.set_trace()
        assert (
            model_mean.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    def p_sample(self, 
                 denoise_model, 
                 x, 
                 t, 
                 clip_denoised=True, 
                 denoised_fn=None,
                 condition=None, 
                 top_p=None, 
                 mask=None, 
                 x_start=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep x_t. q(x_{t-1} | x_t, x_0) 获得均值方差
        这个是计算某个时刻x_{t-1}的输出, 以及在x_t时刻预测出的x0, out里包含了x_t-1的均值方差和预测的x0
        :param denoise_model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(denoise_model,
                                   x,
                                   t,
                                   clip_denoised=clip_denoised,
                                   condition=condition,
                                   denoised_fn=denoised_fn,
                )

        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = torch.randn_like(x)
            replace_mask = torch.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = torch.randn_like(noise[replace_mask])
                replace_mask = torch.abs(noise) > top_p
            assert (torch.abs(noise) <= top_p).all()

        else:
            noise = torch.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        # 
        if mask == None:
            pass
        else:
            sample = torch.where(mask==0, x_start, sample)

        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"], 
            "out": out
        }


    def p_sample_loop(
        self,
        denoise_model,
        shape,
        noise=None,
        clip_denoised=True,
        condition=None,
        denoised_fn=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model.

        :param denoise_model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        # 这个sample是不同时刻的采样值从T->0
        for sample in self.p_sample_loop_progressive(
            denoise_model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            condition=condition,
            denoised_fn=denoised_fn,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start
        ):
            final.append(sample['sample'])
        return final


    def p_sample_loop_progressive(
        self,
        denoise_model,
        shape,
        noise=None,
        clip_denoised=True,
        condition=None,
        denoised_fn=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(denoise_model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: # custom your the start point of x_0
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)
        indices = list(range(self.timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices: # from T to 0
            t = torch.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with torch.no_grad():
                out = self.p_sample(
                    denoise_model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    condition=condition,
                    denoised_fn=denoised_fn_cur,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]


    def _ddim_sample(
        self,
        denoise_model,
        x,
        t,
        clip_denoised=True,
        condition=None,
        denoised_fn=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None
    ):
        """
        x_start: triple emb
        x: 
        mask:
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            denoise_model,
            x,
            t,
            clip_denoised=clip_denoised,
            condition=condition,
            denoised_fn=denoised_fn,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.predict_noise_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # print(sigma.mean())
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            print(t.shape)
            sample=langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)
        
        if mask == None:
            pass
        else:
            sample = torch.where(mask==0, x_start, sample)
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def ddim_reverse_sample(
        self,
        denoise_model,
        x,
        t,
        clip_denoised=True,
        condition=None,
        denoised_fn=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            denoise_model,
            x,
            t,
            clip_denoised=clip_denoised,
            condition=condition,
            denoised_fn=denoised_fn,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}


    def ddim_sample_loop(
        self,
        denoise_model,
        shape,
        noise=None,
        clip_denoised=True,
        condition=None,
        denoised_fn=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        mask: input_ids_mask
        x_start: triple_emb
        noise: x_noised
        Generate samples from the model using DDIM.
        :param gap: compute ddim sampling for each {gap} step

        Same usage as p_sample_loop().
        """
        final = []
        for sample in self.ddim_sample_loop_progressive(
            denoise_model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            condition=condition,
            denoised_fn=denoised_fn,
            device=device,
            progress=progress,
            mask=mask,
            x_start=x_start,
            gap = gap
        ):
            final.append(sample['sample'])
        return final


    def ddim_sample_loop_progressive(
        self,
        denoise_model,
        shape,
        noise=None,
        clip_denoised=True,
        condition=None,
        denoised_fn=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(denoise_model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = torch.randn(*shape, device=device)
        indices = list(range(self.timesteps))[::-1][::gap]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self._ddim_sample(
                    denoise_model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    condition=condition,
                    denoised_fn=denoised_fn,
                    eta=eta,
                    langevin_fn=langevin_fn,
                    mask=mask,
                    x_start=x_start,
                )
                yield out
                sample_x = out["sample"]


    def cal_losses(self, 
                 denoise_model,
                 triple_classifier,
                 input, 
                 condition, 
                 t, 
                 noise=None, 
                 denoise_loss_type="l1"):
        """
        进行去噪的计算 计算恢复的结果及损失
        :param: denoise_model: nn.module
        :param: input_start: tensor: batch of triple_emb, size: batch * hidden_dim or b * len * dim
        :param: condition: tensor: batch of condition (h, r) or (r, t) (batch * hidden_dim, batch * hidden_dim)
        :param: t: batch, 每一个样本一个时间戳
        """
        input_start, label = input
        if noise is None:
            noise = torch.randn_like(input_start)

        noisy_input_t = self.q_sample(x_start=input_start, 
                                      t=t, 
                                      noise=noise) # batch * 3 * dim or batch * path_len * dim

        # predicted_noise = denoise_model(noisy_input, condition, t)
        # pdb.set_trace()
        predicted = denoise_model(noisy_input_t, condition, t) # batch * seq_len * hidden_dim
        # pdb.set_trace()

        # logits = triple_classifier(predicted)
        #*********************************分类时加入*********************************************
        logits = triple_classifier(predicted, condition)

        if denoise_loss_type == 'l1':
            denoise_loss = F.l1_loss(input_start, predicted)
        elif denoise_loss_type == 'l2':
            denoise_loss = F.mse_loss(input_start, predicted)
        elif denoise_loss_type == "huber":
            denoise_loss = F.smooth_l1_loss(input_start, predicted)
        else:
            raise NotImplementedError()
        
        # pdb.set_trace()
        classifier_loss = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        is_positive = label > 0.5
        is_negative = label <= 0.5
        num_positive = is_positive.sum(dim=-1) # B
        num_negative = is_negative.sum(dim=-1) # B
        neg_weight = torch.zeros_like(label).to(label.device)
        neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)
        # 这里是否还需要进行归一化？
        neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
        classifier_loss = (classifier_loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)

        loss = 0.3 * denoise_loss + 0.7 * classifier_loss

# ******************************************************************************************
        # argsort = torch.argsort(logits, dim = 1, descending=True)
        # # pdb.set_trace()
        # logs = []
        # for i in range(logits.shape[0]):
        #     label_i = label[i].nonzero()
        #     #Notice that argsort is not ranking
        #     for j in label_i:
        #         ranking = (argsort[i, :] == j).nonzero()
        #         #ranking + 1 is the true ranking used in evaluation metrics
        #         ranking = 1 + ranking.item()
        #         logs.append({
        #             'MRR': 1.0/ranking,
        #             'MR': float(ranking),
        #             'HITS@1': 1.0 if ranking <= 1 else 0.0,
        #             'HITS@10': 1.0 if ranking <= 10 else 0.0,
        #             'HITS@50': 1.0 if ranking <= 50 else 0.0,
        #             'HITS@100': 1.0 if ranking <= 100 else 0.0,
        #             'HITS@200': 1.0 if ranking <= 200 else 0.0,
        #         })
        
        # metrics = {}
        # for metric in logs[0].keys():
        #     metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        metrics = None
#****************************************************************************        
        return loss, denoise_loss, classifier_loss, metrics
   
   
    @torch.no_grad()
    def ddim_sample(self,
                    condition,
                    shape):
        
        batch_size = shape[0]
        total_timesteps, ddim_sampling_timesteps, eta, objective = self.timesteps, self.ddim_sampling_timesteps, self.ddim_sampling_eta, self.objective
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps-1, steps=ddim_sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        triple_emb = torch.randn(shape).cuda()

        ensemble_triple_embs = []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time).cuda()
            x_start = self.denoise_model(triple_emb, condition, time_cond)
            pred_noise = self.predict_noise_from_xstart(triple_emb, time_cond, x_start)
            
            if time_next < 0:
                triple_emb = x_start
                continue
            
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(triple_emb)

            triple_emb = x_start * alpha_next.sqrt() + \
                         c * pred_noise + \
                         sigma * noise 

        if self.use_ensemble and self.sampling_timesteps > 1:
                pass
        else:
            return x_start


    def training_ddpmkge_losses(self, 
                                denoise_model,
                                input_ids_mask,
                                input,
                                triple_classifier,
                                t,
                                condition=None,
                                noise=None,
                                mode=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start_mean: the [N x C x dim] embed triple. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param input_ids_mask: [1, 0, 0]
        :param condition: 如果不用mask机制, 则用condition(h_emb, r_emb)
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # pdb.set_trace()
        x_start_mean, label = input # triple_embeddings: b, len, dim; label: b, nentity onehot编码

        
        std = extract(self.sqrt_one_minus_alphas_cumprod,
                      torch.tensor([0]).to(x_start_mean.device),
                      x_start_mean.shape) # 这里是t=0的情况所有的batch
        
        # print(std.shape, )
        # x_start_log_var = 2 * th.log(std)
        x_start = self.get_x_start(x_start_mean, std) # word_embeddings => x0

        # print(x_start_mean.shape, x_start.shape)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise, mask=input_ids_mask) # reparametrization trick. 采样的数据是

        terms = {}   

        target = x_start # target是x0 初始emb加上噪声的

        if condition is not None:
            model_output = denoise_model(x_t, condition, self.scale_timesteps(t)) # output是 x_0 or eps
        else:
            model_output = denoise_model(x_t, self.scale_timesteps(t)) # output是 x_0 or eps

        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2) # batch
        # pdb.set_trace()

        model_out_x_start = self._x0_helper(model_output, x_t, t)['pred_xstart'] # predicted_xstart = model_output x_0
        t0_mask = (t == 0)
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2) # 这里是x0 <=> 初始embedding 的损失
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"]) # 这里mse t-1 -- t0

        # tT_mask = (t == self.timesteps - 1)
        out_mean, _, _ = self.q_mean_variance(x_start, torch.LongTensor([self.timesteps - 1]).to(x_start.device))
        tT_loss =  mean_flat(out_mean ** 2)

        decoder_nll = self.entity_discrete_loss(x_start,
                                                triple_classifier,
                                                label,
                                                condition=condition, 
                                                dataset_onehot=self.dataset_onehot,
                                                mode=mode) # embedding regularization

        terms["nll"] = self.entity_discrete_loss(model_out_x_start, 
                                                 triple_classifier, 
                                                 label,
                                                 condition=condition,
                                                 mask=input_ids_mask, 
                                                 dataset_onehot=self.dataset_onehot,
                                                 mode=mode) # x_0->model_out_x_start

        # assert (model.lm_head.weight == model.word_embedding.weight).all()

        # terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        terms["transe"] = self.TransE_Loss(model_out_x_start)
        # pdb.set_trace()
        terms["loss"] = 0.6*terms["mse"] + 0.4*terms["nll"] + tT_loss + terms["transe"]

        return terms        

    def TransE_Loss(self,
                    triple_embeddings,
                    ):
        head_embedding = triple_embeddings[:, 0, :]
        relation_embedding = triple_embeddings[:, 1, :]
        tail_embedding = triple_embeddings[:, 2, :]

        loss = mean_flat((head_embedding + relation_embedding - tail_embedding) ** 2)
        return loss

    def entity_discrete_loss(self, 
                             x_start, 
                             triple_classifier, 
                             label,  
                             mask=None, 
                             condition=None, 
                             dataset_onehot=True,
                             mode=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        if condition is not None:
            # 这个是分类有条件的
            logits = triple_classifier(x_start, self.entity_embedding, mode) # b, nentity
        else:
            logits = triple_classifier(x_start, self.entity_embedding, mode)
        
        # pdb.set_trace()
        if dataset_onehot == True:
            loss_fun = torch.nn.CrossEntropyLoss(reduction='none')
            # pdb.set_trace()
            ids = label.argmax(dim=-1) # b [1024]
            decoder_nll = loss_fun(logits, ids) # [1024]
            return decoder_nll

        else:
            # 如果是多分类的任务, label里有多个hot
            decoder_nll = F.binary_cross_entropy_with_logits(logits, label, reduction='none')
            is_positive = label > 0.5
            is_negative = label <= 0.5
            num_positive = is_positive.sum(dim=-1) # B
            num_negative = is_negative.sum(dim=-1) # B
            neg_weight = torch.zeros_like(label).to(label.device)
            neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)
            # 这里是否还需要进行归一化？
            neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
            decoder_nll = (decoder_nll * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)

            return decoder_nll


    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        """
        param: model: ddpmkge model
        """
        model.train()
        optimizer.zero_grad()

        positive_sample, positive_labels, subsampling_weight, mode = next(train_iterator) # pos: batch * 3, neg: batch * neg_num 
        
        if args.cuda:
            positive_sample = positive_sample.cuda()
            positive_labels = positive_labels.cuda()
            subsampling_weight = subsampling_weight.cuda() 

        # score, denoise_loss, classifier_loss, metrics = model((positive_sample, positive_labels), mode=mode, if_train=True)
        losses = model(ddpmkge_model=model, 
                       sample=(positive_sample, positive_labels), 
                       mode=mode, 
                       if_train=True) # batch * dim
        # pdb.set_trace()

        if args.uni_weight:
            loss = losses["loss"].mean()

        else:
            loss = (subsampling_weight * losses["loss"]).sum()/subsampling_weight.sum()
            # loss = losses["loss"].mean()
       
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
            
        loss.backward()

        # pdb.set_trace()
        # for name, param in model.triple_classifier.named_parameters():
        #     print(f'name = {name}, grad = {param.grad}, value_norm = {torch.sum(param.data ** 2)}')
        # pdb.set_trace()
        # for name, param in model.triple_encoder.named_parameters():
        #     print(f'name = {name}, grad = {param.grad}, value_norm = {torch.sum(param.data ** 2)}')
        # pdb.set_trace()
        # for name, param in model.denoise_model.named_parameters():
        #     print(f'name = {name}, grad = {param.grad}, value_norm = {torch.sum(param.data ** 2)}')

        optimizer.step()

        denoise_loss = losses['mse'].mean()
        classifier_loss = losses["nll"].mean()
        transe_loss = losses["transe"].mean()

        log = {
            **regularization_log,
            'loss': loss.item(),
            'denoise_loss': denoise_loss.item(),
            'classifier_loss': classifier_loss.item(),
            'transe_loss': transe_loss.item(),
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
                Test_Ddpm_Dataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=Test_Ddpm_Dataset.collate_fn
            )

        test_dataloader_tail = DataLoader(
                Test_Ddpm_Dataset(
                    test_triples, 
                    all_true_triples, 
                    nentity, 
                    nrelation, 
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=Test_Ddpm_Dataset.collate_fn
            )
            
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, positive_labels, filter_bias, mode in test_dataset:
                    # positive_sample: batch * 3, postive_labels: batch*nentity
                    if args.cuda:
                        positive_sample = positive_sample.cuda() # batch * 3
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)
 
                    sample, denoise_loss, condition = model(ddpmkge_model=model, 
                                                            sample=(positive_sample, positive_labels), 
                                                            mode=mode, 
                                                            if_train=False) # batch * dim

                    # score = model.triple_classifier(pre_embed) # 进行打分

                    # *******************************************************
                    # score = model.triple_classifier(sample)

                    score = triple_classifier(sample, model.entity_embedding, mode=mode)

                    score += filter_bias

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
                    argsort = torch.argsort(score, dim = 1, descending=True) # index 的排序 实体的得分从高到低排序
                    pdb.set_trace
                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0] # 正确实体的index
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

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
                            'HITS@30': 1.0 if ranking <= 30 else 0.0,
                            'HITS@50': 1.0 if ranking <= 50 else 0.0,
                            'HITS@80': 1.0 if ranking <= 80 else 0.0,
                            'HITS@100': 1.0 if ranking <= 100 else 0.0,
                            'HITS@200': 1.0 if ranking <= 200 else 0.0,
                            'HITS@300': 1.0 if ranking <= 300 else 0.0,
                            'HITS@500': 1.0 if ranking <= 500 else 0.0,
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
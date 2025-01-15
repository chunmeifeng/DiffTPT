
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
from tqdm import tqdm
import os
import copy

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='/data1/stuyuany/cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, num_prompts=1):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.ctx_init = ['a_photo_of_a']
        # self.ctx_all = ["a clear image of a", "a single picture showing a", "an artistic rendition of a", "a vivid depiction of a", 
        #                 "a detailed portrayal of a", "a distinct illustration of a", "a colorful representation of a", "a unique sketch of a"]
        # self.ctx_all = ["a photo of a", 'a sketch of a']
        # ctx_all = ["a photo of a", "an artistic depiction of a", "this illustration represents a", "a creative rendering of a", "this image visually interprets a", 
        #            "a stylized representation of a", "an abstract concept of a", "a colorful artwork of a"]
        
        # ctx_all = ["a photo of a", "an action shot of a", "this frame captures a", "a mid-action photo depicting a", "a dynamic pose of a", 
        #            "this image shows a person performing a", "an active scene of a", "a snapshot from a video of a", "an energetic moment of a"]
        # self.ctx_all = ['a photo of a', 'an artistic dipiction of a']
        # self.ctx_all = ['an artistic dipiction of a']
        ctx_all = ["a clear image of a", "a single picture showing a", "an artistic rendition of a", "a vivid depiction of a", 
                   "a detailed portrayal of a", "a distinct illustration of a", "a colorful representation of a", "a unique sketch of a"]

        # replace ctx_all for the dataset of your choice

        self.ctx_all = ctx_all[0:num_prompts]
        #self.ctx_all = ["a photo of a", "an artistic depiction of a", "this illustration represents a", "a creative rendering of a"]
        self.prefix_list = []
        self.ctx_list = []
        self.ctx_vectors_list = []
        self.n_ctx_list = []
        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        # if ctx_init:
        #     # use given words to initialize context vectors
        #     print("Initializing the contect with given words: [{}]".format(ctx_init))
        #     ctx_init = ctx_init.replace("_", " ")
        #     if '[CLS]' in ctx_init:
        #         ctx_list = ctx_init.split(" ")
        #         split_idx = ctx_list.index("[CLS]")
        #         ctx_init = ctx_init.replace("[CLS] ", "")
        #         ctx_position = "middle"
        #     else:
        #         split_idx = None
        #     self.split_idx = split_idx
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = tokenize(ctx_init).to(self.device)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(dtype)
        #     ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        #     prompt_prefix = ctx_init
        if self.ctx_all:
            print("initializing context with ", self.ctx_all)
            for ctx_init in self.ctx_all:
                ctx_init = ctx_init.replace("_", " ")
                if '[CLS]' in ctx_init:
                    ctx_list = ctx_init.split(" ")
                    split_idx = ctx_list.index("[CLS]")
                    ctx_init = ctx_init.replace("[CLS] ", "")
                    ctx_position = "middle"
                else:
                    split_idx = None
                self.split_idx = split_idx
                n_ctx = len(ctx_init.split(" "))
                self.n_ctx_list.append(n_ctx)
                prompt = tokenize(ctx_init).to(self.device)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
                self.ctx_vectors_list.append(ctx_vectors)
                prompt_prefix = ctx_init
                self.prefix_list.append(prompt_prefix)
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        #self.prompt_prefix = prompt_prefix
        print(f'Initial context: "{self.prefix_list}"')

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            for ctx_vectors in self.ctx_vectors_list:
                ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state_list = [ctx_vectors.detach().clone() for ctx_vectors in self.ctx_vectors_list]
        self.ctx_list = [nn.Parameter(ctx_vectors) for ctx_vectors in self.ctx_vectors_list] # to be optimized

        prompts_group = []
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]

            for prompt_prefix in self.prefix_list:
                prompts = [prompt_prefix + " " + name + "." for name in classnames]
                prompts_group.append(prompts)
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        tokenized_prompts_group = []
        for prompts in prompts_group:
            tokenized_prompts_group.append(torch.cat([tokenize(p) for p in prompts]).to(self.device))
        embedding_group = []
        with torch.no_grad():
            for tokenized_prompts in tokenized_prompts_group:
                embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
                embedding_group.append(embedding)


        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts_group = tokenized_prompts_group  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.classnames = classnames

    def reset(self):
        ctx_vectors_list = self.ctx_init_state_list
        for i, ctx in enumerate(self.ctx_list):
            ctx.copy_(ctx_vectors_list[i]) # to be optimized
        #self.ctx_list=(ctx_vectors_list)
        if self.learned_cls:
            cls_vectors_list = self.cls_init_state_list
            self.cls.copy_(cls_vectors_list)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        prompts_group = []
        # if not self.learned_cls:
        #     classnames = [name.replace("_", " ") for name in classnames]
        #     name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        #     prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]

            for prompt_prefix in self.prefix_list:
                prompts = [prompt_prefix + " " + name + "." for name in classnames]
                prompts_group.append(prompts)
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            for prompt_prefix in self.prefix_list:
                prompts = [prompt_prefix + " " + name + "." for name in classnames]
                prompts_group.append(prompts)
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            for self_cls_init_state in self.cls_init_state_list:
                self_cls_init_state = cls_vectors.detach().clone()
        # tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        tokenized_prompts_group = []
        for prompts in prompts_group:
            tokenized_prompts_group.append(torch.cat([tokenize(p) for p in prompts]).to(self.device))

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        # with torch.no_grad():
        #     embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        embedding_group = []
        with torch.no_grad():
            for tokenized_prompts in tokenized_prompts_group:
                embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)
                embedding_group.append(embedding)

        print(len(embedding_group))
        # self.token_prefix = None
        # self.token_suffix = None
        token_prefix = []
        token_suffix = []
        #print(embedding_group)
        for i, embedding in enumerate(embedding_group):
            token_prefix.append(embedding[:, :1, :])
            token_suffix.append(embedding[:, 1 + self.n_ctx_list[i] :, :])  # CLS, EOS

        #print(type(token_prefix[0])) torch.tensor
        #print(type(token_suffix[0])) torch.tensor

        # print(self.token_prefix[0].shape) torch.Size([1, 512])

        # for token_prefix in self.token_prefix:
        #     print(token_prefix.shape[0])
        
        
        self.token_prefix = torch.stack(token_prefix, dim = 0)

        max_length = max(suffix.size(1) for suffix in token_suffix)
        padded_token_suffix = []
        token_suffix_lengths = []
        for suffix in token_suffix:
            padded_suffix = F.pad(suffix, (0, 0, 0, max_length - suffix.size(1)))
            padded_token_suffix.append(padded_suffix)
            token_suffix_lengths.append(suffix.size(1))

        self.token_suffix = torch.stack(padded_token_suffix, dim = 0)
        self.token_suffix_lengths = token_suffix_lengths

        self.name_lens = name_lens
        # self.tokenized_prompts = tokenized_prompts
        self.tokenized_prompts_group = tokenized_prompts_group
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx_list = self.ctx_list
        if ctx_list[0].dim() == 2:
            ctx_list = [ctx.unsqueeze(0).expand(self.n_cls, -1, -1) for ctx in self.ctx_list]
        elif not ctx_list[0].size()[0] == self.n_cls:
            ctx_list = [ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1) for ctx in self.ctx_list]

        prompts_group=[]
        for i in range(len(self.token_prefix)):
            prefix = self.token_prefix[i]
            padded_suffix = self.token_suffix[i]
            if self.batch_size is not None: 
                # This way only works for single-gpu setting (could pass batch size as an argument for forward())
                prefix = prefix.repeat(self.batch_size, 1, 1, 1)
                padded_suffix = padded_suffix.repeat(self.batch_size, 1, 1, 1)

            if self.learned_cls:
                assert self.class_token_position == "end"
            if self.class_token_position == "end":
                if self.learned_cls:
                    cls = self.cls
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx_list[i],     # (n_cls, n_ctx, dim)
                            cls,     # (n_cls, 1, dim)
                            padded_suffix[:, :self.token_suffix_lengths[i], :],  # (n_cls, *, dim)
                            # restored_tensor = padded_tensor[:, :original_length, :]
                        ],
                        dim=-2,
                    )
                else:
                    # print(prefix.shape)
                    # print(ctx.shape)
                    # print(suffix.shape)
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx_list[i],     # (n_cls, n_ctx, dim)
                            padded_suffix[:, :self.token_suffix_lengths[i], :],  # (n_cls, *, dim)
                        ],
                        dim=-2,
                    )
            elif self.class_token_position == "middle":
                # TODO: to work with a batch of prompts
                if self.split_idx is not None:
                    half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
                else:
                    half_n_ctx = self.n_ctx // 2
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                    ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                    prompt = torch.cat(
                        [
                            prefix_i,     # (1, 1, dim)
                            ctx_i_half1,  # (1, n_ctx//2, dim)
                            class_i,      # (1, name_len, dim)
                            ctx_i_half2,  # (1, n_ctx//2, dim)
                            suffix_i,     # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)

            elif self.class_token_position == "front":
                prompts = []
                for i in range(self.n_cls):
                    name_len = self.name_lens[i]
                    prefix_i = prefix[i : i + 1, :, :]
                    class_i = suffix[i : i + 1, :name_len, :]
                    suffix_i = suffix[i : i + 1, name_len:, :]
                    ctx_i = ctx[i : i + 1, :, :]
                    prompt = torch.cat(
                        [
                            prefix_i,  # (1, 1, dim)
                            class_i,   # (1, name_len, dim)
                            ctx_i,     # (1, n_ctx, dim)
                            suffix_i,  # (1, *, dim)
                        ],
                        dim=1,
                    )
                    prompts.append(prompt)
                prompts = torch.cat(prompts, dim=0)
            else:
                raise ValueError
            prompts_group.append(prompts)
        return prompts_group


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, num_prompts=1):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls, num_prompts)
        self.criterion = criterion
        self.ratio = 0.2 # 0.0001, 0.13
        self.clip_dtype = clip.dtype
        self.arch = arch

        if self.arch == "RN50":
            self.adapter = Adapter(1024, 4).to(self.clip_dtype).cuda()
        elif self.arch == "ViT-B/16":
            self.adapter = Adapter(512, 4).to(self.clip_dtype).cuda()

        # self.adapter = None

    def build_adapter(self, cache_keys=None):
        if cache_keys is not None:
            self.adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(self.clip_dtype).cuda()
            self.adapter.weight = nn.Parameter(cache_keys.t())
        else:
            if self.arch == "RN50":
                self.adapter = Adapter(1024, 4).to(self.clip_dtype).cuda()
            elif self.arch == "ViT-B/16":
                self.adapter = Adapter(512, 4).to(self.clip_dtype).cuda()
        
        # optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=1e-3, eps=1e-4)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))

    def build_cache_model(self, train_loader_cache, load_cache=False, augment_epoch=10, shots='16'):
        cache_dir = os.path.join('tip_cache', 'imagenet_R')
        os.makedirs(cache_dir, exist_ok=True)

        if load_cache == False:    
            cache_keys = []
            cache_values = []

            with torch.no_grad():
                # Data augmentation for the cache model
                for augment_idx in range(augment_epoch):
                    train_features = []

                    print('Augment Epoch: {:} / {:}'.format(augment_idx, augment_epoch))
                    for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                        images = images.cuda()
                        image_features = self.image_encoder(images)
                        train_features.append(image_features)
                        if augment_idx == 0:
                            target = target.cuda()
                            cache_values.append(target)
                    cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
                
            cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
            cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
            cache_keys = cache_keys.permute(1, 0)
            cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

            torch.save(cache_keys, cache_dir + '/keys_' + shots + "shots.pt")
            torch.save(cache_values, cache_dir + '/values_' + shots + "shots.pt")

        else:
            cache_keys = torch.load(cache_dir + '/keys_' + shots + "shots.pt")
            cache_values = torch.load(cache_dir + '/values_' + shots + "shots.pt")

        return cache_keys, cache_values

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self, cache_keys=None):
        self.prompt_learner.reset()
        if cache_keys is not None:
            self.build_adapter(cache_keys)
        else:
            self.build_adapter()
        # adapter_params = self.adapter_init_state
        # self.adapter.load_state_dict(adapter_params)

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        prompts_group = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts_group
        t_features_group = []
        for i in range(len(prompts_group)):
            t_features = self.text_encoder(prompts_group[i], tokenized_prompts[i])
            t_features_group.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        # text_features = torch.stack(text_features, dim=0)

        return t_features_group

    def inference(self, image, cache_keys, cache_values):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
        if self.adapter is not None:
            x = self.adapter(image_features)
            image_features = self.ratio * x + (1 - self.ratio) * image_features

        text_features_group = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logits_all = []
        logit_scale = self.logit_scale.exp()
        for text_features in text_features_group:
            logits = logit_scale * image_features @ text_features.t()
            logits_all.append(logits)
        logits_all_tensor = torch.cat(logits_all, dim=0)
        
        if cache_keys is not None:
            beta = 1
            alpha = 1.17 * 16 / cache_keys.shape[1]
            # affinity = image_features @ cache_keys
            affinity = self.adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            logits = logits + cache_logits * alpha
            
            return logits, image_features

        return logits_all_tensor

    def forward(self, input, cache_keys=None, cache_values=None):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, cache_keys, cache_values)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False, num_prompts=1):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls, num_prompts=num_prompts)

    return model


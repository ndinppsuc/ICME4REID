import datetime
import os
import sys
import time
import warnings
import json
import re
import math
import torch
import torch.nn.functional as F
from copy import deepcopy
from typing import List
from PIL import Image
from easydict import EasyDict
from torch import nn
from tqdm import tqdm

import networkx as nx
import numpy as np
from openai import OpenAI
import concurrent.futures

from data.prompt import prompt_question_generator_v3, prompt_answer_generator
from data.prompt import wrap_question_prompt
from reid_datasets.bases import tokenize_simple
from utils.iotools import LoggerX
from utils.metrics import per_sample_ranks, max_discriminative
from model.IRRA import IRRA
from utils import misc
from .clip_model import convert_weights
from utils.simple_tokenizer import SimpleTokenizer

from .llava.conversation import conv_templates
from .llava.mm_utils import tokenizer_image_token, process_images
from .llava.train.train_utils import pad_sequence
from .llava.utils import rank0_print
from .llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN 
from .llava_reid_utils import delete_prefix, load_llava_reid_model, process_image_train
from .selector import Selector, SelectorConfig

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# AnswerGeneratorVLLM
# =============================================================================
class AnswerGeneratorVLLM:
    def __init__(self, base_url="http://localhost:8000/v1", model_name=None):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = model_name
        self.prompt_template = prompt_answer_generator

    def fuse_results(self, global_qa, local_qa):
        results = [""] * len(global_qa)
        
        def _fuse_one(g_qa, l_qa):
            # [修正] 更加安全的融合策略
            fusion_prompt = (
                f"You are updating the description of a target person for a search algorithm.\n"
                f"Basic Attributes: {g_qa[1]}\n"
                f"Fine-grained Details: {l_qa[1]}\n"
                f"Task: Combine these into a single, dense description sentence of the TARGET.\n"
                f"Crucial Rules:\n"
                f"1. DO NOT mention 'Image 1', 'Image 2' or 'Reference'.\n"
                f"2. DO NOT mention comparisons like 'unlike the other image'.\n"
                f"3. Only state the visual attributes confirmed present on the target.\n"
                f"4. If inputs conflict, trust the Basic Attributes."
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a concise fact summarizer."},
                        {"role": "user", "content": fusion_prompt}
                    ],
                    temperature=0.1, 
                    max_tokens=64
                )
                return response.choices[0].message.content.strip()
            except:
                # 兜底：只返回 Global 描述，放弃 Local 以防噪声
                return g_qa[1]

        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_idx = {
                executor.submit(_fuse_one, global_qa[i], local_qa[i]): i 
                for i in range(len(global_qa))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return results

    def _get_one_answer(self, context, question):
        try:
            # 保持简单的问答，不做 System Prompt 的 trick，防止微调模型不适
            c_converted = context.replace("I saw ", "You saw ")
            content = self.prompt_template.replace('<|context|>', c_converted).replace('<|question|>', question)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=0.0, max_tokens=64, top_p=1.0
            )
            return response.choices[0].message.content.strip().replace("assistant\n", "").strip()
        except: return ""

    def __call__(self, contexts, questions):
        results = [None] * len(contexts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_idx = {
                executor.submit(self._get_one_answer, contexts[i], questions[i]): i 
                for i in range(len(contexts))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                results[future_to_idx[future]] = future.result()
        return results

# =============================================================================
# Main Class
# =============================================================================
class LlavaForPersonReID(nn.Module):
    def __init__(self, config, llava_config=None, logger: LoggerX = None):
        super(LlavaForPersonReID, self).__init__()
        self.retriever_device = "cuda" if misc.is_dist_avail_and_initialized() else "cuda:0"
        self.interact_round = config.interact_round
        self.num_candidates = config.num_candidates
        self.q_length = config.max_question_length
        self.a_length = config.max_answer_length
        self.r_length = config.max_retrieve_length
        self.num_potential_candidates = 200
        self.stage = config.stage
        if llava_config is not None and "data_args" in llava_config:
            self.image_folder = llava_config["data_args"].image_folder
        
        if self.stage == "eval":
            self.retrieval_model = self._build_retrieval_model(config, self.retriever_device, False, logger)
            self.question_model = LlavaForPersonReIDQuestionModel(config, llava_config, logger)
            self.answer_model = AnswerGeneratorVLLM(base_url="http://localhost:8000/v1", model_name="/home/liuteli/reid/ijcai/LLaVA-ReID/Qwen/Qwen2.5-7B-Instruct")
            self.forward = self._inference
        else: pass

    # ... (省略辅助函数) ...
    def _train_retriever(self, batch): pass
    @torch.no_grad()
    def _prepare_data(self, batch): pass
    @torch.no_grad()
    def encode_text(self, text):
        with torch.amp.autocast('cuda', enabled=True):
            text_tokens = [tokenize_simple(t, self.clip_tokenizer, self.r_length) for t in text]
            text_tokens = torch.stack(text_tokens, dim=0).to(self.retriever_device)
            text_feats = F.normalize(self.retrieval_model.encode_text(text_tokens), dim=-1)
        return text_feats
    @torch.no_grad()
    def set_gallery(self, gallery_cls, gallery_image_path, **kwargs):
        self.gallery_cls = F.normalize(gallery_cls.to(self.retriever_device), dim=-1)
        self.gallery_path = gallery_image_path
        if "gallery_pid" in kwargs.keys(): self.gallery_pid = kwargs.pop("gallery_pid").to(self.retriever_device)
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        return self.question_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    def _build_retrieval_model(self, cfg, device, training=True, logger=None):
        retrieval_model_config = EasyDict({'pretrain_choice': cfg.clip_pretrain_model, 'img_size': cfg.img_size, 'stride_size': cfg.stride_size, 'text_length': cfg.max_retrieve_length, 'temperature': cfg.temperature, 'vocab_size': cfg.vocab_size, 'training': cfg.stage == "train_retriever", 'num_classes': cfg.num_classes})
        retrieval_model = IRRA(retrieval_model_config, device)
        ckpt_path = os.path.join(cfg.output_dir, 'retrieval_model_mix_IRRA', 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            model_state = torch.load(ckpt_path, map_location='cpu')
            retrieval_model.load_state_dict(model_state, strict=False)
        convert_weights(retrieval_model)
        retrieval_model.to(device).eval()
        self.clip_tokenizer = SimpleTokenizer()
        return retrieval_model
    @torch.no_grad()
    def get_topK_images(self, text_feats: torch.Tensor, k: torch.Tensor):
        N, max_K = text_feats.shape[0], self.num_potential_candidates
        if text_feats.ndim == 1: text_feats = text_feats.unsqueeze(0)
        similarity = text_feats @ self.gallery_cls.t()
        top_k_img_ids = torch.topk(similarity, max_K, dim=1, largest=True).indices
        top_k_img_feats = self.gallery_cls[top_k_img_ids]
        padding_mask = torch.arange(max_K).to(text_feats.device).unsqueeze(0).expand(N, max_K)
        padding_mask = torch.less(padding_mask, k.unsqueeze(1))
        top_k_img_ids[torch.logical_not(padding_mask)] = -1
        return top_k_img_ids, top_k_img_feats, padding_mask

    # -------------------------------------------------------------------------
    # Graph Logic
    # -------------------------------------------------------------------------
    def find_hard_negatives_via_graph(self, image_feats):
        """
        利用图结构找出 Rank 0 的最强邻居
        """
        B, K, D = image_feats.shape
        hard_negatives_indices = []

        for i in range(B):
            feats = image_feats[i] 
            sim_matrix = feats @ feats.T
            rank0_sims = sim_matrix[0]
            rank0_sims[0] = -1.0 # 排除自己
            hard_idx = torch.argmax(rank0_sims).item()
            hard_negatives_indices.append(hard_idx)
            
        return hard_negatives_indices

    @torch.no_grad()
    def _inference(self, batch):
        if "images" in batch: 
            return EasyDict({"image_feats": self.retrieval_model.encode_image(batch["images"].to(self.retriever_device))[:, 0, :], "image_id": batch["image_id"]})

        B = len(batch["initial_description"])
        initial_description = batch["initial_description"]
        collection = deepcopy(batch["initial_description"])
        questions_log = [[] for _ in range(B)]
        answers_log = [[] for _ in range(B)]
        ret_dict = EasyDict({"text_id": batch["text_id"], "text_feats": [], "conversations": []})
        for i in range(B): ret_dict.conversations.append({"id": batch['text_id'][i].item(), "initial_query": initial_description[i], "interaction": []})

        for r in tqdm(range(self.interact_round), desc="Graph-Guided ReID", file=sys.stdout, ncols=75):
            text_feats = self.encode_text(collection)
            k = torch.full([B], fill_value=math.ceil(self.num_potential_candidates / (r + 1)), device=self.retriever_device)
            
            image_indices, image_feats_topk, _ = self.get_topK_images(text_feats, k)
            image_indices_view = image_indices.view(B, -1)
            
            # Step 1: 寻找 Graph Neighbor
            hard_negative_offsets = self.find_hard_negatives_via_graph(image_feats_topk[:, :8, :])

            # Step 2: 准备双流数据
            global_paths = [] 
            local_paths = [] 

            for idx in range(B):
                indices = image_indices_view[idx]
                global_paths.append([self.gallery_path[indices[0]]]) # Stream A
                hard_idx = hard_negative_offsets[idx]
                local_paths.append([self.gallery_path[indices[0]], self.gallery_path[indices[hard_idx]]]) # Stream B

            # Step 3: 并行提问
            q_global = self.question_model.generate_questions(initial_description, global_paths, questions_log, answers_log, mode="global")
            q_local = self.question_model.generate_questions(initial_description, local_paths, questions_log, answers_log, mode="local")

            # Step 4: 并行回答
            a_global = self.answer_model(batch['fine-grained_description'], q_global)
            a_local = self.answer_model(batch['fine-grained_description'], q_local)

            # Step 5: 融合 (使用修正后的 Prompt)
            g_qa = list(zip(q_global, a_global))
            l_qa = list(zip(q_local, a_local))
            fused_updates = self.answer_model.fuse_results(g_qa, l_qa)

            collection = [(collection[i] + ' ' + delete_prefix(fused_updates[i])).strip() for i in range(B)]

            for i in range(B):
                questions_log[i].append(q_local[i])
                answers_log[i].append(fused_updates[i])
                ret_dict.conversations[i]["interaction"].append({
                    "round": r, "update": fused_updates[i]
                })

            ret_dict.text_feats.append(text_feats.unsqueeze(0))

        text_feats = self.encode_text(collection)
        ret_dict.text_feats.append(text_feats.unsqueeze(0))
        ret_dict.text_feats = torch.cat(ret_dict.text_feats, dim=0)
        
        if B > 0: print(f"\n[Case 0] Final Update: {fused_updates[0]}")
        return ret_dict

# =============================================================================
# Question Model (Prompt 修正版)
# =============================================================================
class LlavaForPersonReIDQuestionModel(nn.Module):
    def __init__(self, questioner_config, llava_args, logger=None):
        super(LlavaForPersonReIDQuestionModel, self).__init__()
        model, tokenizer = load_llava_reid_model(llava_args, logger)
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_processor = self.model.get_vision_tower().image_processor
        self.num_candidates = questioner_config.num_candidates
        self.conv_template = conv_templates['qwen_reid']
        self.prompt_q = prompt_question_generator_v3
        self.mini_batch_size = 4
        self.stage = questioner_config.stage

    def forward(self, *args, **kwargs): return self.model(*args, **kwargs)

    def generate_questions(self, initial_query, candidate_images, questions_log, answers_log, mode="global"):
        B = len(initial_query)
        question_all = []
        
        # [修正] 
        # Global: 传统的 ReID 问题，微调模型最擅长
        prompt_global = "Describe the person's gender, clothing color, and any visible bags. Question: What are the main attributes?"
        
        # Local: 引导模型描述 Image 1，利用 Image 2 作为背景参考，但不强行对比
        prompt_local = (
            "Look at Image 1 (Target). "
            "Describe the specific details of Image 1, such as logos, shoe type, or bag strap style. "
            "Question: What are the fine-grained visual details of Image 1?"
        )

        for idx in range(0, B, self.mini_batch_size):
            end_idx = min(idx + self.mini_batch_size, B)
            batch_query = initial_query[idx:end_idx]
            batch_images = candidate_images[idx:end_idx]
            batch_q_log = questions_log[idx:end_idx]
            batch_a_log = answers_log[idx:end_idx]
            
            image_all = [Image.open(img) for sublist in batch_images for img in sublist]
            image_tensor = process_images(image_all, self.image_processor, self.model.config)
            image_tensor = [_image.to(dtype=torch.bfloat16) for _image in image_tensor]
            image_sizes = [_image.size for _image in image_all]

            if mode == "local":
                prompts_batch = [prompt_local for _ in batch_query]
            else:
                prompts_batch = wrap_question_prompt(self.prompt_q, batch_query, batch_q_log, batch_a_log, len(batch_images[0]), True)

            input_ids_list = []
            for i in range(len(batch_query)):
                conv = deepcopy(self.conv_template)
                curr_prompt = prompts_batch[i]
                if DEFAULT_IMAGE_TOKEN not in curr_prompt: inp = DEFAULT_IMAGE_TOKEN + '\n' + curr_prompt
                else: inp = curr_prompt
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt_text = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt_text, self.tokenizer, return_tensors='pt')
                input_ids_list.append(input_ids)

            input_ids_batch = pad_sequence(self.tokenizer, input_ids_list, True, self.tokenizer.pad_token_id)
            attention_mask = input_ids_batch.ne(self.tokenizer.pad_token_id).to(self.model.device)
            input_ids_batch = input_ids_batch.to(self.model.device)

            with torch.inference_mode():
                gen_ids = self.model.generate(
                    input_ids_batch, images=image_tensor, image_sizes=image_sizes,
                    modalities=["image"] * len(image_all), attention_mask=attention_mask,
                    do_sample=False, temperature=None, max_new_tokens=100,
                    begin_suppress_tokens=[self.tokenizer.eos_token_id]
                )
            questions = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            question_all.extend([q.replace("assistant\n", "").strip() for q in questions])
        return question_all

    def get_model(self): return self.model

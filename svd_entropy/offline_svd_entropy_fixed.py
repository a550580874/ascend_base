#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
from functools import partial
import re
from collections import OrderedDict
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


try:
    import torch_npu  # noqa: F401
    device = 'npu'
except Exception:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


MAX_SLICE = 100000000

FLOAT_DTYPES = {
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
}

SKIP_TOP_KEYS = {
    "args",
    "optimizer",
    "opt_param_scheduler",
    "rng_state",
    "checkpoint_version",
    "iteration",
    "num_floating_point_operations_so_far",
}


def max_eigenvalue(input: torch.Tensor, num_iterations=3):
    input = input.float()
    in_features = input.shape[1]
    u = torch.randn(in_features).to(input.device)
    u = u / u.norm()
    input_seq = torch.matmul(input.T, input)
    for _ in range(num_iterations):
        v = torch.matmul(input_seq, u)
        spectral_norm = torch.matmul(v.T, u)
        u = v / v.norm()
    return spectral_norm.sqrt()


@torch.no_grad()
def cal_sr(module_name, **kwargs):
    result = OrderedDict()
    for ckpt, param in kwargs.items():
        if ckpt.startswith('ckpt'):
            if param.ndim == 2:
                eig = max_eigenvalue(param)
                f_norm = torch.norm(param, p="fro")
                sr = (f_norm / eig).item()
            elif param.ndim == 3:
                sr = []
                for i in range(param.shape[0]):
                    eig = max_eigenvalue(param[i, ...])
                    f_norm = torch.norm(param[i, ...], p="fro")
                    sr.append((f_norm / eig).item())
            else:
                sr = float('nan')
            result[ckpt] = sr
    return result


@torch.no_grad()
def svd_entropy(module_name, **kwargs):
    result = OrderedDict()
    for ckpt, param in kwargs.items():
        if ckpt.startswith('ckpt'):
            if param.ndim == 2:
                q = min(200, min(param.shape))
                u, s, v = torch.svd_lowrank(param.float(), q=q)
                p = s / torch.sum(s)
                entropy = -torch.sum(p * torch.log2(p)).item()
            elif param.ndim == 3:
                entropy = []
                for i in range(param.shape[0]):
                    q = min(200, min(param[i, ...].shape))
                    u, s, v = torch.svd_lowrank(param[i, ...].float(), q=q)
                    p = s / torch.sum(s)
                    entropy.append(-torch.sum(p * torch.log2(p)).item())
            else:
                entropy = float('nan')
            result[ckpt] = entropy
    return result


@torch.no_grad()
def router_weight_similarity(module_name, **kwargs):
    result = OrderedDict()
    for ckpt, w in kwargs.items():
        if ckpt.startswith('ckpt'):
            if w.ndim == 2 and 'router' in module_name:
                d, n = w.shape
                if d < n:
                    w = w.T
                    d, n = w.shape
                normalized_w = F.normalize(w, p=2, dim=0)
                cos_sim = normalized_w.T @ normalized_w
                mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=w.device), diagonal=1)
                value = cos_sim[mask].mean().item()
            else:
                value = float('nan')
            result[ckpt] = value
    return result


def plt_plot(x, y, ylabel, title, save_path):
    plt.plot(x, y, 'b-')
    plt.xlabel('ckpt')
    plt.ylabel(ylabel, color='b')
    plt.grid(True)
    plt.title(title, size=10)
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(save_path, f'{title}_{ylabel}.png'))
    plt.show()
    plt.cla()
    plt.clf()


def plot(results, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(12, 8))

    for layer, value in results.items():
        list_flag = False
        for metric, result in value.items():
            x = []
            y = []
            all_y = []

            for ckpt, v in result.items():
                x.append(ckpt)
                if isinstance(v, list):
                    list_flag = True
                    all_y.append(v)
                else:
                    y.append(v)

            if list_flag:
                all_y = np.array(all_y).T.tolist()
                for i, y in enumerate(all_y):
                    plt_plot(x, y, metric, f"{layer}_{i}", save_path)
            else:
                plt_plot(x, y, metric, layer, save_path)


TREND_FUNC = {
    'svd_entropy': svd_entropy,
    'stable_rank': cal_sr,
    'router_weight_similarity': router_weight_similarity,
}


FRAMEWORK_LOADFUNC = {
    "modellink": partial(torch.load, map_location='cpu'),
    'safetensors': partial(load_file, device='cpu'),
    'default': partial(torch.load, map_location='cpu'),
    'bin': partial(torch.load, map_location='cpu'),
    'deepspeed': partial(torch.load, map_location='cpu'),
    'megatron': partial(torch.load, map_location='cpu', weights_only=False),
}


FRAMEWORK_SUFFIX = {
    'modellink': '.pt',
    'safetensors': '.safetensors',
    'default': 'rng.pt',
    'deepspeed': 'model_states.pt',
    'bin': '.bin',
    'megatron': '.pt',
}


def is_float_tensor(x):
    return torch.is_tensor(x) and x.dtype in FLOAT_DTYPES


def is_target_module(current_module, target_modules):
    match_modules = []
    for tm in target_modules:
        if re.search(tm, current_module):
            match_modules.append(tm)
    return match_modules


def load_keys(ckpt, target_modules, modules, prefix='', count=0):
    """
    从加载后的 checkpoint 对象里递归找目标 tensor。

    修复点：
    1. 原版本只支持 dict.items()
    2. 现在支持 dict / OrderedDict / list / tuple / Tensor
    3. 避免 list 对象触发 AttributeError: 'list' object has no attribute 'items'
    """
    if is_float_tensor(ckpt):
        current_path = prefix
        match_modules = is_target_module(current_path, target_modules)
        if match_modules:
            if current_path not in modules:
                modules[current_path] = OrderedDict()
            modules[current_path][f'ckpt{count}'] = ckpt
        return

    if isinstance(ckpt, (OrderedDict, dict)):
        for k, v in ckpt.items():
            current_path = f"{prefix}.{str(k)}" if prefix else str(k)
            if isinstance(v, (OrderedDict, dict, list, tuple)) or torch.is_tensor(v):
                load_keys(v, target_modules, modules, prefix=current_path, count=count)
        return

    if isinstance(ckpt, (list, tuple)):
        for i, v in enumerate(ckpt):
            current_path = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(v, (OrderedDict, dict, list, tuple)) or torch.is_tensor(v):
                load_keys(v, target_modules, modules, prefix=current_path, count=count)
        return

    return


def trend(param_pair, metrics, layer):
    res = {}
    for m in metrics:
        res[m] = TREND_FUNC[m](layer, **param_pair)
    return res


def add_tensor_leaf(state_dict, key, value, source_tag=""):
    """
    添加 tensor 到 state_dict。

    优先保留原始参数名，避免破坏 config 里类似 ^decoder.layers 的正则。
    如果多 rank 下 key 冲突，再在后面加来源后缀。
    """
    if not key:
        key = f"tensor_{len(state_dict)}"

    if key not in state_dict:
        state_dict[key] = value
        return

    if source_tag:
        tagged_key = f"{key}__{source_tag}"
    else:
        tagged_key = key

    if tagged_key not in state_dict:
        state_dict[tagged_key] = value
        return

    idx = 1
    while f"{tagged_key}__dup{idx}" in state_dict:
        idx += 1

    state_dict[f"{tagged_key}__dup{idx}"] = value


def collect_tensor_leaves(obj, state_dict, prefix="", source_tag=""):
    """
    递归收集浮点 Tensor 叶子节点。

    支持：
      dict / OrderedDict / list / tuple / Tensor

    只收集模型浮点权重，跳过 optimizer / rng / args 等非模型状态。
    """
    if is_float_tensor(obj):
        add_tensor_leaf(state_dict, prefix, obj, source_tag=source_tag)
        return

    if isinstance(obj, (OrderedDict, dict)):
        for k, v in obj.items():
            if str(k) in SKIP_TOP_KEYS:
                continue

            next_prefix = f"{prefix}.{str(k)}" if prefix else str(k)
            collect_tensor_leaves(v, state_dict, prefix=next_prefix, source_tag=source_tag)
        return

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            next_prefix = f"{prefix}.{i}" if prefix else str(i)
            collect_tensor_leaves(v, state_dict, prefix=next_prefix, source_tag=source_tag)
        return

    return


def _load(path, state_dict, suffix, load_func):
    if os.path.isfile(path):
        if not path.endswith(suffix):
            return

        base = os.path.basename(path)

        # SVD entropy 只分析模型权重，不分析分布式优化器 flat buffer。
        if base == "distrib_optim.pt":
            print(f"[SKIP] optimizer flat buffer: {path}")
            return

        print(f"[LOAD] {path}")

        p = load_func(path)

        # Megatron 常见格式：
        # {
        #   "model": ...,
        #   "optimizer": ...,
        #   "rng_state": ...
        # }
        # 这里只分析 model。
        if isinstance(p, (OrderedDict, dict)) and "model" in p:
            p = p["model"]

        before = len(state_dict)

        rank_dir = os.path.basename(os.path.dirname(path))
        file_stem = os.path.splitext(os.path.basename(path))[0]
        source_tag = f"{rank_dir}.{file_stem}" if rank_dir else file_stem

        collect_tensor_leaves(
            p,
            state_dict,
            prefix="",
            source_tag=source_tag,
        )

        after = len(state_dict)

        print(
            f"[LOAD_DONE] {path}, "
            f"added_tensors={after - before}, "
            f"total_tensors={after}"
        )

    elif os.path.isdir(path):
        for sub_path in sorted(os.listdir(path)):
            _load(os.path.join(path, sub_path), state_dict, suffix, load_func)


def load_state(state_path, framework=None):
    state_dict = {}

    if framework not in FRAMEWORK_LOADFUNC:
        framework = 'default'
    elif framework == 'modellink':
        import modellink  # noqa: F401
    elif framework == 'megatron':
        import megatron  # noqa: F401

    suffix = FRAMEWORK_SUFFIX[framework]
    load_func = FRAMEWORK_LOADFUNC[framework]

    _load(state_path, state_dict, suffix, load_func)
    return state_dict


def output_path_check(path, overwrite=False):
    path = os.path.abspath(path)

    if os.path.isfile(path):
        path = os.path.dirname(path)
    elif not os.path.isdir(path):
        path = os.path.join(os.getcwd(), path)
    elif path == '':
        print('result will not be saved')
        return None

    if os.path.exists(path) and not overwrite:
        raise FileNotFoundError(f'dir of the output path {path} does exist')
    else:
        os.makedirs(path, exist_ok=True)
        print(f'results path: {path}')

    return path


def model_trend(match_layers, trend_metrics, save_path):
    results = {}

    for layer, v in match_layers.items():
        results[layer] = trend(v, trend_metrics, layer)

    print(json.dumps(results, indent=4))

    if save_path is not None:
        plot(results, save_path)
        json.dump(
            results,
            open(os.path.join(save_path, 'trend_result.json'), 'w'),
            indent=4,
        )

    return results


def offline_svd_entropy(config_path='config_local.json'):
    print(config_path)

    if isinstance(config_path, str) and os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"config file not found: {config_path}")

    target_modules = config.get("modules", [])
    trend_metrics = config.get("trend_metrics", [])
    framework = config.get("framework", None)
    overwrite = config.get("overwrite", False)

    save_path = output_path_check(config.get("save_path", ''), overwrite)

    match_layers = {}

    for i, ckpt_file in enumerate(config["ckpt_list"]):
        ckpt = load_state(ckpt_file, framework)
        print(f"[CKPT_DONE] index={i}, file={ckpt_file}, loaded_tensors={len(ckpt)}")

        load_keys(ckpt, target_modules, match_layers, count=i)
        print(f"[MATCH_DONE] index={i}, matched_layers={len(match_layers)}")

    if trend_metrics:
        model_trend(match_layers, trend_metrics, save_path)

    print('fin')
    return


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_local.json"
    offline_svd_entropy(config_path)

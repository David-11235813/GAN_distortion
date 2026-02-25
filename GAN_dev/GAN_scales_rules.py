import torch
from typing import Mapping


def first_scale_is_105proc() -> Mapping[int, float]:
    return {0: 1.05}

def scales_2_and_4_are_90proc() -> Mapping[int, float]:
    return {2: 0.9, 4: 0.9}


def combine_by_rule(trained: torch.Tensor, rule: Mapping[int, float]) -> torch.Tensor:
    trained = trained.reshape(-1)
    k = len(rule)
    out_len = trained.numel() + k
    if k == 0:
        return trained.clone()
    pos_set = set(rule.keys())
    if len(pos_set) != k:
        raise ValueError("duplicate rule positions")
    if any((p < 0 or p >= out_len) for p in pos_set):
        raise ValueError("rule position out of range for resulting length")
    out = torch.empty(out_len, dtype=trained.dtype, device=trained.device)
    # place rule values (cast to same dtype/device as b)
    for p, v in rule.items():
        out[p] = torch.as_tensor(v, dtype=out.dtype, device=out.device)
    # fill remaining slots with b (in order)
    bi = 0
    for i in range(out_len):
        if i in pos_set:
            continue
        out[i] = trained[bi]
        bi += 1
    return out

#print(len(first_feature_times_105proc()))
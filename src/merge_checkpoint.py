import os
import torch

checkpoint0 = "runs/roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune-DepthLiberoDataset/ckpt/global_step9058/mp_rank_00_model_states.pt"
checkpoint1 = "runs/pred_depth/ckpt/global_step22645/mp_rank_00_model_states.pt"
merge_checkpoint = "runs/roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune-DepthLiberoDataset/ckpt/merge_ckpt/mp_rank_00_model_states.pt"

ckpt0 = torch.load(checkpoint0, map_location="cpu")
ckpt0_keys = set(ckpt0["module"].keys())
ckpt1 = torch.load(checkpoint1, map_location="cpu")
ckpt1_keys = set(ckpt1["module"].keys())

intersection = ckpt0_keys & ckpt1_keys
for key in intersection:
    assert torch.allclose(ckpt0["module"][key], ckpt1["module"][key])
merge_keys = list(filter(
    lambda x: x.startswith("depth_pred"),
    ckpt1_keys
))
ckpt0["module"].update({
    key: ckpt1["module"][key]
    for key in merge_keys
})
print(ckpt1_keys - set(ckpt0["module"].keys()))
torch.save(ckpt0, merge_checkpoint)

with open("runs/roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune-DepthLiberoDataset/ckpt/latest", "w") as fo:
    fo.write("merge_ckpt")

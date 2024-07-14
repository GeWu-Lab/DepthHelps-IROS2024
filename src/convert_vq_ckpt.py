import torch

checkpoint0 = "runs/roboflamingo-mpt_3b_depth-DepthLiberoDataset/ckpt/global_step22645/mp_rank_00_model_states.pt"
checkpoint1 = "runs/vq/ckpt/epoch4.pt"

ckpt0 = torch.load(checkpoint0, map_location="cpu")
ckpt0_keys = set(ckpt0["module"].keys())
ckpt1 = torch.load(checkpoint1, map_location="cpu")
ckpt1_keys = set(ckpt1.keys())
print(ckpt1_keys)

ckpt = {
    "model_state_dict": {}
}
ckpt["model_state_dict"].update({
    f"module.{key}": ckpt0["module"][key]
    for key in ckpt0_keys
})
ckpt["model_state_dict"].update({
    f"module.depth_vq.{key}": ckpt1[key]
    for key in ckpt1_keys
})

torch.save(ckpt, "runs/vq/ckpt/merge_ckpt/epoch4.pt")

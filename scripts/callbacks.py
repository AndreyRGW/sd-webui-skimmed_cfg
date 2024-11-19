# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import scripts.skimmed_CFG as CFG

from modules import script_callbacks, shared, devices
from modules.script_callbacks import CFGDenoiserParams

def pad_tokens_to_match(conds, unconds):
    """Pad tokens to match lengths between conditional and unconditional"""
    if not isinstance(conds, dict) and not isinstance(unconds, dict):
        if conds.shape[1] > unconds.shape[1]:
            # Pad unconds to match conds length
            pad_length = conds.shape[1] - unconds.shape[1]
            unconds = torch.cat([unconds, unconds[:, -1:].repeat(1, pad_length, 1)], dim=1)
        elif unconds.shape[1] > conds.shape[1]:
            # Pad conds to match unconds length  
            pad_length = unconds.shape[1] - conds.shape[1]
            conds = torch.cat([conds, conds[:, -1:].repeat(1, pad_length, 1)], dim=1)
    return conds, unconds
        
def on_cfg_denoiser(params: CFGDenoiserParams):
    if not shared.opts.data.get("skimmed_cfg_enabled", False):
        return
    
    with devices.autocast():
        x_orig = params.x
        conds = params.text_cond
        unconds = params.text_uncond
        cond_scale = params.denoiser.cond_scale_miltiplier * (getattr(params.denoiser, 'cfg_scale', 7.5) or 7.5)

        if unconds is not None:
            # Extract cross-attention if needed
            if isinstance(unconds, dict):
                unconds = unconds["crossattn"]
            if isinstance(conds, dict):
                conds = conds["crossattn"]

            # Pad tokens to match lengths
            conds, unconds = pad_tokens_to_match(conds, unconds)
            
            # Apply skimmed CFG
            out_unconds = CFG.skimmed_CFG(
                x_orig=unconds,
                cond=conds, 
                uncond=unconds,
                cond_scale=cond_scale,
                skimming_scale=shared.opts.data.get("skimming_scale", 7.0),
                disable_flipping_filter=shared.opts.data.get("disable_flipping_filter", False)
            )
            
            # Update result
            if isinstance(params.text_uncond, dict):
                params.text_uncond["crossattn"] = out_unconds
            else:
                params.text_uncond = out_unconds

script_callbacks.on_cfg_denoiser(on_cfg_denoiser)

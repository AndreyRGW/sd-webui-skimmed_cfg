import torch

MAX_SCALE = 10
STEP_STEP = 2

@torch.no_grad()
def get_skimming_mask(x_orig, cond, uncond, cond_scale, return_denoised=False, disable_flipping_filter=False):
    denoised = x_orig - ((x_orig - uncond) + cond_scale * ((x_orig - cond) - (x_orig - uncond)))
    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = cond.sign() == (cond * cond_scale - uncond * (cond_scale - 1)).sign()

    if disable_flipping_filter:
        outer_influence = matching_pred_signs & matching_diff_after
    else:
        deviation_influence = (denoised.sign() == (denoised - x_orig).sign())
        outer_influence = matching_pred_signs & matching_diff_after & deviation_influence

    if return_denoised:
        return outer_influence, denoised
    else:
        return outer_influence

@torch.no_grad()
def skimmed_CFG(x_orig, cond, uncond, cond_scale, skimming_scale, disable_flipping_filter=False):
    outer_influence, denoised = get_skimming_mask(x_orig, cond, uncond, cond_scale, True, disable_flipping_filter)
    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * ((x_orig - cond) - (x_orig - uncond)))
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] = cond[outer_influence] - (low_cfg_denoised_outer_difference[outer_influence] / cond_scale)
    return cond

@torch.no_grad()
def interpolated_scales(x_orig, cond, uncond, cond_scale, small_scale, squared=False, root_dist=False):
    deltacfg_normal = x_orig - cond_scale  * cond - (cond_scale  - 1) * uncond
    deltacfg_small  = x_orig - small_scale * cond - (small_scale - 1) * uncond
    absdiff = (deltacfg_normal - deltacfg_small).abs()
    absdiff = (absdiff-absdiff.min()) / (absdiff.max()-absdiff.min())
    if squared:
        absdiff = absdiff ** 2
    elif root_dist:
        absdiff = absdiff ** 0.5
    new_scale  = (small_scale - 1) / (cond_scale - 1)
    smaller_uncond = cond * (1 - new_scale) + uncond * new_scale
    new_uncond = smaller_uncond * (1 - absdiff) + uncond * absdiff
    return new_uncond

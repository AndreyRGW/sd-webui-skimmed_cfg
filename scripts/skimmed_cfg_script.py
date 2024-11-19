import gradio as gr

from modules import scripts, shared
import scripts.skimmed_CFG as CFG

class SkimmedCFGScript(scripts.Script):
    def title(self):
        return "Skimmed CFG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            with gr.Row():
                enabled = gr.Checkbox(
                    label="Enable", 
                    value=False
                )
                skimming_scale = gr.Slider(
                    minimum=0.0,
                    maximum=CFG.MAX_SCALE,
                    step=1.0,
                    value=7.0,
                    label="Skimming CFG Scale"
                )

            # with gr.Row():
            #     full_skim_negative = gr.Checkbox(
            #         label="Full Skim Negative",
            #         value=False
            #     )
            #     disable_flipping_filter = gr.Checkbox(
            #         label="Disable Flipping Filter",
            #         value=False
            #     )

        self.infotext_fields = [
            (enabled, "Skimmed CFG Enabled"),
            (skimming_scale, "Skimming CFG Scale"),
            # (full_skim_negative, "Full Skim Negative"),
            # (disable_flipping_filter, "Disable Flipping Filter"),
        ]

        # return [enabled, skimming_scale, full_skim_negative, disable_flipping_filter]
        return [enabled, skimming_scale]

    def process(self, p, enabled, skimming_scale, full_skim_negative, disable_flipping_filter):
        shared.opts.data.pop("skimmed_cfg_enabled", None)
        shared.opts.data.pop("skimming_scale", None)
        # shared.opts.data.pop("full_skim_negative", None)
        # shared.opts.data.pop("disable_flipping_filter", None)

        if not enabled:
            return

        shared.opts.data["skimmed_cfg_enabled"] = enabled
        shared.opts.data["skimming_scale"] = skimming_scale
        # shared.opts.data["full_skim_negative"] = full_skim_negative
        # shared.opts.data["disable_flipping_filter"] = disable_flipping_filter

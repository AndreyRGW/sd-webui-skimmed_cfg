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

        self.infotext_fields = [
            (enabled, "Skimmed CFG Enabled"),
            (skimming_scale, "Skimming CFG Scale"),
        ]

        return [enabled, skimming_scale]

    def process(self, p, enabled, skimming_scale):
        shared.opts.data.pop("skimmed_cfg_enabled", None)
        shared.opts.data.pop("skimming_scale", None)

        if not enabled:
            return

        shared.opts.data["skimmed_cfg_enabled"] = enabled
        shared.opts.data["skimming_scale"] = skimming_scale

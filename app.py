import numpy as np
import gradio as gr
from PIL import Image
import onnxruntime as ort

sess = ort.InferenceSession("mobilevit_qat.onnx")
inp_name = sess.get_inputs()[0].name

MEAN = np.float32([0.485, 0.456, 0.406])
STD = np.float32([0.229, 0.224, 0.225])


def predict(image):
    if image is None:
        return bar_html(50, 50)

    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    img = img.convert("RGB").resize((224, 224))

    x = np.float32(np.array(img)) / 255
    x = (x - MEAN) / STD
    x = x.transpose(2, 0, 1)[None, ...]

    out = sess.run(None, {inp_name: x})[0].ravel()
    p = np.exp(out - out.max())
    p /= p.sum()

    return bar_html(float(p[1]) * 100, float(p[0]) * 100)


def bar_html(person, bg):
    label = "PERSON" if person > bg else "NO PERSON"
    clr = "#4ade80" if person > bg else "#f87171"
    return f"""
<div style="font-family:sans-serif;padding:18px">
 <p style="text-align:center;font-size:1.4em;font-weight:700;color:{clr};margin:0 0 18px">{label}</p>
 <div style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
   <span style="color:#ccc">Person</span>
   <span style="color:#4ade80;font-weight:700;font-variant-numeric:tabular-nums">{person:.1f}%</span>
  </div>
  <div style="background:#1e293b;border-radius:8px;height:16px;overflow:hidden">
   <div style="width:{person:.1f}%;height:100%;background:#4ade80;border-radius:8px;transition:width .3s"></div>
  </div>
 </div>
 <div>
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
   <span style="color:#ccc">Background</span>
   <span style="color:#94a3b8;font-weight:700;font-variant-numeric:tabular-nums">{bg:.1f}%</span>
  </div>
  <div style="background:#1e293b;border-radius:8px;height:16px;overflow:hidden">
   <div style="width:{bg:.1f}%;height:100%;background:#64748b;border-radius:8px;transition:width .3s"></div>
  </div>
 </div>
</div>"""


with gr.Blocks(theme=gr.themes.Soft(), title="VWW MobileViT") as demo:
    gr.Markdown("# Visual Wake Words — MobileViT-XXS\nPerson detection trained on COCO VWW &middot; QAT pipeline &middot; ONNX Runtime")

    with gr.Tabs():
        with gr.TabItem("Webcam"):
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, label="Feed")
                cam_out = gr.HTML(bar_html(50, 50))
            cam.stream(predict, cam, cam_out, time_limit=120)

        with gr.TabItem("Upload"):
            with gr.Row():
                img_in = gr.Image(type="pil", label="Image")
                img_out = gr.HTML(bar_html(50, 50))
            img_in.change(predict, img_in, img_out)

    with gr.Accordion("About", open=False):
        gr.Markdown(
            "MobileViT-XXS with quantization-aware training on Visual Wake Words (COCO 2014). "
            "Conv2d layers trained with per-channel INT8 fake quantisation; transformer blocks kept at FP32. "
            "Exported as ONNX. [Paper](https://arxiv.org/abs/2110.02178)"
        )

demo.launch()

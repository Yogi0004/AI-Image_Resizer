import gradio as gr
import cv2
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------
# IMAGE RESIZER FUNCTION
# ------------------------------------------------
def resize_image(
    image,
    resize_mode,
    width,
    height,
    scale_factor,
    interpolation
):
    # Convert RGB (Gradio) → BGR (OpenCV)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Interpolation mapping
    interp_map = {
        "Nearest": cv2.INTER_NEAREST,
        "Linear": cv2.INTER_LINEAR,
        "Cubic": cv2.INTER_CUBIC,
        "Lanczos": cv2.INTER_LANCZOS4,
        "Linear Exact": cv2.INTER_LINEAR_EXACT,
        "Area": cv2.INTER_AREA,
    }

    interp = interp_map[interpolation]

    # Resize logic
    if resize_mode == "Fixed Width & Height":
        output = cv2.resize(img, (width, height), interpolation=interp)

    elif resize_mode == "Scale Factor":
        output = cv2.resize(
            img,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=interp
        )
    else:
        output = img

    # Convert back to RGB
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


# ------------------------------------------------
# GRADIO UI
# ------------------------------------------------
with gr.Blocks(title="AI Image Resizer Tool") as demo:
    gr.Markdown(
        """
        #      AI Image Resizer Tool
        ✔ Resize images using exact parameters 
        ✔ Fast & Accurate
        """
    )

    with gr.Row():
        input_img = gr.Image(type="numpy", label="Upload Image")

        output_img = gr.Image(label="Resized Output")

    resize_mode = gr.Radio(
        choices=["Fixed Width & Height", "Scale Factor"],
        value="Fixed Width & Height",
        label="Resize Mode"
    )

    with gr.Row():
        width = gr.Slider(
            minimum=64,
            maximum=2048,
            value=512,
            step=1,
            label="Width (px)"
        )

        height = gr.Slider(
            minimum=64,
            maximum=2048,
            value=512,
            step=1,
            label="Height (px)"
        )

    scale_factor = gr.Slider(
        minimum=0.1,
        maximum=4.0,
        value=1.0,
        step=0.1,
        label="Scale Factor (used only in Scale mode)"
    )

    interpolation = gr.Radio(
        choices=["Nearest", "Linear", "Cubic", "Lanczos"],
        value="Lanczos",
        label="Interpolation Method"
    )

    btn = gr.Button("Resize Image")

    btn.click(
        fn=resize_image,
        inputs=[
            input_img,
            resize_mode,
            width,
            height,
            scale_factor,
            interpolation
        ],
        outputs=output_img
    )

# ------------------------------------------------
# RUN APP
# ------------------------------------------------
demo.launch(inbrowser=True)

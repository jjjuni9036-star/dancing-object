import os
import math
import random
import tempfile

import gradio as gr
import numpy as np
import imageio.v2 as imageio
from PIL import Image


MAX_SIZE = 700


def resize_image(image):
    image = image.convert("RGBA")
    w, h = image.size

    if max(w, h) <= MAX_SIZE:
        return image

    ratio = MAX_SIZE / max(w, h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def make_canvas(obj_img, bg_mode="white"):
    obj_img = resize_image(obj_img)

    ow, oh = obj_img.size

    canvas_w = max(ow + 220, 500)
    canvas_h = max(oh + 220, 500)

    if bg_mode == "black":
        bg_color = (0, 0, 0, 255)
    elif bg_mode == "gray":
        bg_color = (240, 240, 240, 255)
    else:
        bg_color = (255, 255, 255, 255)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    return obj_img, canvas


def transform_object(obj_img, angle=0, scale=1.0):
    w, h = obj_img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = obj_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    rotated = resized.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True
    )
    return rotated


def paste_center(canvas, obj_img, center_x, center_y):
    result = canvas.copy()

    w, h = obj_img.size
    x = int(center_x - w / 2)
    y = int(center_y - h / 2)

    result.alpha_composite(obj_img, (x, y))
    return result


def motion(action, frame_idx, total_frames):
    phase = 2 * math.pi * frame_idx / total_frames

    if action == "dance":
        dx = 28 * math.sin(2 * phase)
        dy = 10 * math.sin(4 * phase)
        angle = 14 * math.sin(2 * phase)
        scale = 1.0 + 0.03 * math.sin(4 * phase)

    elif action == "jump":
        dx = 0
        dy = -55 * abs(math.sin(phase))
        angle = 6 * math.sin(2 * phase)
        scale = 1.0 + 0.04 * abs(math.sin(phase))

    elif action == "spin":
        dx = 0
        dy = 0
        angle = 360 * frame_idx / total_frames
        scale = 1.0

    elif action == "wiggle":
        dx = 30 * math.sin(6 * phase)
        dy = 0
        angle = 18 * math.sin(6 * phase)
        scale = 1.0

    elif action == "bounce":
        dx = 0
        dy = 18 * math.sin(4 * phase)
        angle = 0
        scale = 1.0 + 0.08 * abs(math.sin(4 * phase))

    elif action == "runaway":
        dx = 90 * frame_idx / total_frames - 45
        dy = -10 * math.sin(6 * phase)
        angle = 10 * math.sin(8 * phase)
        scale = 1.0

    elif action == "surprise":
        dx = 8 * math.sin(10 * phase)
        dy = -18 * abs(math.sin(4 * phase))
        angle = 10 * math.sin(10 * phase)
        scale = 1.0 + 0.10 * abs(math.sin(4 * phase))

    else:
        dx = 0
        dy = 0
        angle = 0
        scale = 1.0

    return dx, dy, angle, scale


def generate_animation(image, action_mode, duration, fps, bg_mode, progress=gr.Progress()):
    if image is None:
        raise gr.Error("이미지를 먼저 업로드해주세요.")

    progress(0.05, desc="이미지를 준비하고 있습니다...")

    action_map = {
        "춤추기": "dance",
        "점프하기": "jump",
        "회전하기": "spin",
        "흔들기": "wiggle",
        "통통 튀기": "bounce",
        "도망가기": "runaway",
        "놀라기": "surprise"
    }

    action_list = list(action_map.values())

    if action_mode == "랜덤":
        action = random.choice(action_list)
    else:
        action = action_map.get(action_mode, "dance")

    obj_img, canvas = make_canvas(image, bg_mode=bg_mode)

    cw, ch = canvas.size
    center_x = cw / 2
    center_y = ch / 2

    total_frames = max(8, min(int(duration * fps), 60))

    frames = []

    for i in range(total_frames):
        dx, dy, angle, scale = motion(action, i, total_frames)

        transformed = transform_object(
            obj_img,
            angle=angle,
            scale=scale
        )

        frame = paste_center(
            canvas,
            transformed,
            center_x + dx,
            center_y + dy
        )

        frames.append(np.array(frame.convert("RGB")))

        progress(
            0.15 + 0.6 * (i / total_frames),
            desc="애니메이션을 만들고 있습니다..."
        )

    temp_dir = tempfile.mkdtemp()
    gif_path = os.path.join(temp_dir, "result.gif")
    mp4_path = os.path.join(temp_dir, "result.mp4")

    progress(0.8, desc="GIF를 저장하고 있습니다...")
    imageio.mimsave(
        gif_path,
        frames,
        duration=1 / fps,
        loop=0
    )

    progress(0.9, desc="MP4를 저장하고 있습니다...")
    writer = imageio.get_writer(
        mp4_path,
        fps=fps,
        codec="libx264",
        quality=6,
        macro_block_size=2
    )

    for frame in frames:
        writer.append_data(frame)

    writer.close()

    action_name_kor = {
        "dance": "춤추기",
        "jump": "점프하기",
        "spin": "회전하기",
        "wiggle": "흔들기",
        "bounce": "통통 튀기",
        "runaway": "도망가기",
        "surprise": "놀라기"
    }

    progress(1.0, desc="완료!")

    return gif_path, mp4_path, f"완성되었습니다. 선택된 행동: {action_name_kor[action]}"


with gr.Blocks(title="사진 속 물체 춤추기 AI") as demo:
    gr.Markdown("""
    # 🕺 사진 속 물체 춤추기 AI (초경량 버전)

    이 버전은 매우 빠르게 작동합니다.  
    단, **업로드한 이미지 전체를 하나의 물체처럼 움직입니다.**

    **가장 잘 되는 방법**
    - 물체만 크게 나오게 미리 잘라서 업로드
    - 가능하면 배경이 단순한 이미지 사용
    - 배경 없는 PNG면 더 자연스럽습니다
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="이미지 업로드")

            action_mode = gr.Dropdown(
                choices=[
                    "랜덤",
                    "춤추기",
                    "점프하기",
                    "회전하기",
                    "흔들기",
                    "통통 튀기",
                    "도망가기",
                    "놀라기"
                ],
                value="랜덤",
                label="행동 선택"
            )

            bg_mode = gr.Dropdown(
                choices=["white", "gray", "black"],
                value="white",
                label="배경 색상"
            )

            duration = gr.Slider(
                minimum=1,
                maximum=4,
                value=2,
                step=1,
                label="영상 길이(초)"
            )

            fps = gr.Slider(
                minimum=6,
                maximum=15,
                value=10,
                step=1,
                label="FPS"
            )

            run_button = gr.Button("움직이게 하기", variant="primary")

        with gr.Column():
            gif_output = gr.Image(label="GIF 미리보기")
            video_output = gr.Video(label="MP4 결과")
            status_output = gr.Textbox(label="상태")

    run_button.click(
        fn=generate_animation,
        inputs=[input_image, action_mode, duration, fps, bg_mode],
        outputs=[gif_output, video_output, status_output]
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print(f"서버 시작: 0.0.0.0:{port}", flush=True)

    demo.queue(default_concurrency_limit=1)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

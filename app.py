import os
import math
import random
import tempfile
from functools import lru_cache

import gradio as gr
import numpy as np
import imageio.v2 as imageio
from PIL import Image


# =============================
# 기본 설정
# =============================

MAX_IMAGE_SIZE = 900


# =============================
# 이미지 크기 제한
# =============================

def resize_if_needed(image):
    image = image.convert("RGBA")

    w, h = image.size

    if max(w, h) <= MAX_IMAGE_SIZE:
        return image

    ratio = MAX_IMAGE_SIZE / max(w, h)

    new_w = int(w * ratio)
    new_h = int(h * ratio)

    return image.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )


# =============================
# rembg 세션
# =============================

@lru_cache(maxsize=1)
def get_rembg_session():

    # 서버가 켜질 때가 아니라
    # 실제 사진을 처리할 때 rembg를 불러옵니다.
    from rembg import new_session

    # u2netp = 비교적 가벼운 모델
    return new_session("u2netp")


# =============================
# 물체 분리
# =============================

def extract_foreground(image):

    from rembg import remove

    image = resize_if_needed(image)

    session = get_rembg_session()

    removed = remove(
        image,
        session=session
    )

    removed = removed.convert("RGBA")

    arr = np.array(removed)

    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 15)

    if len(xs) == 0 or len(ys) == 0:

        w, h = image.size

        return image, image.copy(), (0, 0, w, h)

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    pad = 10

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)

    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)

    foreground = removed.crop(
        (x1, y1, x2, y2)
    )

    return image, foreground, (x1, y1, x2, y2)


# =============================
# 물체 붙이기
# =============================

def paste_center(
    canvas,
    foreground,
    center_x,
    center_y
):

    result = canvas.copy()

    fw, fh = foreground.size

    x = int(center_x - fw / 2)
    y = int(center_y - fh / 2)

    result.alpha_composite(
        foreground,
        (x, y)
    )

    return result


# =============================
# 회전 / 크기 변경
# =============================

def transform_object(
    foreground,
    angle=0,
    scale=1
):

    w, h = foreground.size

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = foreground.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    rotated = resized.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=True
    )

    return rotated


# =============================
# 행동별 움직임
# =============================

def motion(action, frame, total_frames):

    phase = 2 * math.pi * frame / total_frames

    # 춤추기
    if action == "dance":

        dx = 22 * math.sin(2 * phase)
        dy = 8 * math.sin(4 * phase)

        angle = 13 * math.sin(2 * phase)

        scale = (
            1
            + 0.035 * math.sin(4 * phase)
        )

    # 점프
    elif action == "jump":

        dx = 0

        dy = (
            -45
            * abs(math.sin(phase))
        )

        angle = (
            6
            * math.sin(2 * phase)
        )

        scale = (
            1
            + 0.04
            * abs(math.sin(phase))
        )

    # 회전
    elif action == "spin":

        dx = 0
        dy = 0

        angle = (
            360
            * frame
            / total_frames
        )

        scale = 1

    # 좌우 흔들기
    elif action == "wiggle":

        dx = (
            25
            * math.sin(6 * phase)
        )

        dy = 0

        angle = (
            18
            * math.sin(6 * phase)
        )

        scale = 1

    # 통통 튀기
    elif action == "bounce":

        dx = 0

        dy = (
            18
            * math.sin(4 * phase)
        )

        angle = 0

        scale = (
            1
            + 0.07
            * abs(math.sin(4 * phase))
        )

    # 도망가기
    elif action == "runaway":

        dx = (
            70
            * frame
            / total_frames
        )

        dy = (
            -8
            * math.sin(6 * phase)
        )

        angle = (
            10
            * math.sin(8 * phase)
        )

        scale = 1

    # 놀라기
    elif action == "surprise":

        dx = (
            8
            * math.sin(8 * phase)
        )

        dy = (
            -15
            * abs(math.sin(4 * phase))
        )

        angle = (
            7
            * math.sin(8 * phase)
        )

        scale = (
            1
            + 0.1
            * abs(math.sin(4 * phase))
        )

    else:

        dx = 0
        dy = 0
        angle = 0
        scale = 1

    return dx, dy, angle, scale


# =============================
# 영상 생성
# =============================

def generate_animation(
    image,
    action_mode,
    duration,
    fps,
    progress=gr.Progress()
):

    if image is None:

        raise gr.Error(
            "사진을 먼저 업로드해주세요."
        )

    progress(
        0.05,
        desc="사진을 준비하고 있습니다..."
    )

    # 행동 선택
    actions = [
        "dance",
        "jump",
        "spin",
        "wiggle",
        "bounce",
        "runaway",
        "surprise"
    ]

    mapping = {

        "춤추기": "dance",
        "점프하기": "jump",
        "회전하기": "spin",
        "흔들기": "wiggle",
        "통통 튀기": "bounce",
        "도망가기": "runaway",
        "놀라기": "surprise"
    }

    if action_mode == "랜덤":

        action = random.choice(actions)

    else:

        action = mapping.get(
            action_mode,
            "dance"
        )

    progress(
        0.12,
        desc="사진 속 물체를 찾고 있습니다..."
    )

    background, foreground, bbox = \
        extract_foreground(image)

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    total_frames = int(
        float(duration)
        * int(fps)
    )

    total_frames = max(
        8,
        min(total_frames, 60)
    )

    frames = []

    # =============================
    # 각 프레임 만들기
    # =============================

    for frame_number in range(
        total_frames
    ):

        dx, dy, angle, scale = motion(
            action,
            frame_number,
            total_frames
        )

        transformed = transform_object(
            foreground,
            angle,
            scale
        )

        frame = paste_center(
            background,
            transformed,
            center_x + dx,
            center_y + dy
        )

        frame_rgb = np.array(
            frame.convert("RGB")
        )

        frames.append(frame_rgb)

        progress(
            0.2
            + 0.5
            * (
                frame_number
                / total_frames
            ),
            desc="애니메이션을 만들고 있습니다..."
        )

    temp_dir = tempfile.mkdtemp()

    gif_path = os.path.join(
        temp_dir,
        "result.gif"
    )

    mp4_path = os.path.join(
        temp_dir,
        "result.mp4"
    )

    # =============================
    # GIF
    # =============================

    progress(
        0.75,
        desc="GIF를 만들고 있습니다..."
    )

    imageio.mimsave(
        gif_path,
        frames,
        duration=1 / int(fps),
        loop=0
    )

    # =============================
    # MP4
    # =============================

    progress(
        0.85,
        desc="MP4 영상을 만들고 있습니다..."
    )

    writer = imageio.get_writer(
        mp4_path,
        fps=int(fps),
        codec="libx264",
        quality=6,
        macro_block_size=2
    )

    for frame in frames:

        writer.append_data(frame)

    writer.close()

    action_names = {

        "dance": "춤추기",
        "jump": "점프하기",
        "spin": "회전하기",
        "wiggle": "흔들기",
        "bounce": "통통 튀기",
        "runaway": "도망가기",
        "surprise": "놀라기"
    }

    selected_action = action_names[action]

    progress(
        1,
        desc="완료!"
    )

    status = (
        f"완성되었습니다! "
        f"선택된 행동: {selected_action}"
    )

    return (
        gif_path,
        mp4_path,
        status
    )


# =============================
# 홈페이지 UI
# =============================

with gr.Blocks(
    title="사진 속 물체 춤추기 AI"
) as demo:

    gr.Markdown(
        """
        # 🕺 사진 속 물체 춤추기 AI

        사진 한 장을 올리면  
        사진 속 주 물체가 춤추거나 랜덤한 행동을 합니다.
        """
    )

    with gr.Row():

        # 왼쪽
        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="📷 사진 업로드"
            )

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

                label="🎲 행동"
            )

            duration = gr.Slider(
                minimum=1,
                maximum=4,
                value=2,
                step=1,
                label="⏱ 영상 길이 (초)"
            )

            fps = gr.Slider(
                minimum=6,
                maximum=15,
                value=10,
                step=1,
                label="🎞 FPS"
            )

            make_button = gr.Button(
                "✨ 움직이게 하기",
                variant="primary"
            )

        # 오른쪽
        with gr.Column():

            gif_output = gr.Image(
                label="GIF 미리보기"
            )

            video_output = gr.Video(
                label="MP4 영상"
            )

            status_output = gr.Textbox(
                label="상태"
            )

    make_button.click(

        fn=generate_animation,

        inputs=[
            input_image,
            action_mode,
            duration,
            fps
        ],

        outputs=[
            gif_output,
            video_output,
            status_output
        ]
    )


# =============================
# Render 서버 실행
# =============================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    print(
        f"서버 시작: 0.0.0.0:{port}",
        flush=True
    )

    demo.queue(
        default_concurrency_limit=1
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

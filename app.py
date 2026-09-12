import os
import math
import random
import tempfile
import numpy as np
import cv2
import imageio.v2 as imageio
from PIL import Image
import gradio as gr
from rembg import remove


# -----------------------------
# 유틸 함수
# -----------------------------
def pil_to_rgba(img):
    return img.convert("RGBA")


def extract_foreground(image: Image.Image):
    """
    이미지에서 전경(주 물체)을 배경 제거 방식으로 추출합니다.
    반환:
      original_rgba: 원본 RGBA
      fg_crop: 전경만 crop한 RGBA
      bbox: (x1, y1, x2, y2)
    """
    original_rgba = pil_to_rgba(image)
    removed = remove(original_rgba)  # 배경 제거
    removed = pil_to_rgba(removed)

    arr = np.array(removed)
    alpha = arr[:, :, 3]

    ys, xs = np.where(alpha > 10)
    if len(xs) == 0 or len(ys) == 0:
        # 물체 검출 실패 시 전체 이미지를 전경으로 취급
        w, h = original_rgba.size
        return original_rgba, original_rgba.copy(), (0, 0, w, h)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # 약간 여백 추가
    pad = 10
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(original_rgba.size[0], x2 + pad)
    y2 = min(original_rgba.size[1], y2 + pad)

    fg_crop = removed.crop((x1, y1, x2, y2))
    return original_rgba, fg_crop, (x1, y1, x2, y2)


def paste_center(canvas, fg_img, center_x, center_y):
    """
    RGBA canvas 위에 fg_img를 중심 좌표 기준으로 붙입니다.
    """
    out = canvas.copy()
    fw, fh = fg_img.size
    x = int(center_x - fw / 2)
    y = int(center_y - fh / 2)
    out.alpha_composite(fg_img, (x, y))
    return out


def transform_fg(fg_img, angle=0, scale=1.0):
    """
    전경 이미지를 회전/스케일 변환합니다.
    """
    w, h = fg_img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = fg_img.resize((new_w, new_h), Image.LANCZOS)
    rotated = resized.rotate(angle, resample=Image.BICUBIC, expand=True)
    return rotated


# -----------------------------
# 행동 애니메이션 정의
# -----------------------------
def make_motion_params(action, t, total_frames):
    """
    프레임별 동작 파라미터 생성
    반환: dx, dy, angle, scale
    """
    phase = 2 * math.pi * t / total_frames

    if action == "dance":
        dx = 20 * math.sin(2 * phase)
        dy = 10 * math.sin(4 * phase)
        angle = 12 * math.sin(2 * phase)
        scale = 1.0 + 0.03 * math.sin(4 * phase)

    elif action == "jump":
        dx = 0
        # 점프 곡선 느낌
        dy = -35 * abs(math.sin(phase))
        angle = 5 * math.sin(2 * phase)
        scale = 1.0 + 0.04 * abs(math.sin(phase))

    elif action == "spin":
        dx = 0
        dy = 0
        angle = 360 * (t / total_frames)
        scale = 1.0

    elif action == "wiggle":
        dx = 25 * math.sin(6 * phase)
        dy = 0
        angle = 18 * math.sin(6 * phase)
        scale = 1.0

    elif action == "bounce":
        dx = 0
        dy = 15 * math.sin(4 * phase)
        angle = 0
        scale = 1.0 + 0.07 * abs(math.sin(4 * phase))

    elif action == "runaway":
        dx = 50 * (t / total_frames)
        dy = -8 * math.sin(4 * phase)
        angle = 10 * math.sin(8 * phase)
        scale = 1.0

    else:
        dx = 0
        dy = 0
        angle = 0
        scale = 1.0

    return dx, dy, angle, scale


def generate_animation(image, action_mode, duration_sec, fps):
    """
    업로드 이미지 -> 애니메이션 GIF, MP4 생성
    """
    if image is None:
        return None, None, "이미지를 먼저 업로드해주세요."

    if action_mode == "랜덤":
        action = random.choice(["dance", "jump", "spin", "wiggle", "bounce", "runaway"])
    else:
        mapping = {
            "춤추기": "dance",
            "점프하기": "jump",
            "회전하기": "spin",
            "흔들기": "wiggle",
            "통통 튀기": "bounce",
            "도망가기": "runaway"
        }
        action = mapping.get(action_mode, "dance")

    original_rgba, fg_crop, bbox = extract_foreground(image)
    bg = original_rgba.copy()

    # 원본 bbox 위치 중심
    x1, y1, x2, y2 = bbox
    base_cx = (x1 + x2) / 2
    base_cy = (y1 + y2) / 2

    total_frames = max(8, int(duration_sec * fps))
    frames_rgba = []

    for t in range(total_frames):
        dx, dy, angle, scale = make_motion_params(action, t, total_frames)
        fg_trans = transform_fg(fg_crop, angle=angle, scale=scale)

        frame = bg.copy()
        frame = paste_center(frame, fg_trans, base_cx + dx, base_cy + dy)
        frames_rgba.append(frame)

    # 임시 파일 저장
    temp_dir = tempfile.mkdtemp()
    gif_path = os.path.join(temp_dir, "result.gif")
    mp4_path = os.path.join(temp_dir, "result.mp4")

    # GIF 저장
    gif_frames = [np.array(f.convert("RGBA")) for f in frames_rgba]
    imageio.mimsave(gif_path, gif_frames, duration=1/fps, loop=0)

    # MP4 저장
    h, w = np.array(frames_rgba[0].convert("RGB")).shape[:2]
    writer = cv2.VideoWriter(
        mp4_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )
    for frame in frames_rgba:
        rgb = np.array(frame.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()

    action_kor = {
        "dance": "춤추기",
        "jump": "점프하기",
        "spin": "회전하기",
        "wiggle": "흔들기",
        "bounce": "통통 튀기",
        "runaway": "도망가기"
    }.get(action, action)

    msg = f"완료되었습니다. 선택된 행동: {action_kor}"
    return gif_path, mp4_path, msg


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks(title="사진 속 물체 춤추기 AI") as demo:
    gr.Markdown("# 사진 속 물체 춤추기 AI")
    gr.Markdown(
        "사진을 업로드하면 사진 속 주 물체를 분리한 뒤, 춤추거나 랜덤한 행동을 하는 애니메이션을 생성합니다."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="사진 업로드")
            action_mode = gr.Dropdown(
                choices=["랜덤", "춤추기", "점프하기", "회전하기", "흔들기", "통통 튀기", "도망가기"],
                value="랜덤",
                label="행동 선택"
            )
            duration_sec = gr.Slider(1, 5, value=2, step=1, label="길이(초)")
            fps = gr.Slider(6, 20, value=12, step=1, label="FPS")
            run_btn = gr.Button("애니메이션 만들기")

        with gr.Column():
            output_gif = gr.Image(label="GIF 미리보기")
            output_video = gr.Video(label="MP4 결과")
            output_text = gr.Textbox(label="상태")

    run_btn.click(
        fn=generate_animation,
        inputs=[input_image, action_mode, duration_sec, fps],
        outputs=[output_gif, output_video, output_text]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

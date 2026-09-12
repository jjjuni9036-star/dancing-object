import os
import gradio as gr

def hello(name):
    return f"안녕하세요, {name}!"

demo = gr.Interface(
    fn=hello,
    inputs="text",
    outputs="text",
    title="Render 테스트"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"서버 시작: 0.0.0.0:{port}", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )

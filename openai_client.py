import base64
import io
import os

from openai import OpenAI
from PIL import Image

OPENAI_API_KEY = "<APIKEY>"
gpt_model = "gpt-5.2-2025-12-11"

def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or OPENAI_API_KEY.strip()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key missing. Set OPENAI_API_KEY environment variable or OPENAI_API_KEY in openai_client.py."
        )
    return OpenAI(api_key=api_key)


def message_formator(prompt, texts_imgs):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(item)}"},
                }
                if isinstance(item, Image.Image)
                else {"type": "text", "text": str(item)}
                for item in texts_imgs
            ],
        },
    ]


def gpt(prompt, texts_imgs=None, temperature=0.0, seed=0, n=1):
    del seed, n
    if texts_imgs is None:
        texts_imgs = []

    client = get_client()
    messages = message_formator(prompt, texts_imgs)
    # import json
    # with open("test.txt", "w") as f:
    #     json.dump(messages, f, indent=4)
    # exit()

    response = client.chat.completions.create(
        model=gpt_model,
        messages=messages,
        temperature=temperature,
    )
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()


def self_test() -> int:
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            temperature=0.0,
        )
        text = (response.choices[0].message.content or "").strip()
        print("OpenAI API test passed.")
        print(f"Response: {text}")
        return 0
    except Exception as exc:
        print("OpenAI API test failed.")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(self_test())

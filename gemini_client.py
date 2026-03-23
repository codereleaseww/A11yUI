import os

from google import genai


GEMINI_API_KEY = "<APIKEY>"


def gemini(prompt, texts_imgs=None, temperature=0, seed=0, n=1):
    del seed, n
    if texts_imgs is None:
        texts_imgs = []

    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY.strip()
    if not api_key:
        raise RuntimeError(
            "Gemini API key missing. Set GEMINI_API_KEY environment variable or GEMINI_API_KEY in gemini_client.py."
        )

    generation_config = {"temperature": temperature}
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[prompt] + texts_imgs,
        config=generation_config,
    )
    return response.text


def self_test() -> int:
    """Run a minimal API connectivity test."""
    try:
        text = gemini(
            "You are a test assistant.",
            texts_imgs=["Reply with exactly: OK"],
            temperature=0.0,
        )
        print("Gemini API test passed.")
        print(f"Response: {(text or '').strip()}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print("Gemini API test failed.")
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(self_test())

from typing import Optional
from transformers import pipeline
from core.mixins import LoggingMixin, ValidationMixin
from core.decorators import timed, requires_input


class BaseAdapter(LoggingMixin, ValidationMixin):
    def __init__(self) -> None:
        self.pipe = None  # lazy init

    def _ensure_loaded(self):
        if self.pipe is None:
            raise RuntimeError(
                "Pipeline not initialized. Call .load() first or use a subclass that guards it."
            )


class GPT2TextAdapter(BaseAdapter):
    """Text generation using openai-community/gpt2 on Hugging Face."""
    def __init__(self, model_name: str = "openai-community/gpt2", device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.device = device

    @timed
    def load(self):
        self.log(f"Loading text-generation pipeline: {self.model_name}")
        self.pipe = pipeline("text-generation", model=self.model_name, device=self.device)
        return self

    @timed
    @requires_input
    def run(self, prompt: str, max_new_tokens: int = 60, do_sample: bool = True, temperature: float = 0.8):
        if self.pipe is None:
            self.load()
        return self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )


class ViTGPT2CaptionAdapter(BaseAdapter):
    """Image captioning using nlpconnect/vit-gpt2-image-captioning."""
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning", device: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.device = device

    @timed
    def load(self):
        self.log(f"Loading image-to-text pipeline: {self.model_name}")
        self.pipe = pipeline("image-to-text", model=self.model_name, device=self.device)
        return self

    @timed
    @requires_input
    def run(self, image_path: str, max_new_tokens: int = 30):
        self.ensure_file_exists(image_path)
        if self.pipe is None:
            self.load()
        return self.pipe(image_path, max_new_tokens=max_new_tokens)

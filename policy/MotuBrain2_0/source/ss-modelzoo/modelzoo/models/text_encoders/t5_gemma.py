import logging

import torch
from transformers import AutoTokenizer
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaModel

from .text_encoder import AbstractEncoder

logger = logging.getLogger(__name__)


class T5GemmaEmbedder(AbstractEncoder):
    """
    T5Gemma encoder from Hugging Face.

    Args:
        pretrained_model_name_or_path (str): model name from Hugging Face
        device (str, optional, default "cuda"): device to use
        max_length (int, optional, default 4096): max length of the input text
        use_compile (bool, optional, default False): whether to use compile
        use_flash_attn (bool, optional, default False): whether to use flash attention

    NOTE:
        if use_flash_attn, you need to exec the following command to install flash_attn_3:
            python_path=`python -c "import site; print(site.getsitepackages()[0])"`
            mkdir -p $python_path/flash_attn_3
            cp $python_path/flash_attn_interface.py  $python_path/flash_attn_3/
            pip list | grep flash_attn_3
    """

    def __init__(
        self,
        pretrained_model_name_or_path,
        device="cuda",
        max_length=4096,
        use_compile=False,
        use_flash_attn=False,
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.transformer: T5GemmaModel = T5GemmaModel.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_3" if use_flash_attn else "eager",
        )
        del self.transformer.decoder
        self.freeze()
        self.transformer.to(device)
        if use_compile:
            self.transformer = torch.compile(self.transformer)  # type: ignore

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, padding: bool = False):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.transformer.encoder(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)

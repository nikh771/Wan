from typing import Any

class LoraConfig:
    def __init__(
        self,
        r: int = ...,
        lora_alpha: int = ...,
        init_lora_weights: str = ...,
        target_modules: list[str] | None = ...,
        **kwargs: Any,
    ) -> None: ...

def set_peft_model_state_dict(model: Any, peft_model_state_dict: Any, **kwargs: Any) -> None: ...

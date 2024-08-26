from transformers import AutoModelForCausalLM, PreTrainedModel


def load_transformer(model_name: str) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

import torch
from peft import PeftModel
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}

### Response:"""


BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = "tloen/alpaca-lora-7b"    


def evaluate(
    tokenizer,
    model,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2048,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()    


def main():
    # NOTE: There will be a warning about mismatched tokenizer names, which
    # seems to be caused by LlamaTokenizer used to be called LLaMATokenizer. I
    # think it cane be safely ignored.
    tokenizer= LlamaTokenizer.from_pretrained(BASE_MODEL)
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        # TODO(breakds): Add bitsandbytes and turn 8bit on.
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.float16)
    model.eval()

    print(evaluate(
        tokenizer, model,
        instruction="Tell me about framework laptop"))


if __name__ == "__main__":
    main()

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from llava import LlavaLlamaForCausalLM


def apply_delta(base_model_path, target_model_path, delta_path):

    print("Loading base model")
    base = LlavaLlamaForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16, 
        cache_dir='/mnt/keremaydin/llava/model',
        low_cpu_mem_usage=True)

    print("Loading delta")
    delta = LlavaLlamaForCausalLM.from_pretrained(delta_path, 
                                                  torch_dtype=torch.float16, 
                                                  cache_dir='/mnt/keremaydin/llava-med/model',
                                                  low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

    print("Applying delta")
    for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
        if name not in base.state_dict():
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            param.data += base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
                f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving target model")
    delta.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == '__main__':

    apply_delta(base_model_path='liuhaotian/llava-v1.5-7b',
                target_model_path='/mnt/keremaydin/llava-med/model',
                delta_path='microsoft/llava-med-7b-delta')
import pandas as pd
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms
from preprocessing import *
from llava.utils import disable_torch_init
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from peft import (
    PeftModel,
)
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates

device = 'cpu'

disable_torch_init()

base_path = "/mnt/bulentsiyah/multimodal-fine-tuning/llava/fine_tuning_llava_beta_clock/best_so_far"

# Parser extracts the parameters for arguments from run_script.sh
parser = transformers.HfArgumentParser(
(ModelArguments, DataArguments, TrainingArguments))
model_args, _, training_args = parser.parse_args_into_dataclasses(args_filename='/mnt/bulentsiyah/multimodal-fine-tuning/llava/sh_files/run_script_1.sh')

model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=device
        )
model.config.use_cache = True

tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        device_map=device
    )

vision_tower = model.get_vision_tower()
vision_tower.load_model()
vision_tower.to(device=model.device)
image_processor = vision_tower.image_processor


trained_model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map=device
        )
trained_model.config.use_cache = True

trained_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        device_map=device
    )

vision_tower = trained_model.get_vision_tower()
vision_tower.load_model()
vision_tower.to(device=trained_model.device)
image_processor = vision_tower.image_processor

        
trained_model=PeftModel.from_pretrained(trained_model, base_path)  

trained_model.to(training_args.device, dtype=torch.float16)  



# Define a transformation to convert PIL image to PyTorch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])

import json

with open('/mnt/bulentsiyah/multimodal-fine-tuning/data/jsons/synthetic/df_clock.json', 'r') as f:
    train_data = json.load(f)

bf_rows = []
af_rows = []

root_path = '/mnt/bulentsiyah/multimodal-fine-tuning/data/datasets/syn_dataset/train_data'
counter = 0

for time in os.listdir(root_path):

    for img in os.listdir(os.path.join(root_path, time)):

        print(f'{counter+1}/{len(train_data)}')

        image_tensor = Image.open(os.path.join(root_path, time, img)).convert('RGB')

        image_tensor = image_processor.preprocess(image_tensor,  return_tensors='pt')['pixel_values'].to(dtype=model.dtype, device=model.device)

        image_features = model.get_model().get_vision_tower()(image_tensor)
        trained_image_features = trained_model.get_model().get_vision_tower()(image_tensor)
        
        old_proj = model.get_model().mm_projector(image_features)

        new_proj = trained_model.get_model().mm_projector(trained_image_features)


        bf_rows.append({
            'prediction_id':img,
            'prediction_ts':counter,
            'image_vector':np.array2string(np.array(torch.mean(before_fine_tune, axis=1).detach().cpu().numpy().squeeze()), threshold=np.inf),
            'image_path':os.path.join(time, img),
            'actual_time':time,
            'predicted_time':time
        })

        af_rows.append({
            'prediction_id':img,
            'prediction_ts':counter,
            'image_vector':np.array2string(np.array(torch.mean(after_fine_tune, axis=1).detach().cpu().numpy().squeeze()), threshold=np.inf),
            'image_path':os.path.join(time, img),
            'actual_time':time,
            'predicted_time':time
        })

        counter += 1

df_bf = pd.DataFrame(bf_rows)
df_af = pd.DataFrame(af_rows)


df_bf.to_csv('before_embeddings.csv', index=False)
df_af.to_csv('after_embeddings.csv', index=False)
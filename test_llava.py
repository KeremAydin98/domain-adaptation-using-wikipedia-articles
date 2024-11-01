import torch
import transformers
from peft import (
    PeftModel,
)
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    tokenizer_image_token,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from PIL import Image
from io import BytesIO
from rouge_score import rouge_scorer
import requests
from llava.utils import disable_torch_init
from preprocessing import *
from termcolor import colored
import re

import warnings
warnings.filterwarnings("ignore")

class ModelLLaVa:

    def __init__(self):
        
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.conv_img = None
        self.img_tensor = None
        self.roles = None
        self.stop_key = None
        self.load_models()

    def load_models(self):
        
        '''
        Load the model, process and tokenizer
        '''

        disable_torch_init()

        self.base_path = "/mnt/keremaydin/llava/results/try/8_2e-4/checkpoint-8700"
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, _, training_args = parser.parse_args_into_dataclasses(args_filename='/mnt/keremaydin/llava/run_script_1.sh')

        self.model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    device_map='auto'
                )
        self.model.config.use_cache = True

        self.model=PeftModel.from_pretrained(self.model, self.base_path)  

        self.model.to(self.model.device, dtype=torch.float16)  
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                device_map='auto'
            )

        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model()
        vision_tower.to(device=self.model.device)
        self.image_processor = vision_tower.image_processor

    def setup_image(self, img_path):
        """
        Load and process the image.
        """

        if img_path.startswith('http') or img_path.startswith('https'):
            response = requests.get(img_path)
            conv_img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            conv_img = Image.open(img_path).convert('RGB')
        
        img_tensor = self.image_processor.preprocess(conv_img,  return_tensors='pt')['pixel_values']

        return img_tensor.to(dtype=self.model.dtype, device=self.model.device)

    def generate_answer(self, prompt, img_path):

        """
        Generates answer from fine-tuned model
        """
        
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        image_tensor = self.setup_image(img_path)
        image_sizes = [x.size for x in image_tensor]

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)

        with torch.inference_mode():
            output_ids=self.model.generate(
                                inputs=input_ids,
                                images=image_tensor,
                                image_sizes=image_sizes,
                                max_new_tokens=12,
                                do_sample=True,
                                temperature=0.01,
                                use_cache=True,
                                )
            
        return self.tokenizer.decode(output_ids[0, input_ids.shape[0]:], 
                                     skip_special_tokens=True).strip()
   

if __name__ == '__main__':

    torch.cuda.empty_cache()

    print("----test basladı----")

    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

    with open('/mnt/keremaydin/llava/data/df_test_Turkey.json', 'r') as f:
        test_data = json.load(f)

    image_folder = '/mnt/keremaydin/llava/data'

    model_llava = ModelLLaVa()

    red = 0
    yellow = 0
    green = 0

    if len(test_data) <= 100:
        length = len(test_data)
    else:
        length = 100

    for i in range(length):

        print(f'{round((i+1) / (length) * 100, 3)}%')

        prediction = model_llava.generate_answer(prompt= test_data[i]['conversations'][0]['value'].split('<image>\n')[-1], 
                                            img_path=os.path.join(image_folder, test_data[i]['image'])).split('###')[0]

        rouge_score = scorer.score(test_data[i]['conversations'][1]['value'], prediction)['rouge1'].recall

        if rouge_score >= 0.75: 
            green += 1
        elif rouge_score >= 0.5: 
            yellow += 1
        else:
            red += 1


    print(model_llava.base_path)
    print(f'Red: {red}, Yellow: {yellow}, Green: {green}')    
    print("----test bitti----")




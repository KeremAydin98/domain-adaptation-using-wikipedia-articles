import torch
import transformers
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from PIL import Image
from io import BytesIO
import requests
from llava.utils import disable_torch_init
from preprocessing import *
import pandas as pd

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
                
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, _, training_args = parser.parse_args_into_dataclasses(args_filename='/mnt/keremaydin/llava/run_script.sh')

        self.model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    device_map='auto'
                )
        self.model.config.use_cache = True
        
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
        

        with open('/mnt/keremaydin/llava/example.json', 'r') as f:
            text = json.load(f)[0]

        all_text = ''
        for line in text['lines']:
            if line['text'].isdigit():
                continue
            all_text = all_text + line['text'] + '\n'

        all_text = '<text_inside_img>\n' + all_text + '\n</text_inside_img>'

        image_sizes = [x.size for x in image_tensor]

        # just one turn, always prepend image token
        inp = DEFAULT_IMAGE_TOKEN + '\n' + all_text + '\n' + prompt

        print(inp)
        conv.append_message(conv.roles[0], inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=self.model.device)

        with torch.inference_mode():
            output_ids=self.model.generate(
                                inputs=input_ids,
                                images=image_tensor,
                                image_sizes=image_sizes,
                                do_sample=True,
                                temperature=0.01,
                                use_cache=True,
                                )
            
        return self.tokenizer.decode(output_ids[0, input_ids.shape[0]:], 
                                     skip_special_tokens=True).strip()


if __name__ == '__main__':

    torch.cuda.empty_cache()

    print("----Test started----")

    image_folder = '/mnt/keremaydin/data/screen_dataset'

    model_llava = ModelLLaVa()

    results = []

    for filename in os.listdir(image_folder):

        file_path = os.path.join(image_folder, 'screen1.png')

        prediction = model_llava.generate_answer(prompt= 'Explain what is on the screen in detail by using both the image and text inside the image.', 
                                                img_path=file_path)

        print(filename)
        print(prediction)
        print('========================================')


    
    print("----Test Ended----")




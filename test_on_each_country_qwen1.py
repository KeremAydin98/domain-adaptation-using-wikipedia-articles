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
import json
import pandas as pd

df = pd.read_csv('/mnt/keremaydin/llava/data/full_dataframe.csv')

class_labels = ', '.join(list(df['title'].unique()))

with open('/mnt/keremaydin/llava/data/geography.json', "r") as json_file:
    geo_info = json.load(json_file)

class ModelLLaVa:

    def __init__(self, base_path, run_script_path):
        
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.conv = None
        self.conv_img = None
        self.img_tensor = None
        self.roles = None
        self.stop_key = None
        self.base_path = base_path
        self.run_script_path = run_script_path
        self.load_models()

    def load_models(self):
        
        '''
        Load the model, process and tokenizer
        '''

        disable_torch_init()
        
        # Parser extracts the parameters for arguments from run_script.sh
        parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
        model_args, _, training_args = parser.parse_args_into_dataclasses(args_filename=self.run_script_path)

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

    def generate_answer(self, prompt, img_path, max_new_tokens):

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
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=0.01,
                                use_cache=True,
                                )
            
        return self.tokenizer.decode(output_ids[0, input_ids.shape[0]:], 
                                     skip_special_tokens=True).strip()

def create_test_data(test_df, class_labels):

    combined_data = []
    if len(test_df) > 1000:
        length = 1000
    else:
        length = len(test_df)
        
    for i in range(length):

        question = f'Classify the image with a single label from \
                    this set: {class_labels}. You must provide a \
                    single label.'


        combined_data.append({
            "image": test_df.iloc[i]['filepath'],
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": test_df.iloc[i]['title']
                }
            ]
        })

            
    df = pd.DataFrame(combined_data)
    df= df.sample(frac=1).reset_index(drop=True)
    df_final = df.to_dict(orient='records')

    return df_final

scores = []
geo_scores = []

model_llava = ModelLLaVa(base_path='/mnt/keremaydin/llava/results/try/8_2e-4/checkpoint-8700', run_script_path='/mnt/keremaydin/llava/run_script_1.sh')

list_of_countries = list(df['country'].unique())
image_folder = '/mnt/keremaydin/llava/data'

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

for index, country in enumerate(list_of_countries):

    torch.cuda.empty_cache()

    test_df = df[df['country'] == country].reset_index(drop=True)

    test_data = create_test_data(test_df, class_labels)

    direct_classification = {country:{'red':0, 'yellow':0, 'green':0}}

    print(colored(country, 'green'))

    for i in range(len(test_data)):

        print(f'{index+1}/{len(list_of_countries)}:{round((i+1) / (len(test_data)) * 100, 3)}%')

        prediction = model_llava.generate_answer(prompt= test_data[i]['conversations'][0]['value'].split('<image>\n')[-1], 
                                            img_path=os.path.join(image_folder, test_data[i]['image']),
                                            max_new_tokens=12).split('###')[0]

        rouge_score = scorer.score(test_data[i]['conversations'][1]['value'], prediction)['rouge1'].recall

        if rouge_score >= 0.75: 
            direct_classification[country]['green'] += 1
        elif rouge_score >= 0.5: 
            direct_classification[country]['yellow'] += 1
        else:
            direct_classification[country]['red'] += 1
        

    scores.append(direct_classification)

df_scores = pd.DataFrame(scores)
    
df_scores.to_csv('scores.csv', index=False)





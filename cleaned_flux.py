import os 
import time
import random 
import traceback
from PIL import Image
import torch
import cv2
from briarmbg import BriaRMBG
from utilities import preprocess_image, postprocess_image
from rembg import remove, new_session
import uuid
import json
from numpyencoder import NumpyEncoder
import numpy as np
import re 
import argparse
from otf_combined import *
from helper_function import *
from config import GEMINI_KEY
from setup_model import load_model,load_llm    

parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--country', required=True, help='Country name')
parser.add_argument('--json1', required=True, help='Path to the first JSON file keywords and prompts')
parser.add_argument('--output', required=True, help='Path to the output images')
parser.add_argument('--json2', required=True, help='Path to the second JSON file categories and outfits')
parser.add_argument('--sticker_type', required=True, help='type of sticker')
parser.add_argument('--font', required=True, help='default font')
parser.add_argument('--api', action='store_true', help='API for language model or llama') #not used in this script but can be used later
total_number_of_images =3 #number of images per gender for a keyword

args = parser.parse_args()
json_path=args.json1
output_dir=args.output
# Now you can use the arguments as variables
country=args.country
json_file_path=args.json2
sticker_type=args.sticker_type
font_default=args.font
is_api=args.api

os.makedirs(output_dir,exist_ok=True)
assets_path = '/genAIcontentgeneration-assets'

if not is_api:
    tic = time.time()
    pipeline=load_model()
    toc = time.time()
    print(f"Model loading time: {toc - tic:.2f} seconds")
    print('############################# LLM pipeline loaded ######################')

    
def query_gemini_api(input_text, api_key='GEMINI_KEY'):
    tic1 = time.time()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={api_key}"
    # gemini-2.5-flash-preview-05-20
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{
            "parts": [{"text": input_text}]
        }]
    }

    response = requests.post(url, headers=headers, json=payload)
    toc1 = time.time()
    print(f"API call time: {toc1 - tic1:.2f} seconds")
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]['parts'][0]['text']
    else:
        print("API Error:", response.status_code, response.text)
        return None
    
lora_models = {'expression_lora':[os.path.join(assets_path,"/non_personalised_lora/pytorch_lora_weights.safetensors"),"exaggerated expressions"] }
pipe=load_model()
print('############################# Flux pipeline loaded ######################')
                
net = BriaRMBG()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
net = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
net.to(device)

gender=['middle aged woman','little girl','middle aged man','little boy','teenage woman','teenage man']
skin_tons = ['fair']
color='colourful vibrant shiny' #color of the outfit, this is used in prompt
body_styles=['full body','potrait size','small short body']

############################# The above section defined constant parameters for age groups and body styles######################

# Load the JSON data from the file
with open(json_file_path, 'r') as file:   #category json is being loaded here. This has outfits,trasnport etc.
    data = json.load(file)

for item in data:
    if 'Traditional Outfits for women' in item:
        outfits_female=[f'{country} outfit']  

    if 'Traditional Outfits for men' in item:
        outfits_male=[f'{country} outfit']  #just using country outfit was working better


with open(json_path, "r") as json_file:   #keywords json is being loaded here. This has keywords and prompts for each keyword.
    data = json.load(json_file)

category_listy=[category_name for category in data for category_name in category.keys()]
        
for i in range(len(data)):
    category_name=category_listy[i]

    slangs_dict = data[i].get(f"{category_name}")

    # Iterating over the key-value pairs in the dictionary
    if slangs_dict:
        prompt_csv=[]
        for keyword, keyword_prompt in slangs_dict.items():            
            for pr in keyword_prompt[1:]:
                prompt_csv.append(pr)  #to be replaced by [1:] for prompts
            english=keyword_prompt[0]   
            letter=keyword+'('+english+')'
   
            for select_gender in gender:
                i = 0
                while i != total_number_of_images:
                    ############ Sample random prompt ###################
                    prompt_random=random.choice(prompt_csv)

                    #outfit and lora selection based on gender
                
                    print(select_gender)
                    if select_gender in ['teenage man','little boy','middle aged man']:   
                        select_outfit=random.choice(outfits_male)
                    else:  #else if female                      
                        select_outfit=random.choice(outfits_female)
                    
                    select_skin_tone = random.choice(skin_tons)
                    body_style=random.choice(body_styles)
                
                    #prompts are different for non personalized and personalized
                    if sticker_type=='non personalized':
                        prompt_base1 = f"""Write a prompt in strictly 40 words only for single and only single  {country} character , 3d pixar cartoon image, from the given keywords create a simple character with plain white background which is very much related to given keywords:person depicting the expression {letter}.Describe exact pose.Do not mention the word "exaggerated" anywhere in prompt. Understand the cultural context of {letter}. \
                        keywords:- cartoon , random {select_gender} Single Character wearing {color} {select_outfit}(in 2-3 words only strictly),{letter}, {select_skin_tone} type skin tone.You are supposed to describe the clothes {select_outfit} strictly in 2-3 words,exaggerated or funny expressions,different poses,and props for {letter} only.we need actionable prompts depicting what the charater must look like along with actions and expressions,postures,poses. make it exaggerated with props,expressions and actions to depict characters.You made add props for greetings or wherever necessary. The props must be strictly relevant to {letter} and not random stuff. Do not add if you don't know relevant props.Focus on expressions. The sticker must tell a story.Dont add flags strictly. Flags are prohibitted. Do not give hard, abstract feelings to characters.Dont write long descriptions.
                        The output should be a markdown code snippet formatted in the following schema, including the leading and trailing 'json' and "": """

                        prompt_base2 = '```json \
                                {"prompt": "output" // image generation prompt}  \
                                The output must be 40 words. The output must has only "prompt" in the above format. The prompt focus must be on the hands action and facial expression and must be exaggerated about actions and props.Do not write long descriptions.Dont add too many rainbow type thing.\
                                you must add atleast 1 character to each prompt.\
                                Do not add flags strictly prohibitted.\
                                ```'
                    else:
                        body_styles=['full body','potrait size','small short body']
                        body_style=random.choice(body_styles)
                        prompt_base1 = f"Write a prompt in strictly 40 words only for single and only single {body_style}character (strictly bald) bald head and very long neck, cartoon image, from the given keywords create a simple character with plain simple background which is very much related to given keywords.  \
                                    keywords:- cartoon , random {select_gender} Single Character wearing bright colourful{select_outfit}, {letter}, fair type skin tone.Do not mention {letter} in prompt.Explain in simple words.Do not give long background descriptions.You may add props in hand but background should be white.The props must be in hand only and should be related to {letter} but they are not necessary.we need actionable prompts depicting what the charater must look like along with actions and props if any,postures,poses,hand gestures related to expressions shown for saying {letter}. make it exaggerated with props,expressions and actions to depict characters.\ Do not give hard, abstract feelings to characters.Dont write long descriptions.Do not mention vague text like {letter} expression,explain expression from your understanding in simple words.You must add actions and hand gestures.Strictly do not add hats/caps/abaya/scarf on head.\
                                    The output should be a markdown code snippet formatted in the following schema, including the leading and trailing 'json' and "": "

                        #this prompt remains same for both non personalized and personalized
                        prompt_base2 = '```json \
                                {"prompt": "output" // image generation prompt}  \
                                The output must be 40 words. The output must has only "prompt" in the above format. The prompt focus must be on the hands action and facial expression and must be exaggerated about actions and props.Do not write long descriptions.\
                                you must add atleast 1 character to each prompt.Strictly do not add hats/caps/abaya/scarf on head.\
                                Do not add flags strictly prohibitted.\
                                ```'
                    prompt_mistral = prompt_base1 + prompt_base2
                    # Please keep in mind that the prompt is written for whats app sticker so make it accordingly without introducing sticker keyword into prompt \ # ]
                    messages = [{"role": "user", "content": prompt_mistral},]
                    if not is_api:
                        prompt= pipeline.tokenizer.apply_chat_template(messages,add_generation_prompt=True, tokenize=False) 
                        terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]   
                        outputs = pipeline(prompt, max_new_tokens = 256, eos_token_id = terminators,do_sample = True,)
                    
                        output=outputs[0]["generated_text"][len(prompt):]  
                    else:
                        # tic2 = time.time()
                        output = query_gemini_api(prompt_mistral,api_key=GEMINI_KEY)   
                        # toc2 = time.time()
                        # print(f"API Image gen Output time: {toc2 - tic2:.2f} seconds")
                        print("Output from LLM:", output)  # Debug: Print the output from the LLM         
                    pattern = r'\{.*?\}'
                    cleaned_output = re.sub(r'\\+n*', '', output)
                    print(cleaned_output)
                    matches = re.findall(pattern, cleaned_output, re.DOTALL)
                    print("Matches Found:", matches)  # Debug: Print the matches

                    try:
                        if matches:
                            match_ = matches[0].strip()
                            json_data = json.loads(match_)
                            prompt_ = json_data['prompt']
                            prompt=f' A single {country},in {select_outfit},3d Pixar,{select_gender},white background,{body_style} ' +prompt_random
                            negative_prompt=f"{country}flag,flags,triplets,multiple characters bad drawing, quicky, incoherent, duplicated, twins, twin, grain, pixels, bad proportions,nsfw, multiple characters,more than one,twins,crowd"
                            print(prompt)
                            # tic3 = time.time()
                            overall_tic = time.time()  # Start overall timer
                            img_gen_tic = time.time()  # Start image generation timer
                            image = pipe(prompt=prompt, width=512,height=512,num_inference_steps=50,guidance_scale=3.5,max_sequence_length=512,).images[0]
                            img_gen_toc = time.time()
                            print(f'Image generation time: {img_gen_toc - img_gen_tic:.2f} seconds')
                            path='test.png'
                            image=image.resize((1024,1024))
                            image.save(path)
                            bg_tic = time.time()
                            image_original=Image.open(path)
                            session = new_session('u2netp')
                            # toc3 = time.time()
                            # print(f'Total time required :- {toc3-tic3}')
                            output_rembg = remove(image_original, session=session)
                            bg_rembg_toc = time.time()
                            print(f'background removal time by rembg (U2NetP): {bg_rembg_toc - bg_tic:.2f} seconds')

                            orig_im = np.array(image_original)
                            orig_im_size = orig_im.shape[0:2]
                            image = preprocess_image(orig_im, [1024,1024]).to('cuda')

                            bria_tic = time.time()
                            result = net(image)
                            # post process
                            result_image = postprocess_image(result[0][0], orig_im_size)
                            bria_toc = time.time()
                            print(f'background removal time by BriaRMBG: {bria_toc - bria_tic:.2f} seconds')

                            # save result
                            combined_bg = combine_masks(result_image, np.array(output_rembg)[:, :, -1])
                            pil_im = Image.fromarray(combined_bg)                            
                            no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
                            orig_image = Image.open(path)
                            no_bg_image.paste(orig_image, mask=pil_im)         
                            threshold = 127
                            output = no_bg_image
                            output = threshold_alpha(output,threshold)
                            # bg_toc = time.time()
                            overall_toc = time.time()
                            # print(f'background removal time by rembg:- {bg_toc-bg_tic}')
                            print(f'Overall time for this image: {overall_toc - overall_tic:.2f} seconds')

                            no_bg_dir = os.path.join(output_dir,'no_bg')
                            os.makedirs(no_bg_dir,exist_ok=True)
                            unique=str(uuid.uuid4())
                            english_=english.replace(' ', '_')
                            name_format=english_+f'_{unique}.png'#
                            path=os.path.join(no_bg_dir,name_format)
                            
                            json_file = json_path
                            ################################################# OTF Code begins here###########################
                            alignments = ['center','side']
                            
                            english=english
                            non_english=keyword
                            text_msg_master_master = {}
                            
                            #dict to maintain english-inside and non english-outside the bracket words
                            text_msg_master_master['outside']=keyword
                            text_msg_master_master['inside']=english                            
                            category_name=extract_category_name(category_name)

                            #loading files for downloading fonts, and mapping of category to those fonts.
                            with open(os.path.join(assets_path,'keyboardPopTextStyles.json'), 'r') as file:
                                pop_text_file_mapping = json.load(file)

                            with open(os.path.join(assets_path,'pop_text_font.json'), 'r') as file:
                                id_mapping = json.load(file)

                            poptext_id_list=id_mapping.get(category_name, [])
                            poptext_id=random.choice(poptext_id_list)
                            
                            print(category_name)
                            print(poptext_id)
                            
                            output.save(path)
                            image = cv2.imread(path,cv2.IMREAD_UNCHANGED)
                            x,y,_ = image.shape
                            
                            #NOte: poptext only supports english, hence we use the "english" part with poptext, for native language we used the default font which is passed from bash script for that country/language                            
                            choice=random.choice(['outside','inside','outside']) #randomly chosing whch otf text to place with more emphasis on non english native keywords
                            print(choice)
                            text_msg = text_msg_master_master[choice]
                            print(text_msg)
                            res=get_colors(category_name)
                            print(res)
                            if choice=='inside':
                                text_color_random = random.choice(get_colors(category_name)) 
                                filename, colors, stroke_color, stroke_width=download_otf_font(pop_text_file_mapping,poptext_id,text_color_random)
                                print(filename, colors, stroke_color, stroke_width)
                                font_path=filename
                                print('font path :- #########################################',font_path)
                            else:
                                colors= random.choice(get_colors(category_name))
                                contrast_bgr, darker_bgr, darker_bgr = bgr_contrast_and_darker_and_lighter(hex_to_bgr(colors)) 
                                font_path=font_default    
                                stroke_color = tuple(random.choice([contrast_bgr]))
                                stroke_width = 5  # default to 1 pixel
                                
                            otf_dir = os.path.join(output_dir,'images')
                            os.makedirs(otf_dir,exist_ok=True)
                            #for alignment in alignments: # Get text color
                            alignment=random.choice(alignments)
                            character_layer, text_layer, text_bbox = generate_image_with_text(text_msg, image,stroke_color, stroke_width, font_path, font_size = 150, color=colors, alignment=alignment)
                            character_layer = cv2.resize(np.array(character_layer), (x, y))   
                            character_layer=merge_layers([character_layer, text_layer])

                            print('#'*100)
                            print(alignment)
                            print('#'*100)
                            
                            ######################writing to metadata.json here#########################################
                            name_format=english_+f'_{alignment}_{unique}.png'#name for final sticker with otf                            
                            cv2.imwrite(os.path.join(otf_dir,name_format),character_layer)                            
                            gen_img_annotations = {} 
                            gen_img_annotations['image_name'] = name_format
                            gen_img_annotations[f'alignment_{alignment}'] = text_bbox
                            gen_img_annotations['prompt'] = prompt
                            gen_img_annotations['negative_prompt']=negative_prompt
                            gen_img_annotations['category']=category_name
                            gen_img_annotations['gender']=select_gender
                            gen_img_annotations['tags']=letter
                            gen_img_annotations['None_english_otf']=keyword
                            gen_img_annotations['English_OTF']=english
                            gen_img_annotations['body style']=body_style
                            gen_img_annotations['sticker_type']=sticker_type
                            output_file = os.path.join(output_dir, 'metadata.json')

                            # Load the existing metadata if the file exists
                            if os.path.exists(output_file):
                                with open(output_file, 'r') as f:
                                    metadata = json.load(f)
                            else:
                                metadata = {}  # Create a new dictionary if the file doesn't exist
                            # Add the current image's metadata to the main dictionary
                            metadata[name_format] = gen_img_annotations

                            with open(output_file, 'w') as f:
                                json.dump(metadata, f, cls=NumpyEncoder, ensure_ascii=False, indent=4)
                            i += 1
                    except:
                        traceback.print_exc()
                        pass
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
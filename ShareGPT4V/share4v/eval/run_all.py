import argparse
from io import BytesIO
import pandas as pd
import os, glob
import requests
import torch
from PIL import Image

from share4v.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from share4v.conversation import SeparatorStyle, conv_templates
from share4v.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from share4v.model.builder import load_pretrained_model
from share4v.utils import disable_torch_init


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name)

    qs = args.query
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
            DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    if 'llama-2' in model_name.lower():
        conv_mode = "share4v_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "share4v_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "share4v_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(
            conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

       
    print("-------processing multiart---------")
    artist_val = pd.read_csv('/scratch/bbmr/syang7/art/Multitask100k/multi_meta.csv') 
    features = []
    for index, row in artist_val.iterrows():
        #print(row['pth'], row['artist'])
        image_pth = row['Image Path']
        artist = str(row['artist_label'])

        images = load_image(image_pth)
        images_tensor = image_processor.preprocess(images, return_tensors='pt')[
            'pixel_values'].half().cuda()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, tokenizer, input_ids)

        with torch.inference_mode():
            # Obtain the model's output logits or embeddings
            output = model(
                input_ids=input_ids,
                images=images_tensor,
                return_dict=True,
                output_hidden_states=True  # Ensure hidden states are returned
            )
        last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
        pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling
        name = os.path.splitext(image_pth)[0].split('/')[-1] 
        save_pth = os.path.join('/scratch/bbmr/syang7/art/Multitask100k/comments_share', artist)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
            #file_path = os.path.join(save_pth, f"{name}.txt")
            #with open(file_path, "w", encoding="utf-8") as file:
                #file.write(outputs)
        file_path = os.path.join(save_pth, f"{name}.pt")
        torch.save(pooled_features, file_path)
        
        
#     print("-------processing wikiart---------")
#     artist_val = pd.read_csv('/scratch/bbmr/syang7/art/wikiart/artist_val.csv', index_col=0) 
#     features = []
#     for index, row in artist_val.iterrows():
#         #print(row['pth'], row['artist'])
#         image_pth = os.path.join('/scratch/bbmr/syang7/art/wikiart/dataset', row['pth'])
#         artist = str(row['artist'])

#         images = load_image(image_pth)
#         images_tensor = image_processor.preprocess(images, return_tensors='pt')[
#             'pixel_values'].half().cuda()

#         input_ids = tokenizer_image_token(
#             prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

#         stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#         keywords = [stop_str]
#         stopping_criteria = KeywordsStoppingCriteria(
#             keywords, tokenizer, input_ids)

#         with torch.inference_mode():
#             # Obtain the model's output logits or embeddings
#             output = model(
#                 input_ids=input_ids,
#                 images=images_tensor,
#                 return_dict=True,
#                 output_hidden_states=True  # Ensure hidden states are returned
#             )
#         last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
#         pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling
#         name = os.path.splitext(image_pth)[0].split('/')[-1] 
#         save_pth = os.path.join('/scratch/bbmr/syang7/art/wikiart/comments_share', artist)
#         if not os.path.exists(save_pth):
#             os.makedirs(save_pth)
#             #file_path = os.path.join(save_pth, f"{name}.txt")
#             #with open(file_path, "w", encoding="utf-8") as file:
#                 #file.write(outputs)
#         file_path = os.path.join(save_pth, f"{name}.pt")
#         torch.save(pooled_features, file_path)
        

#     print("-------processing best artwork---------")
    
    
#     for artist in os.listdir('/scratch/bbmr/syang7/art/Best_artwork/artists/'): # artist
#         for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/Best_artwork/artists/{artist}/*'):
            
#             images = load_image(image_pth)
#             images_tensor = image_processor.preprocess(images, return_tensors='pt')[
#                 'pixel_values'].half().cuda()

#             input_ids = tokenizer_image_token(
#                 prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

#             stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#             keywords = [stop_str]
#             stopping_criteria = KeywordsStoppingCriteria(
#                 keywords, tokenizer, input_ids)

#             with torch.inference_mode():
#                 # Obtain the model's output logits or embeddings
#                 output = model(
#                     input_ids=input_ids,
#                     images=images_tensor,
#                     return_dict=True,
#                     output_hidden_states=True  # Ensure hidden states are returned
#                 )
#             last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
#             pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling
            
#             artist = image_pth.split('/')[-2]
#             name = os.path.splitext(image_pth)[0].split('/')[-1]
#             save_pth = os.path.join('/scratch/bbmr/syang7/art/Best_artwork/comments_share', artist)
#             if not os.path.exists(save_pth):
#                 os.makedirs(save_pth)
#             file_path = os.path.join(save_pth, f"{name}.pt")
#             torch.save(pooled_features, file_path)
            
#     print('----processing constable-------')
#     for artist in os.listdir('/scratch/bbmr/syang7/art/dataset_new/'): # artist
#         for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/dataset_new/{artist}/*'):
            
#             images = load_image(image_pth)
#             images_tensor = image_processor.preprocess(images, return_tensors='pt')[
#                 'pixel_values'].half().cuda()

#             input_ids = tokenizer_image_token(
#                 prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

#             stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#             keywords = [stop_str]
#             stopping_criteria = KeywordsStoppingCriteria(
#                 keywords, tokenizer, input_ids)

#             with torch.inference_mode():
#                 # Obtain the model's output logits or embeddings
#                 output = model(
#                     input_ids=input_ids,
#                     images=images_tensor,
#                     return_dict=True,
#                     output_hidden_states=True  # Ensure hidden states are returned
#                 )
#             last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
#             pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling
            
#             artist = image_pth.split('/')[-2]
#             name = os.path.splitext(image_pth)[0].split('/')[-1]
#             save_pth = os.path.join('/scratch/bbmr/syang7/art/GalleryGPT/GalleryGPT/comments_share', artist)
#             if not os.path.exists(save_pth):
#                 os.makedirs(save_pth)
#             file_path = os.path.join(save_pth, f"{name}.pt")
#             torch.save(pooled_features, file_path)
            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Lin-Chen/ShareGPT4V-7B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
    
#prompt = "Please write a paragraph of formal art analysis for this painting"
import os, glob
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    
    artist_val = pd.read_csv('/scratch/bbmr/syang7/art/Multitask100k/multi_meta.csv', index_col=0) 
    features = []
    for index, row in artist_val.iterrows():
        #print(row['pth'], row['artist'])
        image_pth = row['Image Path']
        artist = str(row['artist'])
        images = load_images([image_pth])
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        with torch.inference_mode():
        # Obtain the model's output logits or embeddings
            output = model(
                input_ids=input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                return_dict=True,
                output_hidden_states=True  # Ensure hidden states are returned
            )

        last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
        pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling
        name = os.path.splitext(image_pth)[0].split('/')[-1] #name = row['pth'].split('/')[-1].split('.')[0]     
        # print(name)
        save_pth = os.path.join('/scratch/bbmr/syang7/art/Multitask100k/comments', artist)
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)
            #file_path = os.path.join(save_pth, f"{name}.txt")
            #with open(file_path, "w", encoding="utf-8") as file:
                #file.write(outputs)
        file_path = os.path.join(save_pth, f"{name}.pt")
        print(file_path)
        torch.save(pooled_features, file_path)
        
        
    for artist in os.listdir('/scratch/bbmr/syang7/art/Best_artwork/artists/'): # artist
        for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/Best_artwork/artists/{artist}/*'): # image
            
            images = load_images([image_pth])
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)
        
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )
            with torch.inference_mode():
            # Obtain the model's output logits or embeddings
                output = model(
                    input_ids=input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    return_dict=True,
                    output_hidden_states=True  # Ensure hidden states are returned
                )

            last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
            pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling

            artist = image_pth.split('/')[-2]
            name = os.path.splitext(image_pth)[0].split('/')[-1]
            save_pth = os.path.join('/scratch/bbmr/syang7/art/Best_artwork/comments', artist)
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            #file_path = os.path.join(save_pth, f"{name}.txt")
            #with open(file_path, "w", encoding="utf-8") as file:
                #file.write(outputs)
                
            file_path = os.path.join(save_pth, f"{name}.pt")
            torch.save(pooled_features, file_path)


    # image_files = image_parser(args)
    
#     for artist in os.listdir('/scratch/bbmr/syang7/art/dataset_new/'): # artist
#         for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/dataset_new/{artist}/*'): # image
            
#             images = load_images([image_pth])
#             image_sizes = [x.size for x in images]
#             images_tensor = process_images(
#                 images,
#                 image_processor,
#                 model.config
#             ).to(model.device, dtype=torch.float16)
        
#             input_ids = (
#                 tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#                 .unsqueeze(0)
#                 .cuda()
#             )
#             with torch.inference_mode():
#             # Obtain the model's output logits or embeddings
#                 output = model(
#                     input_ids=input_ids,
#                     images=images_tensor,
#                     image_sizes=image_sizes,
#                     return_dict=True,
#                     output_hidden_states=True  # Ensure hidden states are returned
#                 )

#             last_hidden_states = output.hidden_states[-1]  # Assuming this is the last layer
#             pooled_features = last_hidden_states.mean(dim=1)  # Simple example of pooling
        
# #             with torch.inference_mode():
# #                 output_ids = model.generate(
# #                     input_ids,
# #                     images=images_tensor,
# #                     image_sizes=image_sizes,
# #                     do_sample=True if args.temperature > 0 else False,
# #                     temperature=args.temperature,
# #                     top_p=args.top_p,
# #                     num_beams=args.num_beams,
# #                     max_new_tokens=args.max_new_tokens,
# #                     use_cache=True,
# #                 )
                

# #             # pth = args.image_file
# #             # print(pth)
# #             outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
# #             # print(outputs)
            
#             artist = image_pth.split('/')[-2]
#             name = os.path.splitext(image_pth)[0].split('/')[-1]
#             save_pth = os.path.join('/scratch/bbmr/syang7/art/GalleryGPT/GalleryGPT/comments_raw_features', artist)
#             if not os.path.exists(save_pth):
#                 os.makedirs(save_pth)
#             #file_path = os.path.join(save_pth, f"{name}.txt")
#             #with open(file_path, "w", encoding="utf-8") as file:
#                 #file.write(outputs)
                
#             file_path = os.path.join(save_pth, f"{name}.pt")
#             torch.save(pooled_features, file_path)


    # return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
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

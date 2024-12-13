import torch
import os, glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from PIL import Image
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# If you expect the results to be reproducible, set a random seed.
# torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
preprocess = Compose([
    Resize((224, 224)),  # Resize to model's expected input size
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



##### wikiart ######
print('------------processing multi-----------')
artist_val = pd.read_csv('/scratch/bbmr/syang7/art/Multitask100k/multi_meta.csv') 

for index, row in artist_val.iterrows():
        #print(row['pth'], row['artist'])
    image_pth = row['Image Path']
    artist = str(row['artist_label'])
    image = Image.open(image_pth).convert("RGB")
    input_image = preprocess(image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        visual_features = model.transformer.visual(input_image)
        # Final feature vector without pooling
        feature_vector = model.transformer.visual.ln_post(visual_features)
        feature_vector_aggregated = feature_vector.mean(dim=1)

    name = os.path.splitext(image_pth)[0].split('/')[-1] #name = row['pth'].split('/')[-1].split('.')[0] 
    # print(name)
    save_pth = os.path.join('/scratch/bbmr/syang7/art/Multitask100k/comments_qwen', artist)
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    file_path = os.path.join(save_pth, f"{name}.pt")
    print(file_path)
    torch.save(feature_vector_aggregated, file_path)
    
    
# ##### wikiart ######
# print('------------processing wikiart-----------')
# artist_val = pd.read_csv('/scratch/bbmr/syang7/art/wikiart/artist_val.csv', index_col=0) 

# for index, row in artist_val.iterrows():
#         #print(row['pth'], row['artist'])
#     image_pth = os.path.join('/scratch/bbmr/syang7/art/wikiart/dataset', row['pth'])
#     artist = str(row['artist'])
#     image = Image.open(image_pth).convert("RGB")
#     input_image = preprocess(image).unsqueeze(0).to("cuda")

#     with torch.no_grad():
#         visual_features = model.transformer.visual(input_image)
#         # Final feature vector without pooling
#         feature_vector = model.transformer.visual.ln_post(visual_features)
#         feature_vector_aggregated = feature_vector.mean(dim=1)

#     name = os.path.splitext(image_pth)[0].split('/')[-1] #name = row['pth'].split('/')[-1].split('.')[0] 
#     # print(name)
#     save_pth = os.path.join('/scratch/bbmr/syang7/art/wikiart/comments_qwen', artist)
#     if not os.path.exists(save_pth):
#         os.makedirs(save_pth)

#     file_path = os.path.join(save_pth, f"{name}.pt")
#     print(file_path)
#     torch.save(feature_vector_aggregated, file_path)
    


# ##### best artwork ######        
# print('----------processing best artwork---------')        
        
# for artist in os.listdir('/scratch/bbmr/syang7/art/Best_artwork/artists/'): # artist
#     for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/Best_artwork/artists/{artist}/*'): # image

#         image = Image.open(image_pth).convert("RGB")
#         input_image = preprocess(image).unsqueeze(0).to("cuda")
        
#         with torch.no_grad():
#             visual_features = model.transformer.visual(input_image)
#             # Final feature vector without pooling
#             feature_vector = model.transformer.visual.ln_post(visual_features)
#             feature_vector_aggregated = feature_vector.mean(dim=1)

#         artist = image_pth.split('/')[-2]
#         name = os.path.splitext(image_pth)[0].split('/')[-1]
#         save_pth = os.path.join('/scratch/bbmr/syang7/art/Best_artwork/comments_qwen', artist)
#         if not os.path.exists(save_pth):
#             os.makedirs(save_pth)
#         file_path = os.path.join(save_pth, f"{name}.pt")
#         torch.save(feature_vector_aggregated, file_path)
        
        
        

# ##### best artwork ######        
# print('----------processing constable---------')        
        
# for artist in os.listdir('/scratch/bbmr/syang7/art/dataset_new'): # artist
#     for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/dataset_new/{artist}/*'): # image

#         image = Image.open(image_pth).convert("RGB")
#         input_image = preprocess(image).unsqueeze(0).to("cuda")
        
#         with torch.no_grad():
#             visual_features = model.transformer.visual(input_image)
#             # Final feature vector without pooling
#             feature_vector = model.transformer.visual.ln_post(visual_features)
#             feature_vector_aggregated = feature_vector.mean(dim=1)

#         artist = image_pth.split('/')[-2]
#         name = os.path.splitext(image_pth)[0].split('/')[-1]
#         save_pth = os.path.join('/scratch/bbmr/syang7/art/GalleryGPT/GalleryGPT/comments_qwen', artist)
#         if not os.path.exists(save_pth):
#             os.makedirs(save_pth)
#         file_path = os.path.join(save_pth, f"{name}.pt")
#         torch.save(feature_vector_aggregated, file_path)
# print('-----finish constable-------')
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import pandas as pd
import os

#model = models.resnet50(pretrained=True)
model = models.resnet34(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier layer for features
model.eval()
model.cuda()
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize image
    transforms.ToTensor(),           # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])



def extract_features(image_path, model, transform):
    # Load image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to("cuda")  # Add batch dimension
    
    with torch.no_grad():
        features = model(image)
    return features.view(-1) # Flatten to 4096-dim


##### wikiart ######
print('------------processing multi-----------')
artist_val = pd.read_csv('/scratch/bbmr/syang7/art/Multitask100k/multi_meta.csv') 

for index, row in artist_val.iterrows():
        #print(row['pth'], row['artist'])
    image_pth = row['Image Path']
    artist = str(row['artist_label'])
    with torch.no_grad():
        feature_vector_aggregated = extract_features(image_pth, model, transform)

    name = os.path.splitext(image_pth)[0].split('/')[-1] 
    save_pth = os.path.join('/scratch/bbmr/syang7/art/Multitask100k/comments_resnet', artist)
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)

    file_path = os.path.join(save_pth, f"{name}.pt")
    #print(file_path)
    torch.save(feature_vector_aggregated, file_path)
    
    
# #### wikiart ######
# print('------------processing wikiart-----------')
# artist_val = pd.read_csv('/scratch/bbmr/syang7/art/wikiart/artist_val.csv', index_col=0) 

# for index, row in artist_val.iterrows():
#         #print(row['pth'], row['artist'])
#     image_pth = os.path.join('/scratch/bbmr/syang7/art/wikiart/dataset', row['pth'])
#     artist = str(row['artist'])
#     feature_vector_aggregated = extract_features(image_pth, model, transform)
# #     image = Image.open(image_pth).convert("RGB")
# #     input_image = transform(image).unsqueeze(0).to("cuda")

# #     with torch.no_grad():
# #         visual_features = model(input_image)
# #         # Final feature vector without pooling
# #         feature_vector_aggregated = visual_features.view(-1)

#     name = os.path.splitext(image_pth)[0].split('/')[-1] #name = row['pth'].split('/')[-1].split('.')[0] 
#     # print(name)
#     save_pth = os.path.join('/scratch/bbmr/syang7/art/wikiart/comments_resnet2', artist)
#     if not os.path.exists(save_pth):
#         os.makedirs(save_pth)

#     file_path = os.path.join(save_pth, f"{name}.pt")
#     #print(file_path)
#     torch.save(feature_vector_aggregated, file_path)
    


# ##### best artwork ######        
# print('----------processing best artwork---------')        
        
# for artist in os.listdir('/scratch/bbmr/syang7/art/Best_artwork/artists/'): # artist
#     for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/Best_artwork/artists/{artist}/*'): # image
#         feature_vector_aggregated = extract_features(image_pth, model, transform)

# #         image = Image.open(image_pth).convert("RGB")
# #         input_image = preprocess(image).unsqueeze(0).to("cuda")
        
# #         with torch.no_grad():
# #             visual_features = model.transformer.visual(input_image)
# #             # Final feature vector without pooling
# #             feature_vector = model.transformer.visual.ln_post(visual_features)
# #             feature_vector_aggregated = feature_vector.mean(dim=1)

#         artist = image_pth.split('/')[-2]
#         name = os.path.splitext(image_pth)[0].split('/')[-1]
#         save_pth = os.path.join('/scratch/bbmr/syang7/art/Best_artwork/comments_resnet2', artist)
#         if not os.path.exists(save_pth):
#             os.makedirs(save_pth)
#         file_path = os.path.join(save_pth, f"{name}.pt")
#         torch.save(feature_vector_aggregated, file_path)
        
        
        

# ##### best artwork ######        
# print('----------processing constable---------')        
        
# for artist in os.listdir('/scratch/bbmr/syang7/art/dataset_new'): # artist
#     for image_pth in glob.glob(f'/scratch/bbmr/syang7/art/dataset_new/{artist}/*'): # image
#         feature_vector_aggregated = extract_features(image_pth, model, transform)

# #         image = Image.open(image_pth).convert("RGB")
# #         input_image = preprocess(image).unsqueeze(0).to("cuda")
        
# #         with torch.no_grad():
# #             visual_features = model.transformer.visual(input_image)
# #             # Final feature vector without pooling
# #             feature_vector = model.transformer.visual.ln_post(visual_features)
# #             feature_vector_aggregated = feature_vector.mean(dim=1)

#         artist = image_pth.split('/')[-2]
#         name = os.path.splitext(image_pth)[0].split('/')[-1]
#         save_pth = os.path.join('/scratch/bbmr/syang7/art/GalleryGPT/GalleryGPT/comments_resnet2', artist)
#         if not os.path.exists(save_pth):
#             os.makedirs(save_pth)
#         file_path = os.path.join(save_pth, f"{name}.pt")
#         torch.save(feature_vector_aggregated, file_path)
# print('-----finish constable---')
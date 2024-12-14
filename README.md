# CSE597_Project


This repo contains the codes for CSE 597 course project. It utlizes pretrained models from [GalleryGPT](https://github.com/steven640pixel/GalleryGPT), [ShareGPT4v](https://github.com/ShareGPT4Omni/ShareGPT4V) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL). 

For the datasets, WikiArt can be downloaded at [this link](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset), the dataset Multitask Painting100k can be downloaded at [this link](http://www.ivl.disco.unimib.it/activities/paintings/), and the BestArtwork dataset can be downloaded at [this link](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/code?datasetId=130081&sortBy=voteCount). 


## Qwen-VL
To extract features using Qwen-VL, first refer to [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for environment requirements.
Run the following script for feature extraction:
```
python Qwen_extract.py
```
## ShareGPT4v

```
cd ShareGPT4V/share4v/eval
python run_all.py --model-path Lin-Chen/ShareGPT4V-7B --query 
```
## GalleryGPT
```
cd GalleryGPT
conda create -n gallery_gpt python=3.10 -y
conda activate gallery_gpt
pip install -e .
pip install protobuf
```
After installing packages, run the following codes:
```
cd GalleryGPT/llava/eval
python run_all.py --model-path llava-lora-model --model-base share4v/llava-7b --image-file your/image/path --query
```

## Baselines: ResNet
Run the following codes for resnet-feature extraction:
```
python Baselines_extract.py
```
## Regression 
After extracting all features, arrange them in mapping csv as listed in mapping_wikiart.csv, and run:
```
python reg_all.py
```

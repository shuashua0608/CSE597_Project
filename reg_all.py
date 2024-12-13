import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import SGDClassifier

def split_data(features, labels):
    return train_test_split(features, labels, test_size=0.2, random_state=42)

# Train and evaluate model
def train_and_evaluate(features, labels, task):
    X_train, X_test, y_train, y_test = split_data(features, labels)
    
    if task == 'classification':
#         model = SGDClassifier(loss='log', max_iter=1000)
#         model.fit(X_train, y_train)

        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # score = accuracy_score(y_test, y_pred)
        # metric_name = "Accuracy"
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        print('Classification report: ', classification_report(y_test, y_pred))
        
    elif task == 'regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # score = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
        # metric_name = "RMSE"
        metrics = {
            "RMSE": mean_squared_error(y_test, y_pred, squared=False)
        }
    # else:
    #     raise ValueError("Task must be 'classification' or 'regression'")
    
    return metrics


def reg_classifier(df, task = 'classification'):
    
    labels = df['artist_label'] #.to_numpy()
    features_resnet34 = []
    features_resnet50 = []
    features_gallery = []
    features_qwen = []
    features_share = []
    
    for feature_file in df['Feature File01']:
        # Load torch tensor for each painting
        feature_tensor = torch.load(feature_file)
        features_resnet34.append(feature_tensor.cpu().to(torch.float32).numpy()) #.cpu().numpy()
   
    for feature_file in df['Feature File02']:
        # Load torch tensor for each painting
        feature_tensor = torch.load(feature_file)
        features_resnet50.append(feature_tensor.cpu().to(torch.float32).numpy()) #.cpu().numpy()
   
    for feature_file in df['Feature File1']:
        # Load torch tensor for each painting
        feature_tensor = torch.load(feature_file)[0]
        features_gallery.append(feature_tensor.cpu().to(torch.float32).numpy()) #.cpu().numpy()
   
    for feature_file in df['Feature File2']:
    # Load torch tensor for each painting
        feature_tensor = torch.load(feature_file)[0]
        features_qwen.append(feature_tensor.cpu().to(torch.float32).numpy()) #.cpu().numpy()

    for feature_file in df['Feature File3']:
        # Load torch tensor for each painting
        feature_tensor = torch.load(feature_file)[0]
        features_share.append(feature_tensor.cpu().to(torch.float32).numpy()) #.cpu().numpy()

    # Compare feature sets
    feature_sets = [features_resnet34, features_resnet50, features_gallery, features_qwen, features_share]
    feature_names = ['features_resnet34', 'features_resnet50', 'features_gallery', 'features_qwen', 'features_share']

    for i, features in enumerate(feature_sets):
        print(f"{feature_names[i]}:")
        metrics = train_and_evaluate(features, labels, task)
        for metric_name, score in metrics.items():
            print(f"  {metric_name}: {score:.4f}")
            
            
if __name__ =='__main__':          
    
    print('---processing multitask artwork---')
    df = pd.read_csv('../mapping_multitask.csv')
    reg_classifier(df)
    
#     print('---processing best artwork---')
#     df = pd.read_csv('../mapping_best_artwork.csv')
#     reg_classifier(df)


#     print('---processing wikiart---')
#     df = pd.read_csv('../mapping_wikiart.csv')
#     reg_classifier(df)


#     print('---processing cloud---')
#     df = pd.read_csv('../mapping_cloud_dataset.csv')
#     reg_classifier(df)
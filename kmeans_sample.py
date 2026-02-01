import cv2
from PIL import Image
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import transforms
from tqdm import tqdm
import random

def rgb_loader(path):
    img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img, mode='RGB')
    return img


if __name__ == '__main__':
    data_path = './Dataset/TrainDataset/Imgs'
    img_list = sorted(os.listdir(data_path))
    features_save_path = './dino_b_features.npy'

    img_transforms = transforms.Compose([
        transforms.Resize((392, 392)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
    dino.load_state_dict(torch.load('./dinov2_vitl14_pretrain.pth'))
    dino = dino.to('cuda')
    dino.eval()

    features = []

    for img_name in tqdm(img_list):
        img = rgb_loader(os.path.join(data_path, img_name))
        img = img_transforms(img)
        img = img.unsqueeze(0).to('cuda')
        with torch.no_grad():
            feat = dino(img)
            features.append(feat.squeeze(0).cpu().numpy())

    features = np.stack(features, axis=0)
    np.save(features_save_path, features)
    print(f'Features saved to: {features_save_path}')

    features = np.load(features_save_path)
    print(f'Features loaded from: {features_save_path}')

    random_seed = random.randint(0, 10000)
    print(f"Random seed: {random_seed}")
    n_clusters = 40
    top_k = 5  # for each cluster, select top k samples closest to the cluster center
    print("Running KMeans clustering...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    selected_indices = set()

    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_features = features[cluster_indices]

        # calculate distances to cluster center
        distances = np.linalg.norm(cluster_features - centers[i], axis=1)
        sorted_cluster_indices = cluster_indices[np.argsort(distances)]

        # select top k samples closest to the cluster center
        topk_indices = sorted_cluster_indices[:top_k]
        selected_indices.update(topk_indices)

    final_indices = [idx for idx in range(len(features)) if idx in selected_indices]

    # Save the selected image names to a text file
    with open('./sampled_images.txt', 'w') as f:
        for idx in final_indices:
            f.write(f"{img_list[idx][:-4]}\n")

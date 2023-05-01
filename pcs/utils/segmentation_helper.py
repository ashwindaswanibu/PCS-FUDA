import torch
import numpy as np
import torch.nn as nn
import cv2

def masked_select(embeddings, masks):
    embeddings =embeddings.reshape(64, 512)
    masks = masks.reshape(64, 30)

    keys = np.arange(30)
    values = []
    # select all indices of masks which have the same value and append all corresponding embeddings to a list
    indices = [torch.nonzero(masks[:, i], as_tuple=False) for i in range(masks.shape[1])]
    
    embeddings_list = [embeddings[indices[i]] for i in range(len(indices))]
      
    embeddings_list = [embeddings_list[i].float() for i in range(len(embeddings_list))]
    mean_embeddings = []
    mean_embeddings = [torch.mean(embeddings_list[i], dim=0) for i in range(len(embeddings_list))]
    # print(mean_embeddings)
    dictionary_indices = dict(zip(keys, indices))
    dictionary_mean = dict(zip(keys, mean_embeddings))
    
    return dictionary_indices, dictionary_mean



def masked_select_on_batch(batch_data):
    dictionary_indices = []
    dictionary_mean = []
    for i in range(batch_data.shape[0]):
       temp_dict_indices , temp_dict_mean = masked_select(batch_data[i])
       dictionary_indices.append(temp_dict_indices)
       dictionary_mean.append(temp_dict_mean)
    return dictionary_indices, dictionary_mean


def assign_labels_to_regions(cosine_classifier, img_embedding):
    mask = torch.zeros(64, 30)
    count = 0
    for i in range(len(img_embedding)):
        for j in range(len(img_embedding[i])):
            
           mask[count] = cosine_classifier(img_embedding[i][j])
           count += 1
            
    return mask 

import torch

def assign_labels_to_regions_batch(cosine_classifier, img_embeddings_batch):
    batch_size, img_embedding_rows, img_embedding_cols = img_embeddings_batch.shape
    masks = torch.zeros(batch_size, img_embedding_rows * img_embedding_cols, 30)

    # Reshape the image embeddings to apply cosine_classifier
    reshaped_embeddings = img_embeddings_batch.view(batch_size, -1, img_embedding_cols)
    
    # Apply cosine_classifier to all embeddings in the batch
    masks = cosine_classifier(reshaped_embeddings)

    return masks

def initialize_centroids(feat, batch_size, klist, cosine_classifier):
    lbd = cosine_classifier(feat)
    #lbd = lbd.reshape(batch_size, 64, 30)
    assert lbd.shape == (batch_size, 64, 30) 
    lbd = lbd.reshape(batch_size, 64, 30)
    lbd = lbd.reshape(batch_size*64, 30)
    
    masks = lbd
    embeddings = feat.reshape(batch_size* 64, 512)
    
    unique_labels = torch.unique(lbd, dim=0)
    keys = np.arange(30)
    
        
    
    indices = [torch.nonzero(masks[:, i], as_tuple=False) for i in range(masks.shape[1])]

    embeddings_list = [embeddings[indices[i]] for i in range(len(indices))]
  
    embeddings_list = [embeddings_list[i].float() for i in range(len(embeddings_list))]
    mean_embeddings = []
    mean_embeddings = [torch.mean(embeddings_list[i], dim=0) for i in range(len(embeddings_list))]
    
    
    # print(mean_embeddings)
    #dictionary_indices = dict(zip(keys, indices))
    dictionary_mean = dict(zip(keys, mean_embeddings))
    if len(unique_labels) < 30:
    
        for i in range(30):
            if i not in unique_labels:
                #assign random embedding
                dictionary_mean[i] = torch.rand(512)
    nparray = np.array(list(dictionary_mean.values()))
    return nparray


def convert_image_to_regions(feature):
    batch_size, img_embedding_rows, img_embedding_cols, img_embedding_dim = feature.shape
    
    secondary_indices = [torch.arange(img_embedding_rows * img_embedding_cols)] * batch_size
    regions = feature.reshape(batch_size* img_embedding_rows* img_embedding_cols, img_embedding_dim)
    return regions, secondary_indices

import numpy as np
import numpy as np

import numpy as np

import numpy as np

import numpy as np

import numpy as np

def sparse_make(rgb_img, dick):
    # Get number of channels, height, and width from the input tensor
    num_channels, height, width = rgb_img.shape

    # Check if the number of channels is 4 (RGBA), and if so, extract only the RGB channels
    if num_channels > 3:
        rgb_img = rgb_img[:3, :, :]

    # Dimensions of the new_mask (num_mask_channels, height, width)
    num_mask_channels = 30
    new_mask = np.zeros((num_mask_channels, height, width))

    # Iterate through the pixels of the RGB image
    for i in range(height):
        for j in range(width):
            # Get the RGB values of the pixel as a tuple (key)
            rgb_key = tuple(rgb_img[:, i, j])

            # Check if the RGB key is present in the dictionary
            if rgb_key in dick:
                # Get the channel index from the dictionary based on the RGB key
                channel = dick[rgb_key]

                # Set the corresponding channel in new_mask to 1
                if channel is not None and channel < num_mask_channels:
                    new_mask[channel, i, j] = 1

    return new_mask



def transform_loader(dataloader):
    for i, (img, mask) in enumerate(dataloader):
    
        mask = one_hot_mask_data_loader(mask)
        dataloader[i] = (img,mask)
    return dataloader
    # print(mask.shape)

    


def one_hot_mask_data_loader(img):
    dick = np.load('labels.npy',allow_pickle='TRUE').item()
    img = np.array(img)
    sparse_mat = sparse_make(img, dick)
    return sparse_mat

def one_hot_masks(train_dataset):
    
    list_sparse_mat=[]
    dick = np.load('labels.npy',allow_pickle='TRUE').item()
    for img in train_dataset.targets:
        
        x = np.array(x)

        sparse_mat = sparse_make(img, dick)
        list_sparse_mat.append(sparse_mat)
    return np.array(list_sparse_mat)


def down_sample_masks(img):
    #img(b,256,256,30)
    img = torch.from_numpy(img)
    img= img.permute(2,0,1)
    #img(30,256,256)
    max_pool = nn.MaxPool2d(32, stride=32)
    down = max_pool(img)
    #down(30,8,8)
    down = down.permute(1,2,0)
    return np.array(down)

def down_sample_masks_dataset(img_batched):
    arr= []
    for img in img_batched:
        down = down_sample_masks(img)
        arr.append(down)
    return np.array(arr)
        

    
    
    




    
    
    
    
    
    
    
    
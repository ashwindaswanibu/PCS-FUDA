import torch
import numpy as np

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
    
    return dictionary_mean
     
    
    
    
    




    
    
    
    
    
    
    
    
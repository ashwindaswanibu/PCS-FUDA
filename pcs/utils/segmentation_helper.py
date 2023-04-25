import torch
def masked_select(embedding, masks, model):
    # embedding size: (batch_size, 8, 8, 512)
    # masks size: (batch_size, 8, 8, 1)
    for i in range(len(masks)):
        for j in range(len(masks[i])):
            
                if masks[i][j][k] == 0:
                    embedding[i][j][k] = torch.zeros(512).to(model.device)
    
    
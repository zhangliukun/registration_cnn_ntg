
from sklearn import metrics
import torch

def calculate_mutual_info_batch(source_image_batch,target_image_batch):
    batch,c,h,w = source_image_batch.shape
    mutual_info_list = []
    for i in range(batch):
        mutual_info_list.append(metrics.mutual_info_score(
            torch.flatten(source_image_batch[i]).cpu().detach().numpy(),
            torch.flatten(target_image_batch[i]).cpu().detach().numpy()))

    return mutual_info_list

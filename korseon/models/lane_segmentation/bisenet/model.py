import math
import torch
import torch.nn.functional as F
import numpy as np
import cv2

import bisenet.lib.data.transform_cv2 as T
from bisenet.lib.models import model_factory
from bisenet.configs import set_cfg_from_file

class BiSeNetModel:
  def __init__(self, config='bisenet/configs/bisenetv2_city.py', weights='bisenet/weights/model_final_v1_city_new.pth'):
    torch.set_grad_enabled(False)
    np.random.seed(123)
    cfg = set_cfg_from_file(config)

    palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

    # define model
    self.net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
    self.net.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
    self.net.eval()

    # prepare data
    self.to_tensor = T.ToTensor(
        mean=(0.3257, 0.3690, 0.3223), # city, rgb
        std=(0.2112, 0.2148, 0.2115),
    )


  def predict(self, img_path, threshold):
    im = cv2.imread(img_path)[:, :, ::-1]
    im = self.to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    lane_class_id = 7 # road
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    out_raw = self.net(im)[0]
    out_resized = F.interpolate(out_raw, size=org_size, align_corners=False, mode='bilinear')
    out_prob = F.softmax(out_resized, dim=1)  # Shape: (batch_size, n_classes, height, width)
    lane_prob = out_prob[:, lane_class_id]  # Shape: (batch_size, height, width)

    # Convert lane probabilities to a numpy array
    lane_prob_np = lane_prob.squeeze().cpu().numpy()

    # Normalize probabilities
    lane_prob_normalized = (lane_prob_np - lane_prob_np.min()) / (lane_prob_np.max() - lane_prob_np.min())

    # Compute dynamic threshold using percentile
    percentile_threshold = np.percentile(lane_prob_normalized, 95)

    ### Fixed Mask
    # Apply fixed and percentile thresholds
    fixed_mask = lane_prob_normalized > threshold
    percentile_mask = lane_prob_normalized > percentile_threshold

    # Combine the masks (logical OR)
    lane_mask = np.logical_or(fixed_mask, percentile_mask).astype(np.uint8) * 255
    # visualize
    # lane_mask = (lane_prob_normalized > args.threshold).squeeze().astype(np.uint8) * 255
    print("Lane Mask")
    cv2.imwrite('./results/lane_mask.jpg', lane_mask)
    return lane_mask

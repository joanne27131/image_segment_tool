import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import sys
# yolo_path = 'C:/Users/joann/Desktop/Joanne_Project/Image_Recognition_Side_Project'
# sys.path.append(yolo_path)
from detect_test import detect
# sys.path.remove(yolo_path)
yolo_detect = detect()

class mask():

    def show_mask(self, mask, ax, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        
    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
        
    def show_box(self, box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

    def show_res(self, masks, scores, input_point, input_label, input_box, filename, image):
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            self.show_mask(mask, plt.gca())
            if input_box is not None:
                box = input_box[i]
                self.show_box(box, plt.gca())
            if (input_point is not None) and (input_label is not None): 
                self.show_points(input_point, input_label, plt.gca())
            
            print(f"Score: {score:.3f}")
            plt.axis('off')
            plt.savefig(filename,bbox_inches='tight',pad_inches=-0.1)
            plt.close()

    def show_res_multi(self, masks, scores, input_point, input_label, input_box, filename, image):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            self.show_mask(mask, plt.gca(), random_color=True)
        for box in input_box:
            self.show_box(box, plt.gca())
        for score in scores:
            print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight',pad_inches=-0.1)
        plt.close()

    def save_idivisual_imag(self, image, mask, filename) :
        # Check image and mask is None or not
        if image is None:
            raise ValueError("Error: 'image' is None. Please check if the image is correctly loaded.")
        if mask is None:
            raise ValueError("Error: 'mask' is None. Please check if the mask is correctly generated or passed.")
        
        # 如果 masks 是三维的（即包含多个 mask），选择一个或合并所有 mask
        if len(mask.shape) == 3:
            # 使用合并 mask 的方法
            mask = np.max(mask, axis=0)  # 合并所有的 mask
        
        masks = (mask.astype(np.uint8)) * 255
        # masks = np.squeeze(masks)
        if masks.shape[:2] != image.shape[:2]:
            print(f"Resizing mask from {masks.shape[:2]} to {image.shape[:2]}")
            masks = cv2.resize(masks, (image.shape[1], image.shape[0]))

        segmented_object = cv2.bitwise_and(image, image, mask=masks)
        plt.imshow(segmented_object)
        plt.axis('off')
        plt.savefig(filename,bbox_inches='tight',pad_inches=-0.1)
        plt.close()
        print('save indivisual image')
        return segmented_object

    def get_mask(self, image_path):
        sam_checkpoint = "sam_hq/pretrained_checkpoint/sam_hq_vit_b.pth"
        model_type = "vit_b"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)

        # hq_token_only: False means use hq output to correct SAM output. 
        #                True means use hq output only. 
        #                Default: False
        hq_token_only = False 
        # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
        # For images contain single object, we suggest to set hq_token_only = True
        # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

        file_name = os.path.basename(image_path)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_box_1 = yolo_detect.run(image_path)[1]
        input_box_1 = input_box_1.detach()

        
        # multi box input
        # input_box = torch.tensor([[1, 192, 316, 411], [434, 212, 596, 348]],device=predictor.device)
        transformed_box = predictor.transform.apply_boxes_torch(input_box_1, image.shape[:2])
        input_point, input_label = None, None

        batch_box = False if input_box_1 is None else len(input_box_1)>1 
        result_path = 'hq_sam_result/'
        indivisual_result_path = 'hq_sam_indivisual_result/'
        os.makedirs(result_path, exist_ok=True)
        os.makedirs(indivisual_result_path, exist_ok=True)

        if not batch_box: 
            input_box_1 = input_box_1.cpu().numpy()
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box = input_box_1,
                multimask_output=False,
                hq_token_only=hq_token_only, 
            )
            self.show_res(masks,scores,input_point, input_label, input_box_1, result_path + file_name, image)
            return self.save_idivisual_imag(image, masks, indivisual_result_path+file_name)
            
        else:
            masks, scores, logits = predictor.predict_torch(
                point_coords=input_point,
                point_labels=input_label,
                boxes=transformed_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
            masks = masks.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            input_box = input_box_1.cpu().numpy()
            self.show_res_multi(masks, scores, input_point, input_label, input_box, result_path+file_name, image)
            return self.save_idivisual_imag(image, masks, indivisual_result_path+file_name)

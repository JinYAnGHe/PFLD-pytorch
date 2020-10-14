import argparse
import time
import os

import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2

from mmdet.apis import inference_detector, init_detector
from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces, show_bboxes

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def decode_detections(detections, conf_t=0.5):
    results = []
    for detection in detections:
        confidence = detection[4]

        if confidence > conf_t:
            left, top, right, bottom = detection[:4]
            results.append([int(left), int(top), int(right), int(bottom), confidence])

    return np.array(results)


def draw_detections(frame, detections, class_name):
    """Draws detections and labels"""
    for rect in detections:
        left, top, right, bottom = rect[0]
        cv2.rectangle(frame, (left, top), (right, bottom),
                     (0, 255, 0), thickness=2)
        # label = class_name + '(' + str(round(rect[1], 2)) + ')'
        # label_size, base_line = cv2.getTextSize(label,
        #                                        cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        # top = max(top, label_size[1])
        # cv2.rectangle(frame, (left, top - label_size[1]),
        #              (left + label_size[0], top + base_line),
        #              (255, 255, 255), cv.FILLED)
        # cv2.putText(frame, label, (left, top),
        #            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def main(args):
    # face detection model
    model = init_detector(args.config, args.face_model)

    # landmark model
    checkpoint = torch.load(args.mark_model, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    transform = transforms.Compose([transforms.ToTensor()])

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    save_path = '/home/yang/mark.mp4'
    writer = cv2.VideoWriter(save_path, fourcc, 30.0, (1280, 720), True)
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret: break

        height, width = img.shape[:2]

        # bounding_boxes, landmarks = detect_faces(img)
        results = inference_detector(model, img)
        bboxs = decode_detections(results[0], args.d_thresh)

        for i, bbox in enumerate(bboxs):
            
            # x1, y1, x2, y2 = (bbox[:4]+0.5).astype(np.int32)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            add = int(max(w, h))
            bimg = cv2.copyMakeBorder(img, add, add, add, add,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=[127., 127., 127.])
            bbox += add

            face_width = (1 + 0.4) * w
            center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            
            bbox[0] = center[0] - face_width // 2
            bbox[1] = center[1] - face_width // 2
            bbox[2] = center[0] + face_width // 2
            bbox[3] = center[1] + face_width // 2
            bbox = bbox.astype(np.int)

            crop_image = bimg[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            height, width, _ = crop_image.shape
            crop_image = cv2.resize(crop_image, (112, 112))

            cv2.imshow('cropped face %d ' % i, crop_image)

            # input = cv2.resize(cropped, (112, 112))
            input = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
            input = transform(input).unsqueeze(0).to(device)
            _, landmarks = plfd_backbone(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [width, height]

            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (bbox[0] + x - add, bbox[1] + y - add), 1, (0, 255, 0), -1)

        cv2.imshow('0', img)
        writer.write(img)
        if cv2.waitKey(1) == 27:
            break



def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--mark_model',
        default="./checkpoint/snapshot/checkpoint_epoch_50.pth.tar",
        type=str)
    parser.add_argument(
        '--face_model',
        default='./detector/wider_face_tiny_ssd_075x_epoch_70.pth',
        type=str)
    parser.add_argument(
        '--config',
        default='./detector/configs/mobilenetv2_tiny_ssd300_wider_face.py',
        type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--d_thresh', type=float, default=0.5,
                        help='Threshold for FD')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
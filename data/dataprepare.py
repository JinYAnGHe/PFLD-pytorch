import os
import random
import cv2
import numpy as np
import copy
from tqdm import tqdm

DEBUG = False

def draw_landmark_point(image, points, image_size):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(image_size * point[0]), int(
            image_size * point[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)

def read_points(file_name=None):
    """
    Read points from .pts file.
    """
    points = []
    with open(file_name) as file:
        line_count = 0
        for line in file:
            if "version" in line or "points" in line or "{" in line or "}" in line:
                continue
            else:
                loc_x, loc_y = line.strip().split(sep=" ")
                points.append([float(loc_x), float(loc_y)])
                line_count += 1
    return points

def get_images_list(root_path):
    imgs_list = []
    for root, _, files in os.walk(root_path):
        for f in files:
            if '.jpg' in f or '.png' in f or '.jpeg' in f:
                img_path = os.path.join(root, f)
                imgs_list.append(img_path)
    print('total imgs: ', len(imgs_list))
    return imgs_list

def compute_dist(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1-pt2)))

def augmentationCropImage(img, bbox, landmark, is_training=True, image_size=160, x_shift=0.1, y_shift=0.2):
    
    joints = copy.deepcopy(landmark)
    bbox = np.array(bbox).reshape(4, ).astype(np.float32)
    add = max(img.shape[0], img.shape[1])

    bimg = cv2.copyMakeBorder(img, add, add, add, add, borderType=cv2.BORDER_CONSTANT, value=[127., 127., 127.])
    
    

    objcenter = np.array([(bbox[0] + bbox[2]) / 2., (bbox[1] + bbox[3]) / 2.])
    bbox += add
    objcenter += add
    joints[:, :2] += add

    gt_width = (bbox[2] - bbox[0])
    gt_height = (bbox[3] - bbox[1])

    crop_width_half = gt_width * (1 + x_shift* 2) // 2
    crop_height_half = gt_height * (1 + y_shift * 2) // 2

    if is_training:
        min_x = int(objcenter[0] - crop_width_half + \
                    random.uniform(-x_shift, x_shift) * gt_width)
        max_x = int(objcenter[0] + crop_width_half + \
                    random.uniform(-x_shift, x_shift) * gt_width)
        min_y = int(objcenter[1] - crop_height_half + \
                    random.uniform(-y_shift, y_shift) * gt_height)
        max_y = int(objcenter[1] + crop_height_half + \
                    random.uniform(-y_shift, y_shift) * gt_height)
    else:
        min_x = int(objcenter[0] - crop_width_half)
        max_x = int(objcenter[0] + crop_width_half)
        min_y = int(objcenter[1] - crop_height_half)
        max_y = int(objcenter[1] + crop_height_half)

    joints[:, 0] = joints[:, 0] - min_x
    joints[:, 1] = joints[:, 1] - min_y

    img = bimg[min_y:max_y, min_x:max_x, :]

    crop_image_height, crop_image_width, _ = img.shape
    joints[:, 0] = joints[:, 0] / crop_image_width
    joints[:, 1] = joints[:, 1] / crop_image_height


    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST,
                        cv2.INTER_LANCZOS4]
    interp_method = random.choice(interp_methods)

    img = cv2.resize(img, (image_size, image_size),interpolation=interp_method)

    joints[:, 0] = joints[:, 0] * image_size
    joints[:, 1] = joints[:, 1] * image_size

    return img, joints

class ImageData():
    def __init__(self, image_size=160):
        self.image_size = image_size
        # self.imgs_list = imgs_list
        self.repeat = 10
        self.eye_close_thres=0.02
        self.mouth_close_thres = 0.02
        self.big_mouth_open_thres = 0.08
        self.imgs = []
        self.landmarks = []
    
    def balance(self, imgs_list):
        new_imgs_list = copy.deepcopy(imgs_list)
        for img in tqdm(imgs_list):
            label_path = img.rsplit('.', 1)[0] + '.pts'
            landmark = read_points(label_path)
            landmark = np.array(landmark)

            xy = np.min(landmark, axis=0).astype(np.int32) 
            zz = np.max(landmark, axis=0).astype(np.int32)
            bbox_width, bbox_height = zz - xy + 1

            if bbox_height < 50 or bbox_width < 50:
                new_imgs_list.remove(img)

            # eye closed
            left_eye_close = compute_dist(landmark[37], landmark[41]) / bbox_height < self.eye_close_thres \
                or compute_dist(landmark[38], landmark[40]) / bbox_height < self.eye_close_thres
            right_eye_close = compute_dist(landmark[43], landmark[47]) / bbox_height < self.eye_close_thres \
                or compute_dist(landmark[44], landmark[46]) / bbox_height < self.eye_close_thres
            if left_eye_close or right_eye_close:
                for i in range(10):
                    new_imgs_list.append(img)

            if left_eye_close and not right_eye_close:
                for i in range(40):
                    new_imgs_list.append(img)

            if not left_eye_close and right_eye_close:
                for i in range(40):
                    new_imgs_list.append(img)

            # half face
            is_half_face = compute_dist(landmark[36], landmark[45]) / bbox_width < 0.5
            if is_half_face:
                for i in range(20):
                    new_imgs_list.append(img)
            
            # mouth
            is_opened_mouth = compute_dist(landmark[62], landmark[66]) / bbox_height > 0.15
            if is_opened_mouth:
                for i in range(20):
                    new_imgs_list.append(img)

            is_big_mouth = compute_dist(landmark[62], landmark[66]) / self.image_size > self.big_mouth_open_thres
            if is_big_mouth:
                for i in range(50):
                    new_imgs_list.append(img)
            
            if DEBUG and bbox_height > 50 and bbox_width > 50:
                print('left_eye_close: ', left_eye_close)
                print('right_eye_close: ', right_eye_close)
                print('is_half_face: ', is_half_face)
                print('is_opened_mouth: ', is_opened_mouth)
                print('is_big_mouth: ', is_big_mouth)
                tmp = cv2.imread(img)
                tmp = tmp[xy[1]:zz[1], xy[0]:zz[0]]
                cv2.imshow('img', tmp)
                if cv2.waitKey(1000) == ord('q'):
                    exit()

        random.shuffle(new_imgs_list)
        return new_imgs_list     

    def load_data(self, imgs_list, is_train=False):
        
        # if is_train:
        #     imgs_list = self.balance(imgs_list)

        for img in tqdm(imgs_list):

            label_path = img.rsplit('.', 1)[0] + '.pts'
            landmark = read_points(label_path)
            landmark = np.array(landmark)
            
            xy = np.min(landmark, axis=0).astype(np.int32) 
            zz = np.max(landmark, axis=0).astype(np.int32)
            wh = zz - xy + 1
            
            image = cv2.imread(img)

            # imageA = copy.deepcopy(image)
            # bbox = [float(np.min(landmark[:, 0])), float(np.min(landmark[:, 1])), float(np.max(landmark[:, 0])), float(np.max(landmark[:, 1]))]
            # imgA, landmarkA = augmentationCropImage(imageA, bbox, landmark, is_train)

            center = (xy + wh/2).astype(np.int32)
            boxsize = int(np.max(wh)*1.2)
            xy = center - boxsize//2
            x1, y1 = xy
            x2, y2 = xy + boxsize
            height, width, _ = image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            imgT = image[y1:y2, x1:x2]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            if imgT.shape[0] == 0 or imgT.shape[1] == 0:
                imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for x, y in (self.landmark+0.5).astype(np.int32):
                    cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
                cv2.imshow('0', imgTT)
                if cv2.waitKey(0) == 27:
                    exit()

            imgT = cv2.resize(imgT, (self.image_size, self.image_size))
            landmark = (landmark - xy)/boxsize
            assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
            assert (landmark <= 1).all(), str(landmark) + str([dx, dy])

            if DEBUG:
                # draw_landmark_point(imgA, landmarkA, 1)
                # cv2.imshow('imgA', imgA)
                draw_landmark_point(imgT, landmark, self.image_size)
                cv2.imshow('imgT', imgT)
                if cv2.waitKey(1000) == ord('q'):
                    return
            self.imgs.append(imgT)
            self.landmarks.append(landmark)

        if is_train:
            for i in range(self.repeat)
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))

                
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:,0] = 1 - landmark[:,0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

                
if __name__ == "__main__":
    root_path = '/home/shu/Work/landmark_dataset'
    imgs_list = get_images_list(root_path)

    random.shuffle(imgs_list)
    ratio = 0.95
    train_list = imgs_list[:int(ratio*len(imgs_list))]
    val_list = imgs_list[int(ratio*len(imgs_list)):]

    Img = ImageData()
    Img.load_data(train_list, True)


    






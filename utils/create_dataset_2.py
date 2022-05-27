import argparse
import os
import time
from loguru import logger
import pandas as pd
import cv2

import torch
from torch import tensor,float32,cat

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES, VOC_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
import numpy as np

from mmpose.apis import (inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

from Track.Tracker import Detection, Tracker
from Track.yolo_sort import Sort

from ActionsEstLoader import TSSTG

from PoseEstimateLoader import SPPE_FastPose

from fn import draw_single

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
save_path = './createData/animal/fuse_video/deer/deer_pose_lay.csv'   #

annot_file = './createData/animal/fuse_video/deer/deer_lay.csv'       #

video_folder = './createData/animal/fuse_video/deer/lay'              #

columns = [ 'video', 'frame', 
            'Left_eye_x', 'Left_eye_y', 'Left_eye_s',
            'Right_eye_x', 'Right_eye_y', 'Right_eye_s',
            'Nose_x', 'Nose_y', 'Nose_s',
            'Neck_x', 'Neck_y', 'Neck_s',
            'Root_of_Tail_x', 'Root_of_Tail_y', 'Root_of_Tail_s',
            'Left_Shoulder_x', 'Left_Shoulder_y', 'Left_Shoulder_s',
            'Left_Elbow_x', 'Left_Elbow_y', 'Left_Elbow_s',
            'Left_Front_Paw_x', 'Left_Front_Paw_y', 'Left_Front_Paw_s',
            'Right_Shoulder_x', 'Right_Shoulder_y', 'Right_Shoulder_s',
            'Rgiht_Elbow_x', 'Rgiht_Elbow_y', 'Rgiht_Elbow_s',
            'Right_Front_Paw_x', 'Right_Front_Paw_y', 'Right_Front_Paw_s',
            'Left_Hip_x', 'Left_Hip_y', 'Left_Hip_s',
            'Left_Knee_x', 'Left_Knee_y', 'Left_Knee_s',
            'Left_Back_Paw_x', 'Left_Back_Paw_y', 'Left_Back_Paw_s',
            'Right_Hip_x', 'Right_Hip_y', 'Right_Hip_s',
            'Right_Knee_x', 'Right_Knee_y', 'Right_Knee_s',
            'Right_Back_Paw_x', 'Right_Back_Paw_y', 'Right_Back_Paw_s',
            'label']


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy

annot = pd.read_csv(annot_file)
vid_list = annot['video'].unique()

def is_in_box(xy, box):
    if box[0] < xy[0] < box[2] and box[1] < xy[1] < box[3]:
        return True
    else:
        return False


def kpt2bbox(kpt, ex=20):
    """得到一个包括所有关节点的边界框
    kpt: 数组类型为 `(N, 2)`,
    ex: (int) 边界框边缘留白的像素大小,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default='/home/lyg/workspace/YOLOX/YOLOX_outputs/yolox_voc_s/all_best_ckpt.pth', type=str, help="ckpt for eval")
    parser.add_argument("-pose", "--pose_checkpoint", default='/home/lyg/workspace/AP-10K/ckpt/hrnet_w32_ap10k_256x256-18aac840_20211029.pth', type=str, help="pose ckpt path")
    parser.add_argument('-pose_config', default='/home/lyg/workspace/YOLOX_Det/pose/hrnet_w32_ap10k_256_256.py', help='Config file for pose')
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.4, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=8,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=4,
        help='Link thickness for visualization')

    parser.add_argument(
        '--de', default='cuda:0', help='Device used for inference')
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        ret_bbox = []
        np_bbox = bboxes.numpy()
        np_bbox = np_bbox.astype(int)
        np_score = scores.numpy()
        for i in range(len(np_bbox)):
            animal = {}
            info = np.concatenate((np_bbox[i], [np_score[i]]), axis=0)
            animal["bbox"] = info
            ret_bbox.append(animal)

        sort_det = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            cla = cls[i]
            box_conf = output[:,4][i]
            cls_conf = output[:,5][i]
            score = round(float(scores[i]), 4)
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            sort_det.append([x0, y0, x1, y1, box_conf, cls_conf, cla])
        dets = np.array(sort_det)
        dets = torch.from_numpy(dets).cuda()

        return ret_bbox



def imageflow_demo(predictor, args, video_list):
    # tracker init
    max_age = 10
    tracker = Tracker(max_age=max_age, n_init=3)

    # Fastpose
    inp_pose = (int(320), int(256))
    SPPE_model = SPPE_FastPose('resnet50', inp_pose[0], inp_pose[1], device='cuda')

    # action recognition base on skeleton
    action_pre = TSSTG()

    # animal pose predict:  refer to AP-10k
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.de.lower())
    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    for vid in video_list:
        print("Process on:{}".format(vid))
        df = pd.DataFrame(columns=columns)
        cur_row = 0
        frames_label = annot[annot['video'] == vid].reset_index(drop=True)

        cap = cv2.VideoCapture(os.path.join(video_folder, vid))
       
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
      
        frame_id = 0
        while True:
            ret_val, frame = cap.read()
            frame_id += 1
            if ret_val:
                cls_idx = int(frames_label[frames_label['frame'] == frame_id]['label'])
                outputs, img_info = predictor.inference(frame)
                ret_bbox= predictor.visual(outputs[0], img_info, predictor.confthre)  #  list  detection bbox
                if len(ret_bbox)> 0 and len(ret_bbox)<2:
                    print('current frame:{}'.format(frame_id))
                # test a single image, with a list of bboxes.
                    pose_results, returned_outputs = inference_top_down_pose_model(
                        pose_model,
                        frame,
                        ret_bbox,
                        bbox_thr=0.1,
                        format='xyxy',
                        dataset=dataset,
                        dataset_info=dataset_info,
                        return_heatmap=False,
                        outputs=None)  
                    # output_layer_names
                    #print(pose_results)

  
                    kpoints = pose_results[0]["keypoints"]

                    kpoints[:, 0] = kpoints[:, 0]/width
                    kpoints[:, 1] = kpoints[:, 1]/height

                    #print(kpoints)

                    row = [vid, frame_id, *kpoints.flatten().tolist(), cls_idx]
        
                    scr = kpoints[:, 2].mean()

                    df.loc[cur_row] = row
                    cur_row += 1

                    # show the results
                    vis_img = vis_pose_result(
                        pose_model,
                        frame,
                        pose_results,
                        dataset=dataset,
                        dataset_info=dataset_info,
                        kpt_score_thr=0.2,
                        radius=args.radius,
                        thickness=args.thickness,
                        show=False)
                
                    #cv2.imshow('frame', vis_img)

                    ch = cv2.waitKey(1)
                    if ch == 27 or ch == ord("q") or ch == ord("Q"):
                        break
                else:
                    continue
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        if os.path.exists(save_path):
            df.to_csv(save_path, mode='a', header=False, index=False)
        else:
            df.to_csv(save_path, mode='w', index=False)



def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()


    logger.info("loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")
    trt_file = None
    decoder = None
    predictor = Predictor(
        model, exp, VOC_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )

    imageflow_demo(predictor, args, vid_list)



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
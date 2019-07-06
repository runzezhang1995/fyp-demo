from __future__ import division
from util import *
import argparse

from darknet import Darknet
from preprocess import prep_image, inp_to_image

import pickle as pkl


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_path = 'E:/workplace/data/INRIAPerson/TestNew2/pos/'
det_path = 'E:/workplace/data/INRIAPerson/TestNew2/det2/'
colors = pkl.load(open("detection/pallete", "rb"))
color = (0, 0, 255)


def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = image_path, type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = det_path, type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "detection/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "detection/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

def get_model(args):
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    model.cuda()

    # Set the model in evaluation mode
    model.eval()
    return model, inp_dim

def get_bbox_of_image(ori_img, model, CUDA, input_dim = 416):
    image, orig_im, dim = prep_image(ori_img, input_dim)
    image = image.cuda()


    with torch.no_grad():
        prediction = model(Variable(image), CUDA)
    prediction = write_results(prediction, 0.5, 80, nms = True, nms_conf = 0.4)
    output = prediction

    scaling_factor = input_dim / dim[0]
    if scaling_factor > 1:
        scaling_factor = 1

    output[:, [1, 3]] -= (input_dim - scaling_factor * dim[0]) / 2
    output[:, [2, 4]] -= (input_dim - scaling_factor * dim[1]) / 2

    output[:, 1:5] /= scaling_factor

    box = []
    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, dim[0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, dim[1])

        x = output[i]

        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())


        cls = int(x[-1])


        if cls == 0:
            coor = {
                'c1': [c1[0].cpu().numpy().tolist(), c1[1].cpu().numpy().tolist()],
                'c2': [c2[0].cpu().numpy().tolist(), c2[1].cpu().numpy().tolist()],
            }
            size = (c2[1] - c1[1]) * (c2[0] - c1[0])
            if size > 100:
                box.append(coor)
                cv2.rectangle(orig_im, c1, c2, color, 2)

            # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            # c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        # cv2.rectangle(img, c1, c2,color, -1)
        # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    cv2.imwrite('E:/workplace/electron_wp/testApp/public/yolo.png', orig_im)

    return orig_im, box

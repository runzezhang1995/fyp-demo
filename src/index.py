from test import extract_image_feature_pcb, extract_image_feature_dense, prepare_model
from utils import get_opt
from evaluate_gpu import get_nearest_neighbors
from evaluate_rerank import get_nearest_neighbors_rerank
import scipy.io
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from detect import get_model, get_bbox_of_image, arg_parse
from cam import Camera
import _thread
import os



import numpy as np
app = Flask(__name__)
CORS(app)
cam = Camera(800, 600)
import os


model_pcb = None
model_dense = None
model_yolo = None



result_pcb = scipy.io.loadmat('reid/pytorch_result_PCB.mat')
gf_pcb = torch.FloatTensor(result_pcb['gallery_f'])
gf_pcb = gf_pcb.cuda()

gc_pcb = result_pcb['gallery_cam'][0]
gl_pcb = result_pcb['gallery_label'][0]
gp_pcb = result_pcb['gallery_path'][..., 0]
# g_g_dist = np.dot(gf_pcb, np.transpose(gf_pcb))

result_dense = scipy.io.loadmat('reid/pytorch_result_ft_net_dense.mat')
gf_dense = torch.FloatTensor(result_dense['gallery_f'])
gf_dense = gf_dense.cuda()

gc_dense = result_dense['gallery_cam'][0]
gl_dense = result_dense['gallery_label'][0]
gp_dense = result_dense['gallery_path'][..., 0]

CUDA_yolo = torch.cuda.is_available()


@app.route('/')
def hello_world():
    return jsonify({
        'success': True
    })
#
@app.route('/reid/start', methods=['POST'])
def start_reid_model():
    global model_pcb
    global model_dense
    if model_pcb == None:
        opt_reid = get_opt('pcb')
        model_pcb = prepare_model(opt_reid)
    if model_dense == None:
        opt_reid_dense = get_opt('dense')
        model_dense = prepare_model(opt_reid_dense)
    return jsonify({
        'success': True,
        'modelLoaded': True
    })

@app.route('/reid/stop', methods=['POST'])
def stop_reid_model():
    global model_pcb
    global model_dense
    del model_pcb
    del model_dense
    model_pcb = None
    model_dense = None

    torch.cuda.empty_cache()
    return jsonify({
        'success': True,
        'modelLoaded': False
    })
@app.route('/reid/', methods=['GET'])
def get_reid_model_loaded():
    return jsonify({
        'success': 'true',
        'modelLoaded': ((model_pcb != None) and (model_dense != None))
    })


@app.route('/reid/recognize', methods=['POST'])
def get_reid_result():
    global model_pcb
    global model_dense

    if (model_pcb == None) or (model_dense == None):
        start_reid_model()

    root_path = 'E:/workplace/python_wp/data/market/pytorch/query/'


    img_path = root_path + request.form['image_path']
    feature_pcb = extract_image_feature_pcb(model_pcb, img_path)
    results_pcb = get_nearest_neighbors(feature_pcb, gf_pcb, gl_pcb, gc_pcb, gp_pcb)
    r_pcb = list(map(lambda result: result.replace('../../../data/market/pytorch\\gallery\\', '').replace('\\', '/').strip(), results_pcb))

    feature_dense = extract_image_feature_dense(model_dense, img_path)
    results_dense = get_nearest_neighbors(feature_dense, gf_dense, gl_dense, gc_dense, gp_dense)
    r_dense = list(map(lambda result: result.replace('\\', '/').strip().replace('../data/market/pytorch/gallery/', ''), results_dense))

    return jsonify({
        'success':True,
        'results_pcb': r_pcb,
        'results_dense': r_dense
    })
    # return {
    #     'results_pcb': r_pcb,
    #     'results_dense': r_dense
    # }



@app.route('/yolo/start', methods=['POST'])
def sym():
    if start_yolo_model():
        return jsonify({
            'success': True,
            'modelLoaded': True
        })
    else:
        return jsonify({
            'success': False
        })

def start_yolo_model():
    try:
        global model_yolo
        if model_yolo == None:
            arg_yolo = arg_parse()
            model_yolo, _ = get_model(arg_yolo)
        return True
    except:
        print('fail to start yolo model')
        return False



@app.route('/yolo/stop', methods=['POST'])
def stop_yolo_model():
    try:
        global model_yolo
        del model_yolo
        model_yolo = None
        torch.cuda.empty_cache()
        return jsonify({
            'success': True,
            'modelLoaded': False
        })
    except:
        return jsonify({
            'success': False
        })

@app.route('/yolo', methods=['GET'])
def ym():
    return jsonify({
        'success': 'true',
        'modelLoaded': (model_yolo != None)
    })


@app.route('/yolo/detectService/start', methods=['POST'])
def start_cam_service():
    global cam
    if not cam:
        cam = Camera(800, 600)
    if cam and (not cam.operating):
        try:
            _thread.start_new_thread(cam.start, (get_yolo_result, ))
            # _thread.start_new_thread(hello, (1234, ))
            return jsonify({
                'success': True,
                'operating': True
            })
        except:
            print("Error: start thread error ")
            return jsonify({
                'success': False,
                'error': 'START_THREAD_ERROR'
            })
    else:
        return jsonify({
            'success': True,
            'operating': True
        })

def get_yolo_result(ori_img):
    global model_yolo
    if model_yolo == None:
        start_yolo_model()

    # root_path = 'E:/workplace/data/INRIAPerson/TestNew2/pos/'
    # img_path = root_path + request.form['image_path']
    #
    img, boxes = get_bbox_of_image(ori_img, model_yolo, CUDA_yolo)

    return {
        'success': True,
        'boxes': boxes,
    }, img




@app.route('/yolo/detectService/stop', methods=['POST'])
def stop_cam_service():
    if cam:
        try:
            cam.stop()
            # _thread.start_new_thread(hello, (1234, ))

            return jsonify({
                'success': True,
                'operating': False

            })
        except:
            print("Error: start thread error ")
            return jsonify({
                'success': False,
                'error': 'START_THREAD_ERROR'
            })


@app.route('/yolo/detectService/update', methods=['POST'])
def cam_update_img():
    global cam
    if not cam:
        cam = Camera(800, 600)
    try:
        cam.start_once(get_yolo_result)
        # _thread.start_new_thread(hello, (1234, ))
        return jsonify({
            'success': True,
        })
    except:
        print("Error: start thread error ")
        return jsonify({
            'success': False,
        })





@app.route('/yolo/detectService/', methods=['GET'])
def get_cam_status():
    return jsonify({
        'success':True,
        'operating': cam.operating
    })


#
# img_folder = 'E:/workplace/electron_wp/testApp/public/market/query_copy/'
# query_list = os.listdir(img_folder)
#
# available_result = []
#
# with open('test.txt', 'r') as f:
#     for s in f:
#         available_result.append(s.strip())
#
#
# for query in query_list:
#     identity = query
#     query_path = os.path.join(img_folder, query)
#     img_list = os.listdir(query_path)
#     for img in img_list:
#         img_id = query + '/' + img
#         if img_id not in available_result:
#             path = img_folder + img_id
#             os.remove(path)
#




# for query in query_list:
#     identity = query
#     query_path = os.path.join(img_folder, query)
#     img_list = os.listdir(query_path)
#     for img in img_list:
#         img_id = query + '/' + img
#
#         results = get_reid_result(img_id)
#         result_pcb = results['results_pcb']
#         result_dense = results['results_dense']
#         score_pcb = 0
#         score_dense = 0
#
#
#         for i in range(0, 10):
#             pcb = result_pcb[i]
#             iid = pcb.split('/')[0]
#
#
#             if iid == identity:
#                 score_pcb += 1
#
#
#             dense = result_dense[i]
#             iid = dense.split('/')[0]
#             if iid == identity:
#                 score_dense += 1
#
#         if score_pcb - score_dense > 2:
#             available_result.append(img_id)
#
# with open('test.txt', 'w') as f:
#     for s in available_result:
#         f.write(str(s) + '\n')
# print(available_result)
#
if __name__ == '__main__':
    app.run()





#
# #example
# img_path = 'E:\\workplace\\python_wp\\data\\market\\pytorch\\query\\0009\\0009_c1s1_000376_00.jpg'
# get_reid_result(img_path)
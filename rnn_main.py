# main
import tensorflow as tf

# mine
from models.evalnet_head import EvalNet
from models.rnnpp_head   import PolygonModel
from models.ggnn_head    import GGNNPolygonModel
from models.poly_utils   import draw_edge
import models.utils as utils

# draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# others
import glob
import numpy as np
import skimage.io as io
import tqdm
import json
from os import path

# dependency files
rnnpp_checkpoint = "mates/rnnpp/polygonplusplus.ckpt"
eval_checkpoint  = "mates/evalnet/evalnet.ckpt"
ggnn_checkpoint  = "mates/ggnn/ggnn.ckpt"
rnnpp_meta_path  = "mates/rnnpp/polygonplusplus.ckpt.meta"
ggnn_meta_path   = "mates/ggnn/ggnn.ckpt.meta"
input_dir  = "input/"
output_dir = "output/"

tf.logging.set_verbosity(tf.logging.INFO)

_MODEL_INDEX = {'RNNpp':PolygonModel, 'GGNN':GGNNPolygonModel}


def build_evalnet(my_eval_graph):
    """
    :param my_eval_graph:  a graph to load EvalNet
    :return:               list[0] is evaluator; list[1] is tf session
    """
    tf.logging.info("Building EvalNet...")
    with my_eval_graph.as_default():
        with tf.variable_scope('discriminator_network'):
            evaluator = EvalNet(1)
            evaluator.build_graph()
        saver = tf.train.Saver()

        eval_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                               graph=my_eval_graph)
        saver.restore(eval_sess, eval_checkpoint)

    return [evaluator, eval_sess]


def build_model(graph, eval, mate, disc="RNNpp"):
    """
    :param my_graph: model's datastream graph
    :eval:           evalutor and its sess
    :mate            trained model
    :param disc:     target model
    :return:
    """
    # 描述所创建的模型
    tf.logging.info('Building ' + disc + ' model...')
    # 从保存的已训练的模型中生成模型 并加载进数据流图
    model = _MODEL_INDEX[disc](mate["metagraph"], graph)

    model.register_eval_fn(lambda input: eval[0].do_test(eval[1], input))

    my_sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True),
                         graph=graph)
    model.saver.restore(my_sess, mate["checkpoint"])
    return [model, my_sess]


def main(_):
    eval_graph = tf.Graph()
    ploy_graph = tf.Graph()
    ggnn_graph = tf.Graph()

    ploy_mate = {"metagraph": rnnpp_meta_path, "checkpoint": rnnpp_checkpoint}
    ggnn_mate = {"metagraph": ggnn_meta_path, "checkpoint": ggnn_checkpoint}

    evals      = build_evalnet(eval_graph)
    ploy_model = build_model(graph=ploy_graph, eval=evals, mate=ploy_mate, disc="RNNpp")
    ggnn_model = build_model(graph=ggnn_graph, eval=evals, mate=ggnn_mate, disc="GGNN")

    # make sure the output folder and get images' path
    if not path.isdir(output_dir):
        tf.gfile.MakeDirs(output_dir)
    crops_path = glob.glob(path.join(input_dir, '*.png'))

    for crop_path in tqdm.tqdm(crops_path):
        # loading images as 3D array
        image_np = io.imread(crop_path)
        # expanding arrays to 4D
        image_np = np.expand_dims(image_np, axis=0)

        tf.logging.info('Testing PolygonRNN++...')
        # train
        preds = [ploy_model[0].do_test(ploy_model[1], image_np, top_k) for top_k in range(5)]
        # sort predictions based on the eval score and pick the best
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)[0]

        tf.logging.info('Testing GGNNPolygon...')
        polys = np.copy(preds['polys'][0])
        feature_indexs, poly, mask = utils.preprocess_ggnn_input(polys)
        # train
        preds_gnn = ggnn_model[0].do_test(ggnn_model[1], image_np, feature_indexs, poly, mask)
        output = {'polys': preds['polys'], 'polys_ggnn': preds_gnn['polys_ggnn']}
        # saving output of one input
        json_path = save_to_json(crop_path, output)
        img_path  = save_to_img(json_path)
        print('saving '+json_path+' and '+img_path+' success!')

def save_to_json(crop_name, predictions_dict):
    output_dict = {'img_source': crop_name, 'polys': predictions_dict['polys'][0].tolist()}
    output_dict['polys_ggnn'] = predictions_dict['polys_ggnn'][0].tolist()
    # get final path
    json_name = path.basename(crop_name).split('.')[0] + '.json'
    json_path = path.join(output_dir, json_name)
    # write ploys of both rnnpp and ggnn
    json.dump(output_dict, open(json_path, 'w'), indent=4)
    return json_path


def save_to_img(file_path):
    print('draw begin...')

    fig, axes = plt.subplots(1, 2, num=0, figsize=(12, 6))
    axes = np.array(axes).flatten()
    pred = json.load(open(file_path, 'r'))
    file_name = file_path.split('/')[-1].split('.')[0]
    # get image and ploys
    img = io.imread(pred['img_source'])
    rnnpp_ploy = np.array(pred['polys'])
    ggnn_ploy = np.array(pred['polys_ggnn'])
    # draw
    draw_edge(axes[0], img, rnnpp_ploy, 'rnnpp')
    draw_edge(axes[1], img, ggnn_ploy, 'ggnn')
    # save
    img_path = path.join(output_dir, file_name) + '.png'
    fig.savefig(img_path)
    [ax.cla() for ax in axes]
    return img_path


if __name__ == '__main__':
    tf.app.run(main)





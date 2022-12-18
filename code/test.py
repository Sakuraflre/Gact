import os
import argparse
import torch
from networks.vnet_sdf import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/BRATS', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='BRATS', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--detail', type=int,  default=0, help='detail')
parser.add_argument('--test_num', type=int,  default=20, help='test')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./{}".format(FLAGS.model)
test_save_path = os.path.join(snapshot_path, "test/")




num_classes = 2




def test_calculate_metric(save_mode_path, root_path, save_test_path, test_num):
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
    #print(root_path + '/test.list')
    with open(root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    image_list = image_list[:test_num]
    net = VNet(n_channels=1, n_classes=num_classes-1, normalization='batchnorm', has_dropout=False).cuda()
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=save_test_path, metric_detail=FLAGS.detail)

    return avg_metric

def test_calculate_metric_memory(net,train_data_path, test_save_path, test_num):

    net.eval()
    with open(train_data_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    image_list = image_list[:test_num]

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path, metric_detail=FLAGS.detail)

    return avg_metric
if __name__ == '__main__':
    save_mode_path = os.path.join(
        snapshot_path, 'best_model.pth')
    metric = test_calculate_metric(save_mode_path, FLAGS.root_path, test_save_path,parser.test_num )  # 6000
    print(metric)
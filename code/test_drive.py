import os
import argparse
import torch
from networks.mynet import MyNet
from test_util_drive import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/DRIVE/', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='DRIVE/DTC_with_consis_weight', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')


FLAGS = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/{}".format(FLAGS.model + "_16labels_beta_0.3")

num_classes = 2

test_save_path = os.path.join(snapshot_path, "test/")
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + 'test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "converted_data/" + item.replace('\n', '') + ".h5" for item in
              image_list]


def test_calculate_metric():
    net = MyNet(n_channels=3, n_classes=num_classes-1,
               normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(
        snapshot_path, 'iter_6000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=(112, 112, 3), stride_xy=18,
                               save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000
    print(metric)

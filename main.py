import sys
sys.path.extend([r'D:\dl_project\dl_project_cnn\DataInput', r'D:\dl_project\dl_project_cnn\Models'\
                    , r'D:\dl_project\dl_project_cnn\Trainer', r'D:\dl_project\dl_project_cnn\Utils'])
sys.path.extend([r'F:\dl_project_cnn\DataInput', r'F:\dl_project_cnn\Models'\
                    , r'F:\dl_project_cnn\Trainer', r'F:\dl_project_cnn\Utils'])

from Trainer.train import *
from Trainer.test import *
from Models.AD_RoadNet import AD_RoadNet
from Models.DeepMTLNet import DeepMTLNet

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()
    

if __name__ == '__main__':
    config_path = r'\config.ini'
    net = AD_RoadNet(num_classes=1, norm_layer='BN')
    #net = DeepMTLNet(num_classes=1)
    args = get_args()
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location = 'cuda:0')
            )
        logging.info(f'Model is loaded from {args.load}')

    mode = 'test'

    if mode == 'train':
        train(net=net, config_path=config_path)
    if mode == 'test':
        loss, res_dict = test(net=net, config_path=config_path)
        # loss, res_dict = dilated_test(net=net, config_path=config_path, dilated_size=512, size=512)
        print(f'loss = {loss}')
        print('----------------')
        print(f'res_dict={res_dict}')

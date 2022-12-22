import argparse
from trainer import Trainer
import torch
import os
from enum import Enum


class MODE(Enum):
    Train = 0
    Test = 1
    Global = 2


def getargs():
    parser = argparse.ArgumentParser(description='self.Args for training')
    parser.add_argument('-case', type=str, help='cylinder/aero/plate/Font')
    parser.add_argument('-data_dir', type=str, help='/data_dir/outputs_train(or valid, test)/. contains the datas')
    parser.add_argument('-dump_dir', type=str, help='/dump_dir/*method/ to store ckpts, logs, trajs etc')
    parser.add_argument('-space_dim', type=int, help='spatial dimension: 2 or 3')
    parser.add_argument('-mode', type=int, default=0, help='0) train 1) test 2) global roll')

    parser.add_argument('-n_train', type=int, default=1, help='train seq number')
    parser.add_argument('-n_valid', type=int, default=1, help='valid seq number')
    parser.add_argument('-n_test', type=int, default=1, help='test seq number')
    parser.add_argument('-time_len', type=int, default=2, help='time seq length')

    parser.add_argument('-noise_level', nargs='+', type=float, help='shuffle noise level vec')
    parser.add_argument('-noise_gamma', type=float, default=1, help='1~0, 1: no effect on output;0: output has the same noise level of noise')

    parser.add_argument('-recal_mesh', type=bool, default=False, help='force recalculate multi level mesh')
    parser.add_argument('-consist_mesh', type=int, help='same mesh for every snapshots?')
    parser.add_argument('-multi_mesh_layer', type=int, help='how many extra layer (in depth)?')
    parser.add_argument('-mp_time', type=int, default=1, help='how many time of MP')
    parser.add_argument('-hidden_dim', type=int, default=128, help='hidden dim of MLP')
    parser.add_argument('-hidden_depth', type=int, default=2, help='hidden depth of MLP')

    parser.add_argument('-num_epochs', type=int, default=20, help='epochs')
    parser.add_argument('-batch', type=int, default=1, help='batch size')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.93, help='decary rate')

    parser.add_argument('-restart_epoch', type=int, default=-1, help='restart checkpoint epoch')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = getargs()
    args.consist_mesh = bool(args.consist_mesh)
    args.dump_dir = os.path.join(args.dump_dir, 'ours')
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(args, device)
    if MODE(args.mode) == MODE.Train:
        print('Train')
        trainer.train()
    elif MODE(args.mode) == MODE.Test:
        print('Local test')
        trainer.test()
    elif MODE(args.mode) == MODE.Global:
        print('Global Rollout')
        trainer.rollout()
    else:
        raise NotImplementedError("Unknown mode")

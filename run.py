#-----------------------------------------------
#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time:2020/8/3 8:30
#-----------------------------------------------
import argparse, yaml, wandb
from opentqa.models.model_loader import CfgLoader
from utils.engine import Engine


def parse_input_args():
    '''
    parse the input
    :return:
    '''
    args = argparse.ArgumentParser(description='model arguments')
    args.add_argument('--ckpt_v', dest='version',
                      help='checkpoint version',
                      type=str)

    args.add_argument('--ckpt_e', dest='ckpt_epoch',
                      help='checkpoint epoch')

    args.add_argument('--dataset_use', dest='dataset_use',
                      choices=['ndqa', 'dqa'],
                      help='{ndqa: textual question answering,'
                           ' dqa: diagram question answering}',
                      type=str,
                      default='dqa',
                      required=True
                      )

    args.add_argument('--model', dest='model',
                      choices=['hmfn', 'xtqa', 'ban'],
                      type=str,
                      default='xtqa',
                      required=True
                      )

    args.add_argument('--run_mode', dest='run_mode',
                      choices=['train', 'test'],
                      type=str,
                      default='train',
                      required=True
                      )

    args = args.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_input_args()
    cfg_file = 'configs/{}/{}.yml'.format(args.dataset_use, args.model)
    with open(cfg_file, 'r') as f:
        yml_dict = yaml.load(stream=f, Loader=yaml.BaseLoader)

    cfgs = CfgLoader(args.dataset_use, args.model).load()
    args_dict = cfgs.parse_to_dict(args)
    args_dict = {**args_dict, **yml_dict}
    cfgs.add_attr(args_dict)
    cfgs.proc()

    print('Configurations of Networks:')
    print(cfgs)
    engine = Engine(cfgs=cfgs)
    engine.load_method()



import argparse
import os
import sys
import logging
from yacs.config import CfgNode as CfgNode

# Global config object
_C = CfgNode()

args = _C


logger = logging.getLogger()
logger.setLevel(logging.INFO) #日志等级为INFO


# dir config
_C.data_dir = "glue_data/MRPC"
_C.output_dir = ""

# basic config
_C.no_cuda = False
_C.local_rank = -1
_C.n_gpu = 1
_C.save_steps = 51
_C.seed = 42
_C.evaluate_during_training = True
_C.per_gpu_train_batch_size = 32
_C.per_gpu_eval_batch_size = 32
_C.logging_step = 50

# train config



# evolution config





def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.output_dir, 'logger')
    with open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)


def load_cfg_fom_args(description="Config file options."):
    """Load config from command line arguments and set any specified options.
       How to use: python xx.py --cfg path_to_your_config.cfg test1 0 test2 True
       opts will return a list with ['test1', '0', 'test2', 'True'], yacs will compile to corresponding values
    """
    parser = argparse.ArgumentParser(description=description)
    help_s = "Config file location"
    parser.add_argument("--cfg", dest="cfg_file",
                        help=help_s, required=True, type=str)
    help_s = "See pycls/core/config.py for all options"
    parser.add_argument("opts", help=help_s, default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    _C.merge_from_file(args.cfg_file)
    _C.merge_from_list(args.opts)

def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('datagenerator')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger
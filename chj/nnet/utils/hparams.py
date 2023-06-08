from easydict import EasyDict as edict
from chj.base.obj import set_obj_attr, set_obj_by_str, set_obj_by_mp

def HParams(**args): return edict(args)

def hparams_debug_string(hparams):
    values = hparams
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(hparams.keys()) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)

create_hparams = set_obj_by_str


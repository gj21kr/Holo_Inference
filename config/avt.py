config = {
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 1,
    "CLASSES"       : {
            1:"Aorta"
        },
    "interp_mode"   : "trilinear_trilinear",
    "MODEL_NAME"    : "nnunet",
    "MODEL_VERSION" : "avt-13",
    "SPACING"       : [None, None, None], 
    "INPUT_SHAPE"   : [96,96,96], 
    "DROPOUT"       : 0.,
    "CONTRAST"      : [-1000,2000],
    "INT_NORM"      : 'znorm',
    "BATCH_SIZE"    : 1,
    "ACTIVATION"    : 'sigmoid',
    "MODE"          : 'None',
    "WEIGHTS"       : [1.0],
    "MODEL_CHANNEL_IN"  : 32,
    "DEEP_SUPERVISION"   : True,
    "THRESHOLD"     : 0.5,
    "ARGMAX"        : False,
    "FLIP_XYZ"      : [False, False, False],
    "TRANSPOSE"     : [(2,1,0), (2,1,0)],
    "SAVE_CT"       : False,
    "SAVE_MERGE"    : True,
}

from transforms.ImageProcessing import *
# post processing 
transform = [
    # RemoveSamllObjects(min_size=5000),
]
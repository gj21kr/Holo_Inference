config = {
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 2,
    "CLASSES"       : {
            1: "Brain",
            2: "CT_Cerebral_Artery"
        },
    "interp_mode"   : "trilinear_trilinear",
    "MODEL_NAME"    : "nnunet",
    "MODEL_VERSION" : "cmha",
    "SPACING"       : [None, None, None], 
    "INPUT_SHAPE"   : [128,128,128], 
    "DROPOUT"       : 0.,
	"WORKERS"       : 10,
    "CONTRAST"      : [-100,700],
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
    "SAVE_MERGE"    : False,
	## dilateddenseunet
	"GROWTH_RATE"	: 32,
    "NUM_RES_UNITS"	: 4,
    "CHANNEL_LIST"	: (32,),
}

from transforms.ImageProcessing import *
# post processing 
transform = [
    # RemoveSamllObjects(min_size=5000),
]
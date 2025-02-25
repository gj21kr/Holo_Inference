config = {
    "CHANNEL_IN"    : 1,
    "CHANNEL_OUT"   : 1,
    "CLASSES"       : {
            1:"CT_Cerebral_Artery"
        },
    "interp_mode"   : "trilinear_trilinear",
    "MODEL_NAME"    : "dilateddenseunet",
    # "MODEL_VERSION" : "cta_ca_96_-100_500_dilateddenseunet",
    "MODEL_VERSION" : "cta_ca_128_-10_280_dilateddenseunet",
    "SPACING"       : [None, None, None], 
    "INPUT_SHAPE"   : [96,96,96], 
    "DROPOUT"       : 0.,
	"WORKERS"       : 10,
    "CONTRAST"      : [-100,500],
    "INT_NORM"      : 'znorm',
    "BATCH_SIZE"    : 12,
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
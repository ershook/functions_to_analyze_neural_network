

import numpy as np
import h5py 
import numpy as np
import sklearn
import h5py
import sklearn.metrics
import scipy.stats
import os 
import sys
sys.path.append('/om/user/ershook/model_response_orig/')
from generateCochleagrams import GenerateCochleagrams

import pickle
import tensorflow as tf

import scipy.io.wavfile

import matplotlib as plt 
plt.rcParams['axes.color_cycle'] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

sys.path.append('/om2/user/ershook/NetworkAnalysesClass/base')
from NetworkAnalysisBase import NetworkAnalysisBase

def main():
    tf.reset_default_graph()
    sys.path.append('/om/user/ershook/AlexCNNPaperProject/LayerDecoding/WordTrainedDecoding')
    from DecodeWORDTrainedNetwork import *

    #Note: compared the results to /om/user/ershook/AlexCNNPaperProject/JupyterNotebooks/DecemberWordGenreSpeakerSummaryPlots.ipynb
    #Everything checks out! 

    models =[[WORD2016PaperNetworkDecode_conv1_1eN4_W(),  1 ],              
            [WORD2016PaperNetworkDecode_pool1_1eN4_W(),  0  ],
            [WORD2016PaperNetworkDecode_lrnorm1_1eN4_W(),0  ],
            [WORD2016PaperNetworkDecode_conv2_1eN5_W(),  8  ],
            [WORD2016PaperNetworkDecode_pool2_1eN5_W(), 11  ], 
            [WORD2016PaperNetworkDecode_lrnorm2_1eN5_W(),8  ],
            [WORD2016PaperNetworkDecode_conv3_1eN5_W(),   8 ], 
            [WORD2016PaperNetworkDecode_conv4_1eN5_W(),  11 ], 
            [WORD2016PaperNetworkDecode_conv4_1eN5_G(),  8  ],  
            [WORD2016PaperNetworkDecode_conv5_1eN5_G(),  6  ],
            [WORD2016PaperNetworkDecode_conv5_1eN5_W(),  19 ],
            [WORD2016PaperNetworkDecode_pool5_1eN5_W(),  91 ],
            [WORD2016PaperNetworkDecode_fc6_1eN4_W(),    91 ],
            [WORD2016PaperNetworkDecode_pool5_1eN5_G(), 82  ]]

    i = int(sys.argv[1])

    model = models[i]

    stimuli_path = '/home/alexkell/.skdata/PsychophysicsWord2017W_999c6fc475be1e82e114ab9865aa5459e4fd329d/cache/image_cache_429140774cdaa5c2ffd6e9119890223d9b177953_None_hdf5/data.raw'
    meta = np.load('/home/alexkell/.skdata/PsychophysicsWord2017W_999c6fc475be1e82e114ab9865aa5459e4fd329d/__META.npy')
    meta_ = np.concatenate(([meta['bg_snr']], [meta['word']]), axis = 0).T

    networkAnalysisObject = NetworkAnalysisBase(model, stimuli_path, meta_)
    logits = networkAnalysisObject.getLogits()
    networkAnalysisObject.getNetworkPerformance()

if __name__ == '__main__':
    main()
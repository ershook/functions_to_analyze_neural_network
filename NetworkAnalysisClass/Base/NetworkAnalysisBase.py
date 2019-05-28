
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




class NetworkAnalysisBase(object):
    
    #Note: from now on should name cochleagram folders stimulus_type_batch_number
    #Note: assumes only two models genre and word -- in future could extend to support arbitray architecture
    # ie. allow num_units_per_layer to be passed in (should be only necessary change)
    
    def __init__(self, model, stimuli_path, meta, stimulus_files = False, stimulus_name = False):
        #Need to add meta as an input
        
        
        self.base_dir = '/om2/user/ershook/NetworkAnalysesClass/data/'
    
        self.stimuli_path = stimuli_path
        
        self.model = model
        self.meta = meta
        
        #This is an annoying work around for interfacing with CNN.py
        
        if type(model)  == list:
            self.model = model[0]
            self.checkpoint = model[1]
            self.graph, self.tensors,self.session = self.model.getGraph(int(self.checkpoint))
            self.model_name = self.model.__class__.__name__
        
        else:
            self.session = self.model.session
            self.graph = self.model.graph
            self.tensors = self.model.tensors
            self.model_name = self.model.__class__.__name__
        
        if stimulus_files:
            # Cochleagrams have a unique format so just pass in a list of paths to the hdf5s
            # For example psychophysics2018 .hdf5 are indexed by subject number and run no
            # To get cochs in right order need to specify stimulus files ~outside~ of this class.
            self.stimulus_files = stimulus_files
            self.stimulus_name = stimulus_name
            
        else:
            if '/data.raw' in stimuli_path:
                # There is only a single hdf5 with cochleagrams
                self.stimulus_files = [stimuli_path]
                self.stimulus_name = stimuli_path.split('_999c6fc475be1e82e114ab9865aa5459e4fd329d')[0].split('/')[-1]
                
            else:
                # There are multiple cochleagram files
                # Most likely the cochleagram were generated in batches
                self.stimulus_files = self.getStimulusFiles(stimuli_path)
                self.stimulus_name = self.stimuli_path.split('/')[-2]
    
        
        self.class_name = self.model_name + '_' + self.stimulus_name
        
        self.class_dir = self.getClassDir() #This is where all data associated with this model/cochleagrams will be saved
        #If in future need to write somewhere other than /om2 set up a symbolic link 
        
        self.unit_activations_dir = self.getUnitActivationsDir()
        self.unit_activations_hdf5_path = os.path.join(self.unit_activations_dir, self.class_name+'_unit_activations.hdf5')
        self.number_of_units_per_layer = self.getNumberUnitsPerLayer()
        self.number_of_cochleagrams = self.getNumberOfCochleagrams()

    def getClassDir(self):
        d = os.path.join(self.base_dir, self.class_name)
        if not os.path.exists(d):
            os.makedirs(d)
        return d
    
    def getUnitActivationsDir(self):
        d = os.path.join(self.class_dir, 'unit_activations')
        if not os.path.exists(d):
            os.makedirs(d)
        return d
    
    def getNumberUnitsPerLayer(self):
        #Determine the number of units per layer (probably a better way but this is the only way I know how) 
        
        number_of_units_per_layer = {}
        dummy_coch = np.zeros((1,256*256)) #Assumes shape of cochleagram is 256x256
        
        for layer in self.model.layer_params.keys():
            if 'keep_prob' in self.tensors:
                activations = self.session.run(self.tensors[layer], feed_dict={self.tensors['x']: dummy_coch, self.tensors['y_label']: [0]*1, self.tensors['keep_prob']:1})
            else:
                activations = self.session.run(self.tensors[layer], feed_dict={self.tensors['x']: dummy_coch, self.tensors['y_label']: [0]*1})
            activations = np.squeeze(activations)
            number_of_units_per_layer[layer] = np.shape(activations)
            
        return number_of_units_per_layer
    
    def getNumberOfCochleagrams(self):
        #How many total cochleagrams are there across all hdf5s in the stimulus_files
        #Need this for writing hdf5 of all unit activations
        num_cochs = 0 
        for hdf5_file in self.stimulus_files:
            with h5py.File(hdf5_file,'r') as f_in:
                num_cochs += len(f_in['data'])
        return num_cochs


    def getStimulusFiles(self, stimuli_path):
        # There are multiple cochleagram files
        # Most likely the cochleagram were generated in batches
        # Get the path to each hdf5 and sort in batch order 
        stimulus_files = []
        folders = os.listdir(stimuli_path)
        for folder in folders:
            s = GenerateCochleagrams(stimuli_path+folder+'/')
            stimulus_files.append(s.getCochleagramPath())

        #Need to sort cochleagram paths by batch_no
        sorted_stimulus_files = []
        start_name = stimulus_files[0].split('batch')[0]+'batch_'
        end_name = '_'+('_').join(stimulus_files[0].split('batch')[1].split('_')[2:])
        for ii in range(len(stimulus_files)):
            sorted_stimulus_files.append(start_name + str(ii) + end_name)

        return sorted_stimulus_files
    
    def _writeHDF5UnitActivations(self):
        #Write an HDF5 of unit activations to all cochleagrams in stimulus_files
        #But first checks that file doesn't already exist (since getting activations is computationally expensive)
        
        if not os.path.exists(self.unit_activations_hdf5_path):
        
            with h5py.File(self.unit_activations_hdf5_path, 'x') as f_out:
                    #Create dataset store pointer to dataset object in dictionary
                    dataset_dict = {}
                    for layer in self.number_of_units_per_layer.keys():
                        dim = (self.number_of_cochleagrams,) + self.number_of_units_per_layer[layer]
                        dataset_dict[layer] = f_out.create_dataset(layer, dim , dtype='float32')

                    current_index = 0

                    for _, path in enumerate(self.stimulus_files):
                        print 'Getting activations for hdf5 file number '+ str(_) + ' out of '+ str(len(self.stimulus_files))
                        with h5py.File(path, 'r') as f_in:
                            for ind in range(len(f_in['data'])):
                                with self.graph.as_default() as g:
                            
                                    batch = f_in['data'][ind:ind+1,0:65536]
                                    if 'keep_prob' in self.tensors:
                                            measures = self.session.run(self.tensors, feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1, self.tensors['keep_prob']:1})
                                    else:
                                            measures = self.session.run(self.tensors, feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1})
                                    
                                    for layer in self.number_of_units_per_layer.keys():
                                        dataset_dict[layer][current_index] =  np.array(np.squeeze(measures[layer]))
                                current_index += 1
                               
                                if current_index %1000 ==0:
                                        print 'current_index: ' + str(current_index)
        else:
            print "Note: unit activations hdf5 for this model and set of cochleagrams already exists."
                                
    
    def getUnitActivations(self):
        self._writeHDF5UnitActivations()
        return self.unit_activations_hdf5_path
    
    
    def getNetworkPerformance(self): 
        self.correct = self.getCorrect()
        self.logits = self.getLogits()
        self.is_correct = (self.correct == self.logits)
        print 'overall performance is ' + str(np.sum(self.is_correct)/float(len(self.logits)) * 100) +'%'
        self.raw_data_by_cond, self.perf_dict = self.getPerformanceByCondition()
    
    def getCorrect(self): 
        if self.model_name == 'WordGenreNetwork_WORDBranch':
            #ie. the model was trained on the combined dataset not Jenelle's dataset
            word_key = np.load('/om/user/ershook/old_cnn_project_stuff/words.npy')
            correct = []
            for word in self.meta[:,1]:
                correct.append(list(word_key).index(word)) 
        else: 
            # the path to key for Jenelle's dataset so when you are evaluating classifiers
            label_dictionary = '/om/user/ershook/old_cnn_project_stuff/psychophysicsWordResults/dict_word_to_label.save'
            dict_word_to_label = pickle.load( open( label_dictionary, "rb" ) )
            
            correct = []
            for word in self.meta[:,1]:
                correct.append(dict_word_to_label[word]-1) #-1 because 1 indexed not zero...
            
        return np.array(correct)
    
    def getLogits(self, remove_null = False):
        if not os.path.exists(self.unit_activations_hdf5_path):
            self._writeHDF5UnitActivations()
        
        with h5py.File(self.unit_activations_hdf5_path, 'r') as f_in:
            activations = np.array(f_in['fc_top'])
        
        logits = []
        if remove_null: 
            for ii in range(len(activations)):
                row = list(np.argsort(activations[ii,:])) ###sort activations
                ind_588 = list(row).index(588) ### remove null
                row.pop(ind_588)
                ind_242 = list(row).index(242) ### remove null
                row.pop(ind_242)
                logits.append(row[::-1][0]) ### take top activation after removing null
        else:
            logits = np.argmax(activations, axis = 1)

        return np.array(logits)
    
    def getPerformanceByCondition(self):
        perf_dict = {}
        raw_data_by_cond = {}
        print
        print "Performance by condition: "
        for cond in np.unique(self.meta[:,0]):
            
            indices = np.where(self.meta[:,0] == cond)
            raw_data_by_cond[cond] = self.is_correct[indices]
            perf_dict[cond] = np.mean(self.is_correct[indices])
            
            print cond, perf_dict[cond]
        return raw_data_by_cond, perf_dict
            
            

    def consolidateIndividualHDF5s(self):
        pass
                
        
    def makeLinePlot(self, condition_order): 
        pass
    
    def KellEtAlFigure2BNet7TF(self):
        # Code to generate this figure from /om/user/ershook/AlexCNNPaperProject/JupyterNotebooks/plotWordPsychophysics.ipynb
        backgrounds = ['Babble2Spkr', 'SpeakerShapedNoise', 'Music', 'AudScene', 'Babble8Spkr']
        snrs=['neg9db','neg6db','neg3db','0db', '3db']
        for bg in backgrounds:
            to_plot=[]
            for snr in snrs:
                to_plot.append( self.perf_dict[bg+'_'+snr])
            plt.plot(to_plot, label=bg)
        plt.scatter(4.5,self.perf_dict['dry'])

        plt.xticks([0,1,2,3,4], snrs, rotation='horizontal')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title("Word Psychophysics: Network")
        plt.ylabel('Proportion Correct')
        plt.xlabel('SNR(dB)')
        plt.show()

if __name__ == '__main__':
    main()
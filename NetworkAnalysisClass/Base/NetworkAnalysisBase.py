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
    
    
    def __init__(self, model, stimuli_path, meta, stimulus_files = False, stimulus_name = False, masking_exp = False, coch_size= (256,256), keyword = 'data'):
       
        
    
        self.base_dir = '/om2/user/ershook/NetworkAnalysesClass/data/'
    
        self.stimuli_path = stimuli_path
        
        self.model = model
        self.meta = meta
        self.masking_exp = masking_exp #ie. want to mask out bottom 5 subbands of cochleagram
        self.coch_size = coch_size #size of cgram (frequency, time)
        self.coch_size_flattened = self.coch_size[0] * self.coch_size[1]
        self.keyword = keyword #What the cgram dataset is named in the hdf5 ie. 'data' or in the case of spectemp 'spectemp-mags' because inconsistency is the spice of life
        
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
    
      
        
        
        #This is an annoying work around for interfacing with CNN.py
        
        if type(model)  == list:
            self.model = model[0]
            self.model_name = self.model.__class__.__name__ 
            self.class_name = self.model_name + '_' + self.stimulus_name 
            if self.masking_exp:
                self.model_name = self.model.__class__.__name__ + '_masking_exp' ##CHANGEMEBACK
                self.class_name = self.model_name + '_' + self.stimulus_name +'_masking_exp'##CHANGEMEBACK
            self.class_dir = self.getClassDir() #This is where all data associated with this model/cochleagrams will be saved
            #If in future need to write somewhere other than /om2 set up a symbolic link 
            
            self.unit_activations_dir = self.getUnitActivationsDir()
            self.unit_activations_hdf5_path = os.path.join(self.unit_activations_dir, self.class_name+'_unit_activations.hdf5')

            if not os.path.exists(self.unit_activations_hdf5_path):
                self.checkpoint = model[1]
                self.graph, self.tensors,self.session = self.model.getGraph(int(self.checkpoint))

        
        else:
            self.session = self.model.session
            self.graph = self.model.graph
            self.tensors = self.model.tensors
            if self.masking_exp:
                self.model_name = self.model.__class__.__name__+ '_masking_exp'###CHANGEMEBACK
                self.class_name = self.model_name + '_' + self.stimulus_name+ '_masking_exp'###CHANGEMEBACK
                
            else:
                self.model_name = self.model.__class__.__name__
                self.class_name = self.model_name + '_' + self.stimulus_name
            self.class_dir = self.getClassDir()
            
            self.unit_activations_dir = self.getUnitActivationsDir()
            self.unit_activations_hdf5_path = os.path.join(self.unit_activations_dir, self.class_name+'_unit_activations.hdf5')

        self.logits_dir = self.getLogitsDir()
        self.logits_hdf5_path = os.path.join(self.logits_dir, self.class_name+'_logits.hdf5')


     


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
    
    def getLogitsDir(self):
        d = os.path.join(self.class_dir, 'logits')
        if not os.path.exists(d):
            os.makedirs(d)
        return d
    
    
    def getNumberUnitsPerLayer(self):
        #Determine the number of units per layer (probably a better way but this is the only way I know how) 
        
        number_of_units_per_layer = {}
        dummy_coch = np.zeros((1, self.coch_size[0]*self.coch_size[1])) #Assumes shape of cochleagram is 256x256
        
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
                num_cochs += len(f_in[self.keyword])
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
                            for ind in range(len(f_in[self.keyword])):
                                with self.graph.as_default() as g:
                            
                                    batch = f_in[self.keyword][ind:ind+1,0:self.coch_size_flattened]
                                    if 'keep_prob' in self.tensors:
                                            measures = self.session.run(self.tensors, feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1, self.tensors['keep_prob']:1})
                                    else:
                                            measures = self.session.run(self.tensors, feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1})
                                    
                                    for layer in self.number_of_units_per_layer.keys():
                                        dataset_dict[layer][current_index] =  np.array(np.squeeze(measures[layer]))
                                current_index += 1
                               
                                if current_index %100 ==0:
                                        print 'current_index: ' + str(current_index)
        else:
            print "Note: unit activations hdf5 for this model and set of cochleagrams already exists."
    
    def _writeHDF5UnitActivations_batch(self, batch_no):
        #Write an HDF5 of unit activations to all cochleagrams in stimulus_files
        #But first checks that file doesn't already exist (since getting activations is computationally expensive)
        batch_path = os.path.join(self.unit_activations_dir, self.class_name + '_unit_activations_batch_' + str(batch_no) + '.hdf5')
        if not os.path.exists(batch_path):
        
            with h5py.File(batch_path, 'x') as f_out:
                
                 #Create dataset store pointer to dataset object in dictionary
                dataset_dict = {}
                for layer in self.number_of_units_per_layer.keys():
                    dim = (self.number_of_cochleagrams,) + self.number_of_units_per_layer[layer]
                    dataset_dict[layer] = f_out.create_dataset(layer, dim , dtype='float32')

                current_index = 0
                path = self.stimulus_files[batch_no]
                with h5py.File(path, 'r') as f_in:
                    for ind in range(len(f_in[self.keyword])):
                        with self.graph.as_default() as g:

                            batch = f_in[self.keyword][ind:ind+1,0:self.coch_size_flattened]
                            if 'keep_prob' in self.tensors:
                                    measures = self.session.run(self.tensors, feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1, self.tensors['keep_prob']:1})
                            else:
                                    measures = self.session.run(self.tensors, feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1})

                            for layer in self.number_of_units_per_layer.keys():
                                dataset_dict[layer][current_index] =  np.array(np.squeeze(measures[layer]))
                        current_index += 1

                        if current_index %100 ==0:
                                print 'current_index: ' + str(current_index)
        else:
            print "Note: unit activations hdf5 for this model and set of cochleagrams already exists."

               
    def _writeHDF5Logits(self):
        #Write an HDF5 of unit activations to all cochleagrams in stimulus_files
        #But first checks that file doesn't already exist (since getting activations is computationally expensive)
        
        if not os.path.exists(self.logits_hdf5_path):
        
            with h5py.File(self.logits_hdf5_path, 'x') as f_out:
                 
                    dim = (self.number_of_cochleagrams,) + self.number_of_units_per_layer['fc_top']
                    logits = f_out.create_dataset('fc_top', dim , dtype='float32')
                
                    current_index = 0

                    for _, path in enumerate(self.stimulus_files):
                        print 'Getting activations for hdf5 file number '+ str(_) + ' out of '+ str(len(self.stimulus_files))
                        with h5py.File(path, 'r') as f_in:
                            for ind in range(len(f_in[self.keyword])):
                                with self.graph.as_default() as g:
                            
                                    batch = f_in[self.keyword][ind:ind+1,0:self.coch_size_flattened]
                                
                                    if self.masking_exp:
                                        coch = batch.reshape((self.coch_size)) #CHANGEMEBACK
                                        coch[0:5,:] = np.zeros((5,256))#CHANGEMEBACK
                                        batch = coch.reshape((1,self.coch_size_flattened))#CHANGEMEBACK
                                
                                
                                    if 'keep_prob' in self.tensors:
                                            measures = self.session.run(self.tensors['fc_top'], feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1, self.tensors['keep_prob']:1})
                                    else:
                                            measures = self.session.run(self.tensors['fc_top'], feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1})
                                    
                                
                                    logits[current_index] =  np.array(np.squeeze(measures))
                                current_index += 1
                               
                                if current_index %100 ==0:
                                        print 'current_index: ' + str(current_index)
        else:
            print "Note: unit activations hdf5 for this model and set of cochleagrams already exists."
              
                
    def _writeHDF5Logits_batch(self, batch_no):
        #Write an HDF5 of unit activations to all cochleagrams in stimulus_files
        #But first checks that file doesn't already exist (since getting activations is computationally expensive)
        batch_path = os.path.join(self.logits_dir, self.class_name + '_logits_batch_' + str(batch_no) + '.hdf5')
        if not os.path.exists(batch_path):
        
            with h5py.File(batch_path, 'x') as f_out:
                 
                path = self.stimulus_files[batch_no]
                with h5py.File(path, 'r') as f_in:
                    dim = (len(f_in[self.keyword]),) + self.number_of_units_per_layer['fc_top']
                    logits = f_out.create_dataset('fc_top', dim , dtype='float32')
                
                    for ind in range(len(f_in[self.keyword])):
                        print ind
                        with self.graph.as_default() as g:

                            batch = f_in[self.keyword][ind:ind+1,0:self.coch_size_flattened]
                            if 'keep_prob' in self.tensors:
                                measures = self.session.run(self.tensors['fc_top'], feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1, self.tensors['keep_prob']:1})
                            else:
                                measures = self.session.run(self.tensors['fc_top'], feed_dict={self.tensors['x']: batch, self.tensors['y_label']: [0]*1})


                            logits[ind] =  np.array(np.squeeze(measures))
                            
        else:
            print "Note: unit activations hdf5 for this model and set of cochleagrams already exists."
                               
    
    def getUnitActivations(self):
        self.number_of_units_per_layer = self.getNumberUnitsPerLayer()
        self.number_of_cochleagrams = self.getNumberOfCochleagrams()
        self._writeHDF5UnitActivations()
        return self.unit_activations_hdf5_path

                
    
    def getUnitActivations_batch(self, batch_no):
        self.number_of_units_per_layer = self.getNumberUnitsPerLayer()
        self.number_of_cochleagrams = self.getNumberOfCochleagrams()
        self._writeHDF5UnitActivations_batch(batch_no)
        
    
    def getNetworkPerformance(self, remove_null = False): 
        self.correct = self.getCorrect()
        self.logits = self.getLogits(remove_null = remove_null)
        
        
        #if the size of the last cochleagram file contains zeros because less wavs than batch size
        if len(self.logits) > len(self.correct):
            if 1 == len(np.unique(self.logits[len(self.correct):])):
                self.logits = self.logits[:len(self.correct)]
        
        assert len(self.correct) == len(self.logits)
        self.is_correct = (self.correct == self.logits)
        self.overall_performance = np.sum(self.is_correct)/float(len(self.logits)) 
        print 'overall performance is ' + str(self.overall_performance * 100) +'%'
        self.raw_data_by_cond, self.perf_dict = self.getPerformanceByCondition()
        return self.overall_performance
    
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
        
        if os.path.exists(self.unit_activations_hdf5_path):
            with h5py.File(self.unit_activations_hdf5_path, 'r') as f_in:
                activations = np.array(f_in['fc_top'])
                
                
        elif not os.path.exists(self.logits_hdf5_path):
            self.number_of_units_per_layer = self.getNumberUnitsPerLayer()
            self.number_of_cochleagrams = self.getNumberOfCochleagrams()
            self._writeHDF5Logits()
            with h5py.File(self.logits_hdf5_path, 'r') as f_in:
                activations = np.array(f_in['fc_top'])
        else:
            with h5py.File(self.logits_hdf5_path, 'r') as f_in:
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
    
    def writeLogitsFile_batch(self, batch_no):
        self.number_of_units_per_layer = self.getNumberUnitsPerLayer()
        self.number_of_cochleagrams = self.getNumberOfCochleagrams()
        self._writeHDF5Logits_batch(batch_no)
          
    
    
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
            
            

    def consolidateLogitBatchHDF5s(self):
        files = os.listdir(self.logits_dir)
        
        num_batch_files = len([f for f in files if 'batch_' in f])
        if not hasattr(self, 'number_of_cochleagrams'):
            self.number_of_cochleagrams = self.getNumberOfCochleagrams()
            
        if not hasattr(self, 'number_of_units_per_layer'):
            self.number_of_units_per_layer = self.getNumberUnitsPerLayer()
        
        with h5py.File(self.logits_hdf5_path, 'x') as f_out:
            
            dim = (self.number_of_cochleagrams,) + self.number_of_units_per_layer['fc_top']
            logits = f_out.create_dataset('fc_top', dim , dtype='float32')
            
            current_index = 0 
            for batch_no in range(num_batch_files):
                print batch_no
                batch_path = os.path.join(self.logits_dir, self.class_name + '_logits_batch_' + str(batch_no) + '.hdf5')
                with h5py.File(batch_path, 'r') as f_in:
                    logits[current_index: current_index+ len(f_in['fc_top'])] = np.array(f_in['fc_top'])
                    current_index += len(f_in['fc_top'])
                
                
        
    def makeLinePlot(self, condition_order): 
        pass
    
    def KellEtAlFigure2BNet7TF(self, prefix = ''):
        # Code to generate this figure from /om/user/ershook/AlexCNNPaperProject/JupyterNotebooks/plotWordPsychophysics.ipynb
        backgrounds = ['Babble2Spkr', 'SpeakerShapedNoise', 'Music', 'AudScene', 'Babble8Spkr']
        snrs=['neg9db','neg6db','neg3db','0db', '3db']
        for bg in backgrounds:
            to_plot=[]
            for snr in snrs:
                to_plot.append( self.perf_dict[prefix+'_'+bg+'_'+snr])
            plt.plot(to_plot, label=bg)
        plt.scatter(4.5, self.perf_dict[prefix+'_'+'dry'])

        plt.xticks([0,1,2,3,4], snrs, rotation='horizontal')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if prefix !='':
            plt.title(prefix+' ' +"Word Psychophysics: Network")
        else:
            plt.title("Word Psychophysics: Network")
        plt.ylabel('Proportion Correct')
        plt.xlabel('SNR(dB)')
        plt.show()
        
    def makeBarPlot(self, conds = False):
        
        if not hasattr(self, 'perf_dict'):
            self.overall_performance = getNetworkPerformance(remove_null =True)
        
        n_groups = len(self.perf_dict.keys())
        fig, ax = plt.subplots()
        index = np.arange(n_groups)

        to_plot = []
        if conds == False: 
            conds = sorted(self.perf_dict.keys())
                           
        for key in conds:
            to_plot.append(self.perf_dict[key])

        rects1 = plt.bar(index, to_plot,label='network', color = '#1f77b4' )

        plt.ylim(0,1)
        plt.xticks(index+.5, sorted(self.perf_dict.keys()))
        plt.title(self.stimulus_name)

    
    def plotCochleagram(self, batch_no, clip):
        with h5py.File(self.stimulus_files[batch_no],'r') as f_in:
            plt.matshow(f_in[self.keyword][clip,0:self.coch_size_flattened].reshape(self.coch_size), origin='lower')
            plt.set_cmap('Blues')
            plt.title(self.stimulus_name)
            return f_in[self.keyword][clip,0:self.coch_size_flattened].reshape(self.coch_size)






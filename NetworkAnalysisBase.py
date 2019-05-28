class NetworkAnalysisBase(object):
    
    #Note: from now on should name cochleagram folders stimulus_type_batch_number
    #Note: assumes only two models genre and word -- in future could extend to support arbitray architecture
    # ie. allow num_units_per_layer to be passed in (should be only necessary change)
    
    def __init__(self, model, stimuli_path, meta, stimulus_files = False, stimulus_name = False):
        #Need to add meta as an input
        
        self.base_dir = '/om2/user/ershook/NetworkAnalysesClass/data/'
    
        self.stimuli_path = stimuli_path
        self.model_name = model.__class__.__name__
        self.model = model
        self.meta = meta
        
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
        
        for layer in model.layer_params.keys():
            activations = self.model.session.run(model.tensors[layer], feed_dict={self.model.tensors['x']: dummy_coch, self.model.tensors['y_label']: [0]*1, self.model.tensors['keep_prob']:1})
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
                                with model.graph.as_default() as g:
                            
                                    batch = f_in['data'][ind:ind+1,0:65536]
                                    measures = self.model.session.run(self.model.tensors, feed_dict={self.model.tensors['x']: batch, self.model.tensors['y_label']: [0]*1, self.model.tensors['keep_prob']:1})
                                    for layer in num_units_per_layer.keys():
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
        else: 
            pass
            #Need to add the path to key for Jenelle's dataset so when you are evaluating classifiers
    
        correct = []
        for word in self.meta[:,1]:
            correct.append(list(word_key).index(word)) 
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


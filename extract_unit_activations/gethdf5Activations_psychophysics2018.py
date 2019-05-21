def gethdf5Activations_psychophysics2018(num_units_per_layer, num_sounds, output_path):
        #Extract the activations to the psychophysics2018 dataset and save to a hdf5
        with h5py.File(output_path, 'x') as f_out:
            #Create datasets 
            dataset_dict = {}
            for layer in num_units_per_layer.keys():
                print layer
                dim = (num_sounds,)+ num_units_per_layer[layer]
                dset = f_out.create_dataset(layer, dim , dtype='float32')
                dataset_dict[layer] = dset 
            for layer in num_units_per_layer.keys():
                logits = []
                for subj_idx in range(0,20):
                    print subj_idx
                    for run_count in range(1,6):
                        hdf5_path = '/home/ershook/.skdata/network_analyses_2018x_v1_subj'+str(subj_idx)+'_run'+str(run_count)+'_999c6fc475be1e82e114ab9865aa5459e4fd329d/cache/image_cache_429140774cdaa5c2ffd6e9119890223d9b177953_None_hdf5/data.raw'

                        logits = getLayerUnitActivations(model, hdf5_path, logits, layer)
                dataset_dict[layer] = logits

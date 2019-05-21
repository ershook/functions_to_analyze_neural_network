To extract the activations to the psychophysics2018 dataset call gethdf5Activations_psychophysics2018(num_units_per_layer, num_sounds, output_path)
where the number of sounds= 8000 and

num_units_per_layer = {'conv1':(  86, 86, 96), 
                       'conv2': ( 22, 22, 256),
                       'conv3': (11, 11, 512),
                       'conv4': (11, 11, 1024),
                       'conv5': (  11, 11, 512),
                       'fc6': ( 1024,),
                       'fc_top': (589,)}

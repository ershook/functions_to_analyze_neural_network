def getLayerUnitActivations(model, path, logits, layer):
    # Returns the activations of all units in a given layer to all cochleagrams present in an hdf5
    with h5py.File(path, 'r') as f_in:
        for i in range(len(f_in['data'])):
            with model.graph.as_default() as g:
                batch = f_in['data'][i:i+1,0:65536]
                measures = model.session.run(model.tensors[layer], feed_dict={model.tensors['x']: batch, model.tensors['y_label']: [0]*1, model.tensors['keep_prob']:1})
                logits.append(measures)
    return logits



def load_model_unstrict(model, from_state_dict, def_layers=[]): 
    pre_trained_dict={}
    model_state_dict = model.state_dict()
    
    if def_layers:        
        try:
            for (tgt_layer, src_layer) in def_layers:
                pre_trained_dict[tgt_layer] = from_state_dict[src_layer]
            model_state_dict.update(pre_trained_dict)
            model.load_state_dict(model_state_dict)
        except Exception as e:
            print(e)
            print("Couldnt load specified layers maybe there are size differences or wrong names")
            return None, None, None

    keys_loaded = list()
    keys_notloaded = list()
    pre_trained_dict={}
    for key, param in model_state_dict.items():
        if key in def_layers:
            keys_loaded.append(key)
            continue
        if key in from_state_dict:
            value = from_state_dict[key]
            if param.shape == value.shape:
                pre_trained_dict[key] = value
                keys_loaded.append(key)
            else:
                keys_notloaded.append(key)
        else:
            keys_notloaded.append(key)
    try:
        model_state_dict.update(pre_trained_dict)
        model.load_state_dict(model_state_dict)
    except Exception as e:
        print("Error: {}".format(e))
        return None, None, None
    return model, keys_loaded, keys_notloaded


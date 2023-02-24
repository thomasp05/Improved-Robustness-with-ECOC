from Models import simple, simple_ensemble
import save_load_interface


##########################
### Ensemble of simple ###
##########################
# ensemble of simple
def get_ensemble(nb_models, base_path, dataset, no_softmax=False):
    path_1 = base_path + 'simple_1/simple1.pth'
    path_2 = base_path + 'simple_2/simple2.pth'
    path_3 = base_path + 'simple_3/simple3.pth'
    path_4 = base_path + 'simple_4/simple4.pth'
    path_5 = base_path + 'simple_5/simple5.pth'
    path_6 = base_path + 'simple_6/simple6.pth'
    path_7 = base_path + 'simple_7/simple7.pth'
    path_8 = base_path + 'simple_8/simple8.pth'
    path_9 = base_path + 'simple_9/simple9.pth'
    path_10 = base_path + 'simple_10/simple10.pth'
    path_11 = base_path + 'simple_11/simple11.pth'
    path_12 = base_path + 'simple_12/simple12.pth'
    path_13 = base_path + 'simple_13/simple13.pth'
    path_14 = base_path + 'simple_14/simple14.pth'
    path_15 = base_path + 'simple_15/simple15.pth'
    path_16 = base_path + 'simple_16/simple16.pth'
    model_paths = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9, path_10, path_11, path_12, path_13, path_14, path_15, path_16]

    if dataset == "CIFAR10": 
        filters = [32, 64, 128] # params A, B, C
        filter2 = [16] 

    elif dataset == "Fashion-MNIST": 
        filters = [32, 32, 32] # params A, B, C
        filter2 = [4]
 
    mod1 = simple(filters, filter2, dataset)
    mod2 = simple(filters, filter2, dataset)
    mod3 = simple(filters, filter2, dataset)
    mod4 = simple(filters, filter2, dataset)
    mod5 = simple(filters, filter2, dataset)
    mod6 = simple(filters, filter2, dataset)
    mod7 = simple(filters, filter2, dataset)
    mod8 = simple(filters, filter2, dataset)
    mod9 = simple(filters, filter2, dataset)
    mod10 = simple(filters, filter2, dataset)
    mod11 = simple(filters, filter2, dataset)
    mod12 = simple(filters, filter2, dataset)
    mod13 = simple(filters, filter2, dataset)
    mod14 = simple(filters, filter2, dataset)
    mod15 = simple(filters, filter2, dataset)
    mod16 = simple(filters, filter2, dataset)
    model_list = [mod1, mod2, mod3, mod4, mod5, mod6, mod7, mod8, mod9, mod10, mod11, mod12, mod13, mod14, mod15, mod16]

    final_model_list = []
    for i, model in enumerate(model_list):
        if i < nb_models:
            checkpoint = save_load_interface.load_checkpoint(model_paths[i])
            model.load_state_dict(checkpoint['model'])
            final_model_list.append(model.to("cuda"))
    model = simple_ensemble(final_model_list, no_sotfmax=no_softmax)

    return model
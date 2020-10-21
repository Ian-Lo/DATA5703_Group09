import torch

def val_step(features_map_val, structural_tokens_val,triggers_val,cells_content_tokens_val, model, LAMBDA):
    # NOT COMPLETED
    encoded_features_map_val = model.encoder.forward(features_map_val)
    predictions_val, loss_s_val, storage_hidden_val, pred_triggers = model.decoder_structural.predict(encoded_features_map_val, structural_target = structural_tokens_val )

    ### PROCESSING STORAGE ###
    # input to cell decoder:
        # hidden state of prediction: OK
        # encoded features map: OK
        # cell content target: not ok
    ### PROCESSING STORAGE ###
    list1 = []
    list2 = []
    list3 = []
    print("triggers_val.shape")
    print(triggers_val.shape)
    print("here")
    print(cells_content_tokens_val.shape)
    new_cells_content_tokens = cells_content_tokens_val[:,:]
    for n, example_triggers in enumerate(pred_triggers):#
        print("example_triggers")
        print(example_triggers)
        print("triggers_val[n]")
        print(triggers_val[n])

        indices_not_in_pred = torch.tensor([n for n in triggers_val[n] if n not in torch.tensor(example_triggers) ])
        print("indices_not_in_pred ")
        print(indices_not_in_pred )
        new_cells_content_tokens[n,:,:] = torch.zeros()
#        cells_content_tokens_val[]
        quit()
        # for each imestep:
            # compare predicted triggers to true triggers
                # if identical: keep both
                # if different: change true tokens to sequence of pad

        # finding intersection of elements in the predicted triggers:
    #    print("example_num")
    #    print(example_num)
    #    print("example_triggers")
    #    print(example_triggers)
    #    print("real triggers")
    #    print(triggers_val[example_num])
#        print(example_num, example_trigger)
        # find true predicted tokens for predicted cell
        ##### this is where I am at ##### reverting to implementing batching.

#    for example_num, example_triggers in enumerate(triggers_val):

#        cc_tk = cells_content_tokens[example_num]

#        for cell_num, example_trigger in enumerate(example_triggers):

#            if example_trigger != 0:
#                list1.append(encoded_features_map[example_num])

        #        list2.append(storage_hidden[example_trigger, 0, example_num, :])

#                list3.append(cc_tk[cell_num])

#        new_encoded_features_map = torch.stack(list1)
#        structural_hidden_state = torch.stack(list2).unsqueeze(0)
#    new_cells_content_tokens = torch.stack(list3)

#    predictions_cell, loss_cc_val = decoder_cell_content.predict(encoded_features_map, storage_hidden_val,cell_content_target =new_cells_content_tokens  )
    ####### validation end ########

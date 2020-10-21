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

    # create tensor of shape (num_examples, max_struc_token_pred, max_cell_tokens_true)
    # to hold ground truth predictions for cell tokens  corresponding to
    # structural predicitons
    num_examples = features_map_val.shape[0]
    max_struc_token_pred = max([ len(t) for t in pred_triggers])
    max_cell_tokens_true = cells_content_tokens_val.shape[2]

    new_cells_content_tokens = torch.zeros(num_examples , max_struc_token_pred , max_cell_tokens_true, dtype = cells_content_tokens_val.dtype)
    # fill up new_cell_content_tokens with only the truths that are relevant for
    # prediction
    for n, example_triggers in enumerate(pred_triggers):#
        # convert to list
        triggers_example_val = triggers_val[n].tolist()
        # remove trailing pads: below is the stupidest line of code I have ever written, but I can't find a way to unpad.
        triggers_example_val2 = [trigger for n, trigger in enumerate(triggers_example_val) if trigger not in [0] or n ==0 ]
        # get indices in the structural predictions that have a ground truth
        print("example_triggers")
        print(example_triggers)
        print("triggers_example_val")
        print(triggers_example_val)
        both = set(example_triggers).intersection(triggers_example_val2)
        indices_in_pred = [example_triggers.index(x) for x in sorted(both)]
        indices_in_truth = [triggers_example_val2.index(x) for x in sorted(both)]
        print("indices_in_pred")
        print(indices_in_pred)
        print("indices_in_truth")
        print(indices_in_truth)
        new_cells_content_tokens[n,indices_in_pred,:] = cells_content_tokens_val[n, indices_in_truth, :]
        ##### this is where I am at ##### reverting to implementing batching.
    quit()
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

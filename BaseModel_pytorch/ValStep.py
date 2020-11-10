import torch

def val_step(features_map_val, structural_tokens_val,triggers_val,cells_content_tokens_val, model, LAMBDA):

    encoded_features_map_val = model.encoder.forward(features_map_val)
    predictions_val, loss_s_val, storage_hidden_val, pred_triggers = model.decoder_structural.predict(encoded_features_map_val, structural_target = structural_tokens_val )

    # get shapes
    num_examples = features_map_val.shape[0]
    max_struc_token_pred = max([ len(t) for t in pred_triggers])
    max_cell_tokens_true = cells_content_tokens_val.shape[2]


    ### PROCESSING STORAGE ###
    list_features_map = []

    # create new array
    new_cells_content_tokens = torch.zeros(num_examples , max_struc_token_pred , max_cell_tokens_true, dtype = cells_content_tokens_val.dtype)

    # fill up new_cell_content_tokens with only the truths that are relevant for
    # prediction

    for n, example_triggers in enumerate(pred_triggers):#

        # convert to list
        triggers_example_val = triggers_val[n].tolist()

        # remove trailing pads: below is the stupidest line of code I have ever written, but I can't find a way to unpad.
        triggers_example_val2 = [trigger for n, trigger in enumerate(triggers_example_val) if trigger not in [0] or n ==0 ]

        # set indices in the structural predictions that have a ground truth
        both = set(example_triggers).intersection(triggers_example_val2)
        indices_in_pred = [example_triggers.index(x) for x in sorted(both)]
        indices_in_truth = [triggers_example_val2.index(x) for x in sorted(both)]
        new_cells_content_tokens[n,indices_in_pred,:] = cells_content_tokens_val[n, indices_in_truth, :]

    # run cell decoder
    if abs(1.0 - LAMBDA)>=0.001:
        # call cell decoder
        predictions_cell_val, loss_cc_val = model.decoder_cell_content.predict(encoded_features_map_val, storage_hidden_val,cell_content_target =new_cells_content_tokens  )
        loss_val = LAMBDA * loss_s_val + (1.0-LAMBDA) * loss_cc_val

    # do not run cell decoder:
    if abs( 1.0 - LAMBDA ) < 0.001:
        loss_val = loss_s_val
        predictions_cell_val = None
        loss_cc_val = None


    return predictions_val, loss_s_val, predictions_cell_val, loss_cc_val, loss_val

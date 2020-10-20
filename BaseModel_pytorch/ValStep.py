def val_step(features_map_val, structural_tokens_val,triggers_val,cells_content_tokens_val, model, LAMBDA):
    # NOT COMPLETED
    encoded_features_map_val = model.encoder.forward(features_map_val)
    predictions_val, loss_s_val, storage_hidden_val, pred_triggers = model.decoder_structural.predict(encoded_features_map_val, structural_target = structural_tokens_val )
    print("decoder_structure done")
    quit()
    ### PROCESSING STORAGE ###

    # merge input for predicted and ground truth

    ### PROCESSING STORAGE ###
    list1 = []
    list2 = []
    list3 = []

    for example_num, example_triggers in enumerate(pred_triggers):
        print(example_num, example_trigger)
        # find true predicted tokens for predicted cell
        ##### this is where I am at ##### reverting to implementing batching.

    for example_num, example_triggers in enumerate(triggers_val):

        cc_tk = cells_content_tokens[example_num]

        for cell_num, example_trigger in enumerate(example_triggers):

            if example_trigger != 0:
                list1.append(encoded_features_map[example_num])

                list2.append(storage_hidden[example_trigger, 0, example_num, :])

                list3.append(cc_tk[cell_num])

#        new_encoded_features_map = torch.stack(list1)
#        structural_hidden_state = torch.stack(list2).unsqueeze(0)
    new_cells_content_tokens = torch.stack(list3)

    predictions_cell, loss_cc_val = decoder_cell_content.predict(encoded_features_map, storage_hidden_val,cell_content_target =new_cells_content_tokens  )
    ####### validation end ########

def train_step(features_maps, structural_tokens, triggers, cells_content_tokens, LAMBDA=0.5):

    encoded_features_map = encoder.forward(features_map)
    predictions, loss_s, storage_hidden = decoder_structural.forward(encoded_features_map, structural_tokens)

    # only run cell decoder if LAMBDA!=0
    if LAMBDA:
        ### PROCESSING STORAGE ###
        list1 = []
        list2 = []
        list3 = []

        for example_num, example_triggers in enumerate(triggers):
            cc_tk = cells_content_tokens[example_num]

            for cell_num, example_trigger in enumerate(example_triggers):

                if example_trigger != 0:
                    list1.append(encoded_features_map[example_num])

                    list2.append(storage_hidden[example_trigger, 0, example_num, :])

                    list3.append(cc_tk[cell_num])

        new_encoded_features_map = torch.stack(list1)
        structural_hidden_state = torch.stack(list2).unsqueeze(0)
        new_cells_content_tokens = torch.stack(list3)
        predictions_cell, loss_cc = decoder_cell_content.forward(new_encoded_features_map, structural_hidden_state, new_cells_content_tokens)

    if lambdas[epoch]:
        loss = lambdas[epoch] * loss_s + (1.0-lambdas[epoch]) * loss_cc
    else:
        loss = lambdas[epoch] * loss_s

    # Back propagation
    decoder_cell_content_optimizer.zero_grad()
    decoder_structural_optimizer.zero_grad()
    encoder_optimizer.zero_grad()
    loss.backward()

    # Update weights
    decoder_cell_content_optimizer.step()#
    decoder_structural_optimizer.step()
    encoder_optimizer.step()

  return predictions, loss_s, predictions_cell, loss_cc, loss

import torch

def train_step(features_map, structural_tokens, triggers, cells_content_tokens, model, LAMBDA=0.5):

    # pass features through encoder
    encoded_structural_features_map = model.encoder_structural.forward(features_map)
    encoded_cell_content_features_map = model.encoder_cell_content.forward(features_map)

    # pass encoded features through structural decoder
    predictions, loss_s, storage_hidden = model.decoder_structural.forward(encoded_structural_features_map, structural_tokens)

    # run structural decoder only if lambda != 1
    if abs(LAMBDA-1)>0.001:
        ### PROCESSING STORAGE ###
        list1 = []
        list2 = []
        list3 = []

        for example_num, example_triggers in enumerate(triggers):
            cc_tk = cells_content_tokens[example_num]

            for cell_num, example_trigger in enumerate(example_triggers):

                if example_trigger != 0:
                    list1.append(encoded_cell_content_features_map[example_num])

                    list2.append(storage_hidden[example_trigger, 0, example_num, :])

                    list3.append(cc_tk[cell_num])

        new_encoded_features_map = torch.stack(list1)
        structural_hidden_state = torch.stack(list2).unsqueeze(0)
        new_cells_content_tokens = torch.stack(list3)
        predictions_cell, loss_cc = model.decoder_cell_content.forward(new_encoded_features_map, structural_hidden_state, new_cells_content_tokens)
    # calculate loss and update weights

    if abs(LAMBDA-1.0)>=0.001:
        loss = LAMBDA * loss_s + (1.0-LAMBDA) * loss_cc
        # Back propagation
        model.decoder_cell_content_optimizer.zero_grad()
        model.decoder_structural_optimizer.zero_grad()
        model.encoder_cell_content_optimizer.zero_grad()
        model.encoder_structural_optimizer.zero_grad()
        loss.backward()

        # Update weights
        model.decoder_cell_content_optimizer.step()#
        model.decoder_structural_optimizer.step()
        model.encoder_cell_content_optimizer.step()
        model.encoder_structural_optimizer.step()

    if abs(LAMBDA-1.0)<0.001:
        loss = loss_s
        # Back propagation
        model.decoder_structural_optimizer.zero_grad()
        model.encoder_structural_optimizer.zero_grad()
        loss.backward()
        # Update weights
        model.decoder_structural_optimizer.step()
        model.encoder_structural_optimizer.step()
        predictions_cell = None
        loss_cc = None

    return predictions, loss_s, predictions_cell, loss_cc, loss

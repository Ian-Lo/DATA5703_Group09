import torch

def train_step(features_map,
                structural_tokens,
                triggers,
                cells_content_tokens,
                model,
                LAMBDA=0.5,
                alpha_c_struc = 0.0,
                alpha_c_cell_content = 0.0):

    # pass features through encoder
    encoded_structural_features_map = model.encoder_structural.forward(features_map)
    encoded_cell_content_features_map = model.encoder_cell_content.forward(features_map)

    # pass encoded features through structural decoder
    predictions, loss_s, storage_hidden = model.decoder_structural.forward(encoded_structural_features_map, structural_tokens, alpha_c_struc = alpha_c_struc)


    # run structural decoder only if lambda != 1
    if abs(LAMBDA-1)>0.001:
        # find number of cells in each example
        cells_in_examples = []
        for n, example in enumerate(triggers):
            num_cells = example.numel() - (example == 0).sum()
            cells_in_examples.append(num_cells)

        # stack feature maps. One example corresponds to on trigger predicted
        list1_ = torch.repeat_interleave(encoded_cell_content_features_map, torch.stack(cells_in_examples) , dim = 0)

        # prepare to do the same for hidden states
        storage_hidden_ = torch.squeeze(storage_hidden, 1).permute([1,0,2])

        l1 = []
        l2 = []
        for example_num, example_triggers in enumerate(triggers):
            for cell_num, example_trigger in enumerate(example_triggers):
                if example_trigger != 0:
                    l1.append(example_num)
                    l2.append(example_trigger)
        l1 = torch.tensor(l1)
        l2 = torch.tensor(l2)

        list2_ = storage_hidden_[l1, l2]

        # get the ground truths into the same format
        l3 = []
        l4 = []
        for n, cells_number in enumerate(cells_in_examples):
            l3.extend([n]*cells_number)
            l4.extend(list(range(cells_number)))

        list3_ = cells_content_tokens[l3, l4]


        # ### PROCESSING STORAGE ###
        # list1 = []
        # list2 = []
        # list3 = []
        #
        # count = 0
        # for example_num, example_triggers in enumerate(triggers):
        #     cc_tk = cells_content_tokens[example_num]
        #
        #     for cell_num, example_trigger in enumerate(example_triggers):
        #
        #         if example_trigger != 0:
        #             # list1_[count, :,:] = encoded_cell_content_features_map[example_num]
        #             #
        #             # list2_[:, count, : ] = storage_hidden[example_trigger, 0, example_num, :]
        #             #
        #             # list3_[count, : ] = cc_tk[cell_num]
        #             list1.append(encoded_cell_content_features_map[example_num])
        #
        #             list2.append(storage_hidden[example_trigger, 0, example_num, :])
        #
        #             list3.append(cc_tk[cell_num])
        #             count += 1


        # new_encoded_features_map = torch.stack(list1)
        # structural_hidden_state = torch.stack(list2).unsqueeze(0)
        # new_cells_content_tokens = torch.stack(list3)
        new_encoded_features_map = list1_
        structural_hidden_state = list2_.unsqueeze(0)
        new_cells_content_tokens = list3_

        predictions_cell, loss_cc = model.decoder_cell_content.forward(new_encoded_features_map, structural_hidden_state, new_cells_content_tokens, alpha_c_cell_content = alpha_c_cell_content)

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

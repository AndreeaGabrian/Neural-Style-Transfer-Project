import utils.image_utils as utils
import utils.args_configurations as args_config
import utils.model_utils as model_utils

import os
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


def make_tuning_step(model, optimizer, target_representation, should_reconstruct_content, content_feature_maps_index, style_feature_maps_indices):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        # Finds the current representation
        set_of_feature_maps = model(optimizing_img)
        if should_reconstruct_content:
            current_representation = set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        else:
            current_representation = [model_utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices]

        # Computes the loss between current and target representations
        loss = 0.0
        if should_reconstruct_content:
            loss = torch.nn.MSELoss(reduction='mean')(target_representation, current_representation)
        else:
            for gram_gt, gram_hat in zip(target_representation, current_representation):
                loss += (1 / len(target_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item(), current_representation

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def reconstruct_image_from_representation(config):
    should_reconstruct_content = config['reconstruct_content']
    should_visualize_representation = config['visualize_representation']
    saving_path = os.path.join(config['output_img_dir'], ('c' if should_reconstruct_content else 's') + '_reconstruction_' + config['optimizer'])
    saving_path = os.path.join(saving_path, os.path.basename(config['content_image_name']).split('.')[0] if should_reconstruct_content else os.path.basename(config['style_image_name']).split('.')[0])
    os.makedirs(saving_path, exist_ok=True)

    content_img_path = os.path.join(config['content_images_dir'], config['content_image_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_image_name'])
    img_path = content_img_path if should_reconstruct_content else style_img_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = utils.transform_image(img_path, config['height'], device)
    white_noise_img = np.random.uniform(-90., 90., img.shape).astype(np.float32)
    init_img = torch.from_numpy(white_noise_img).float().to(device)
    optimizing_img = Variable(init_img, requires_grad=True)

    # indices pick relevant feature maps (say conv4_1, relu1_1, etc.)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = model_utils.prepare_model(config['model'], device)

    num_of_iterations = {'lbfgs': 300}
    set_of_feature_maps = neural_net(img)

    # Visualize feature maps and Gram matrices (depending on whether you're reconstructing content or style img)
    if should_reconstruct_content:
        target_content_representation = set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        if should_visualize_representation:
            num_of_feature_maps = target_content_representation.size()[0]
            print(f'Number of feature maps: {num_of_feature_maps}')
            for i in range(num_of_feature_maps):
                feature_map = target_content_representation[i].to('cpu').numpy()
                feature_map = np.uint8(utils.get_uint8_range(feature_map))
                plt.imshow(feature_map)
                plt.title(f'Feature map {i+1}/{num_of_feature_maps} from layer {content_feature_maps_index_name[1]} (model={config["model"]}) for {config["content_image_name"]} image.')
                plt.show()
                filename = f'fm_{config["model"]}_{content_feature_maps_index_name[1]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
                utils.save_image(feature_map, os.path.join(saving_path, filename))
    else:
        target_style_representation = [model_utils.gram_matrix(fmaps) for i, fmaps in enumerate(set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
        if should_visualize_representation:
            num_of_gram_matrices = len(target_style_representation)
            print(f'Number of Gram matrices: {num_of_gram_matrices}')
            for i in range(num_of_gram_matrices):
                Gram_matrix = target_style_representation[i].squeeze(axis=0).to('cpu').numpy()
                Gram_matrix = np.uint8(utils.get_uint8_range(Gram_matrix))
                plt.imshow(Gram_matrix)
                plt.title(f'Gram matrix from layer {style_feature_maps_indices_names[1][i]} (model={config["model"]}) for {config["style_image_name"]} image.')
                plt.show()
                filename = f'gram_{config["model"]}_{style_feature_maps_indices_names[1][i]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
                utils.save_image(Gram_matrix, os.path.join(saving_path, filename))

    if config['optimizer'] == 'lbfgs':
        iteration = 0
        # closure is required by lbfgs optimizer
        def closure():
            nonlocal iteration
            optimizer.zero_grad()
            loss = 0.0
            if should_reconstruct_content:
                loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, neural_net(optimizing_img)[content_feature_maps_index_name[0]].squeeze(axis=0))
            else:
                current_set_of_feature_maps = neural_net(optimizing_img)
                current_style_representation = [model_utils.gram_matrix(fmaps) for i, fmaps in enumerate(current_set_of_feature_maps) if i in style_feature_maps_indices_names[0]]
                for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                    loss += (1 / len(target_style_representation)) * torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
            loss.backward()
            with torch.no_grad():
                print(f'Iteration: {iteration}, current {"content" if should_reconstruct_content else "style"} loss={loss.item()}')
                utils.save_generate_images(optimizing_img, saving_path, config, iteration, num_of_iterations[config['optimizer']], should_display=False)
                iteration += 1
            return loss

        optimizer = torch.optim.LBFGS((optimizing_img,), max_iter=num_of_iterations[config['optimizer']], line_search_fn='strong_wolfe')
        optimizer.step(closure)

    return saving_path


if __name__ == "__main__":
    config = args_config.get_config()
    results_path = reconstruct_image_from_representation(config)




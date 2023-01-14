import utils.image_utils as image_utils
import utils.model_utils as model_utils
import utils.args_configurations as args_config
import torch
from torch.optim import Adam, LBFGS
from torch.autograd import Variable
import numpy as np
import os


def total_variation(input):
    return torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))


def compute_content_loss(target_content_representation, current_content_representation):
    """
    Computed using L2 norm(original) or L2 norm with reduction=sum and L1 norm
    :param target_content_representation: target_content_representation
    :param current_content_representation: current_content_representation
    :return: content loss
    """
    # MSELoss measures the mean squared error (squared L2 norm) between each element in the input x and target y.
    # mean reduction L2
    #content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)
    # sum reduction L2
    #content_loss = torch.nn.MSELoss(reduction='sum')(target_content_representation, current_content_representation)
    #L1
    content_loss = torch.nn.functional.l1_loss(current_content_representation, target_content_representation, size_average=None, reduce=None, reduction='mean')
    return content_loss


def compute_style_loss(target_style_representation, style_feature_maps_indices, current_set_of_feature_maps):
    style_loss = 0.0
    current_style_representation = [model_utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if
                                    cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)
    return style_loss


def compute_variation_loss(image_to_optimize):
    return total_variation(image_to_optimize)


def compute_loss(cnn, image_to_optimize, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]

    current_set_of_feature_maps = cnn(image_to_optimize)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)

    content_loss = compute_content_loss(target_content_representation, current_content_representation)
    style_loss = compute_style_loss(target_style_representation, style_feature_maps_indices, current_set_of_feature_maps)
    total_variation = compute_variation_loss(image_to_optimize)

    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['total_variation_weight'] * total_variation

    return total_loss, content_loss, style_loss, total_variation


def make_one_step(cnn, config, optimizer, result_image, target_representations, content_feature_maps_index, style_feature_maps_indices):
    total_loss, content_loss, style_loss, total_variation_loss = compute_loss(cnn, result_image, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return total_loss, content_loss, style_loss, total_variation_loss


def run_transfer(config):
    style_path = os.path.join(config['style_images_dir'], config['style_image_name'])
    content_path = os.path.join(config['content_images_dir'], config['content_image_name'])
    out_dir_name = 'transferred_' + os.path.split(content_path)[1].split('.')[0] + '_' + os.path.split(style_path)[1].split('.')[0] + '_' + config["init_method"] + '_' + str(config["content_weight"]) + '_' + str(config["style_weight"]) + '_' + str(config["total_variation_weight"])
    saving_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(saving_path, exist_ok=True)
    # if there is cuda available the computation will be done by it, it is much, much faster this way
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_image = image_utils.transform_image(content_path, config['height'], device)
    style_image = image_utils.transform_image(style_path, config['height'], device)

    if config['init_method'] == 'random':  # generate random noise as starting image, using gaussian noise
        noise_image = np.random.normal(loc=0, scale=90., size=content_image.shape).astype(np.float32)
        init_img = torch.from_numpy(noise_image).float().to(device)
    elif config['init_method'] == 'content':  # consider the content image as starting image
        init_img = content_image
    else:
        # initial image needs to have the same dimension as the content image and both to have
        # same sie for the feature maps
        init_img = image_utils.transform_image(style_path, np.asarray(content_image.shape[2:]), device)
    # we are modifying the pixel's values, so we need to set requires_grad = True
    result_image = Variable(init_img, requires_grad=True)
    cnn, content_feature_maps_index_name, style_feature_maps_indices_names = model_utils.prepare_model(config['model'], device)
    print(f'Running {config["model"]} with {config["optimizer"]} optimizer and init_image = {config["init_method"]}.')

    style_img_set_of_feature_maps = cnn(style_image)
    content_img_set_of_feature_maps = cnn(content_image)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [model_utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]
    iteration_number = {
        "lbfgs": 1000,
    }

    # Here optimization starts
    if config['optimizer'] == 'lbfgs':
        optimizer = LBFGS((result_image,), max_iter=iteration_number['lbfgs'], line_search_fn='strong_wolfe')
        iteration = 0

        def closure():
            nonlocal iteration
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, total_variation_loss = compute_loss(cnn, result_image, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'lbfg | iteration: {iteration:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, total variation loss={config["total_variation_weight"] * total_variation_loss.item():12.4f}')
                image_utils.save_generate_images(result_image, saving_path, config, iteration, iteration_number[config['optimizer']], should_display=False)
            iteration += 1
            return total_loss
        optimizer.step(closure)
    return saving_path


if __name__ == "__main__":
    config = args_config.get_config()
    results_path = run_transfer(config)


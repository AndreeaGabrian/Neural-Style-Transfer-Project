import os


def get_config():
    default_resource_dir = r"C:\Users\gabri\Documents\AN 3 SEM 1\CVDL\NeuralStyleTransferProject\data"
    content_images_dir = os.path.join(default_resource_dir, 'content-images')
    style_images_dir = os.path.join(default_resource_dir, 'style-images')
    output_img_dir = os.path.join(default_resource_dir, 'output-images')
    img_format = (4, '.jpg')  # saves images in the format: %04d.jpg

    config = dict()

    config['content_images_dir'] = content_images_dir
    config['style_images_dir'] = style_images_dir
    config['output_img_dir'] = output_img_dir
    config['img_format'] = img_format

    # reconstruct_content = pick between content or style image reconstruction
    config['reconstruct_content'] = True
    # visualize_representation = visualize feature maps or Gram matrices
    config['visualize_representation'] = False
    # content_img_name
    config['content_image_name'] = 'puppets.jpg'
    # style_img_name
    config['style_image_name'] = 'starry_night.jpg'
    # height = width of content and style images (-1 keep in original)
    config['height'] = 500
    # saving_freq = saving frequency for intermediate images (-1 = final image)
    config['saving_freq'] = 1
    # model
    config['model'] = 'vgg19'
    # optimizer = lbfgs or adam
    config['optimizer'] = 'lbfgs'
    # reconstruct_script = dummy param - used in saving func
    config['reconstruct_script'] = True
    # init_method = random, content or style
    config['init_method'] = 'noise'
    # content_weight = weight for content loss
    config['content_weight'] = 1e5
    # style_weight = weight for style loss
    config['style_weight'] = 1e3
    # tv_weight = weight for total variation loss
    config['total_variation_weight'] = 1e1

    return config

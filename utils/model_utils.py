import torch
from model.vgg import Vgg19


def prepare_model(model, device):
    # requires_grad=False because we are not tuning model weights, just image's pixels
    model = Vgg19(requires_grad=False, show_progress=True)
    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names


def gram_matrix(input, should_normalize=True):
    (batch_size, nr_feature_maps, height, width) = input.size()
    features = input.view(batch_size, nr_feature_maps, height * width)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= nr_feature_maps * width * height
    return gram


def gram_matrix2(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d) c=width d=height
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)




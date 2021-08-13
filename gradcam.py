import torch
import torch.nn.functional as F

# from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer


class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if layer_name == 'layer1':
            target_layer = self.model_arch.layer1
        elif layer_name == 'layer2':
            target_layer = self.model_arch.layer2
        elif layer_name == 'layer3':
            target_layer = self.model_arch.layer3
        elif layer_name == 'down_channel':
            target_layer = self.model_arch.down_channel

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 8, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[3:])

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, t, c, h, w = input.size()
        logit = self.model_arch(input, return_logits = True)
        if class_idx is None:
            score = logit.max(1)[0]
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        # print(score.size())
        score.backward( retain_graph=True)

        gradients = self.gradients['value'].squeeze(1)
        activations = self.activations['value'].squeeze(1)
        # b, t, k, u, v = gradients.size()
        # print(gradients.size())

        if len(gradients.size()) == 4:
            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
            weights = alpha.view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)  # 1 x 1 x 16 x 8
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        else:
            b, s, k, u, v = gradients.size()
            # alpha = gradients.view(b, s, k, -1).mean(3)
            alpha = F.relu(gradients.view(b, s, k, -1).mean(3))
            weights = alpha.view(b, s, k, 1, 1)

            saliency_map = (weights * activations).sum(2, keepdim=True)
            saliency_map = F.relu(saliency_map).squeeze(0)
            saliency_map = F.interpolate(saliency_map, size=(h,w), mode='bilinear', align_corners=False)
            saliency_map_min = torch.min(saliency_map.view(s, -1), 1)[0].view(s, 1, 1, 1)
            saliency_map_max = torch.max(saliency_map.view(s, -1), 1)[0].view(s, 1, 1, 1)
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        # b, c, h, w = input.size()

        logit = self.model_arch(input, return_logits = True)
        if class_idx is None:
            score = logit.max(1)[0]
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'].squeeze(1)
        activations = self.activations['value'].squeeze(1)

        if len(gradients.size()) == 4:
            b, k, u, v = gradients.size()

            alpha_num = gradients.pow(2)
            alpha_denom = gradients.pow(2).mul(2) + \
                            activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

            alpha = alpha_num.div(alpha_denom + 1e-7)
            positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(256, 128), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        else:
            b, s, k, u, v = gradients.size()

            alpha_num = gradients.pow(2)
            alpha_denom = gradients.pow(2).mul(2) + \
                          activations.mul(gradients.pow(3)).view(b, s, k, u * v).sum(-1, keepdim=True).view(b, s, k, 1, 1)
            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
            alpha = alpha_num.div(alpha_denom + 1e-7)
            positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha * positive_gradients).view(b, s, k, u * v).sum(-1).view(b, s, k, 1, 1)

            saliency_map = (weights * activations).sum(2, keepdim=True)
            saliency_map = F.relu(saliency_map).squeeze(0)
            saliency_map = F.interpolate(saliency_map, size=(256, 128), mode='bilinear', align_corners=False)
            saliency_map_min = torch.min(saliency_map.view(s, -1), 1)[0].view(s, 1, 1, 1)
            saliency_map_max = torch.max(saliency_map.view(s, -1), 1)[0].view(s, 1, 1, 1)
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

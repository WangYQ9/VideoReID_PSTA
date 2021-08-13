import os.path as osp
import cv2
import PIL
import torch.nn.functional as F
from utils import mkdir_if_missing
import numpy as np
import shutil
import torch
from matplotlib import pyplot as plt
from gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image

class MA(object):
    def __init__(self, save_dir, width = 128, height = 256):
        super(MA, self).__init__()

        self.save_dir = save_dir
        self.width = width
        self.height = height

    def heat_map(self, testloader, model, test_name, dataset_name, print_freq=10):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        with torch.no_grad():
            GRID_SPACING = 10
            save_dir = osp.join(self.save_dir, test_name)
            target = dataset_name

            actmap_dir = osp.join(save_dir, 'actmap_' + target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(testloader):
                imgs, paths = data[0], data[3]
                imgs = imgs.cuda()

                # forward to get convolutional feature maps
                try:
                    outputs = model(imgs)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_featuremaps". ' \
                                    'Please add return_featuremaps as an input argument to forward(). When ' \
                                    'return_featuremaps=True, return feature maps only.')

                if outputs.dim() != 5:
                    raise ValueError('The model output is supposed to have ' \
                                     'shape of (b, c, h, w), i.e. 5 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs.dim()))

                # compute activation maps
                outputs = (outputs).sum(2)
                b, k, h, w = outputs.size()
                outputs = outputs.view(b * k, h * w)
                # outputs = F.normalize(outputs, p=2, dim=1)
                outputs = outputs.view(b, k, h, w)
                imgs, outputs = imgs.cpu(), outputs.cpu()

                for j in range(b):

                    for i in range(k):
                        # get image name
                        path = paths[i][j]
                        imname = osp.basename(osp.splitext(path)[0])
                        # RGB image
                        img = imgs[j, i, :, :, :]
                        img_np = np.uint8(np.floor(img.numpy() * 255))
                        img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

                        # activation map
                        am = outputs[j, i, :, :].detach().numpy()
                        am = cv2.resize(am, (self.width, self.height))
                        am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                        am = np.uint8(np.floor(am))
                        am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                        # overlapped
                        overlapped = img_np * 0.3 + am * 0.7
                        overlapped[overlapped > 255] = 255
                        overlapped = overlapped.astype(np.uint8)

                        # save images in a single figure (add white spacing between images)
                        # from left to right: original image, activation map, overlapped image
                        grid_img = 255 * np.ones((self.height, 3 * self.width + 2 * GRID_SPACING, 3), dtype=np.uint8)
                        grid_img[:, :self.width, :] = img_np[:, :, ::-1]
                        grid_img[:, self.width + GRID_SPACING: 2 * self.width + GRID_SPACING, :] = am
                        grid_img[:, 2 * self.width + 2 * GRID_SPACING:, :] = overlapped
                        cv2.imwrite(osp.join(actmap_dir, imname + '.jpg'), grid_img)

                if (batch_idx + 1) % print_freq == 0:
                    print('- done batch {}/{}'.format(batch_idx + 1, len(testloader)))

    def visualize_ranked_results(self, distmat, dataset, data_type, width=128, height=256, topk=10):
        """Visualizes ranked results.

        Supports both image-reid and video-reid.

        For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
        saved in folders each containing a tracklet.

        Args:
            distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
            dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
                tuples of (img_path(s), pid, camid).
            data_type (str): "image" or "video".
            width (int, optional): resized image width. Default is 128.
            height (int, optional): resized image height. Default is 256.
            save_dir (str): directory to save output images.
            topk (int, optional): denoting top-k images in the rank list to be visualized.
                Default is 10.
        """
        GRID_SPACING = 10
        QUERY_EXTRA_SPACING = 90
        BW = 5  # border width
        GREEN = (0, 255, 0)
        RED = (0, 0, 255)
        save_dir = osp.join(self.save_dir, 'Rank')

        num_q, num_g = distmat.shape
        mkdir_if_missing(save_dir)

        print('# query: {}\n# gallery {}'.format(num_q, num_g))
        print('Visualizing top-{} ranks ...'.format(topk))

        query, gallery = dataset
        assert num_q == len(query)
        assert num_g == len(gallery)

        indices = np.argsort(distmat, axis=1)

        for q_idx in range(num_q):
            qimg_path, qpid, qcamid = query[q_idx]
            qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

            if data_type == 'image':
                qimg = cv2.imread(qimg_path)
                qimg = cv2.resize(qimg, (width, height))
                qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                # resize twice to ensure that the border width is consistent across images
                qimg = cv2.resize(qimg, (width, height))
                num_cols = topk + 1
                grid_img = 255 * np.ones((height, num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING, 3),
                                         dtype=np.uint8)
                grid_img[:, :width, :] = qimg
            else:
                qdir = osp.join(save_dir, osp.basename(osp.splitext(qimg_path_name)[0]))
                mkdir_if_missing(qdir)
                self._cp_img_to(qimg_path, qdir, rank=0, prefix='query')

            rank_idx = 1
            for g_idx in indices[q_idx, :]:
                gimg_path, gpid, gcamid = gallery[g_idx]
                invalid = (qpid == gpid) & (qcamid == gcamid)

                if not invalid:
                    matched = gpid == qpid
                    if data_type == 'image':
                        border_color = GREEN if matched else RED
                        gimg = cv2.imread(gimg_path)
                        gimg = cv2.resize(gimg, (width, height))
                        gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                        gimg = cv2.resize(gimg, (width, height))
                        start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        end = (rank_idx + 1) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                        grid_img[:, start: end, :] = gimg
                    else:
                        self._cp_img_to(gimg_path, qdir, rank=rank_idx, prefix='gallery', matched=matched)

                    rank_idx += 1
                    if rank_idx > topk:
                        break

            if data_type == 'image':
                imname = osp.basename(osp.splitext(qimg_path_name)[0])
                cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

            if (q_idx + 1) % 100 == 0:
                print('- done {}/{}'.format(q_idx + 1, num_q))

        print('Done. Images have been saved to "{}" ...'.format(save_dir))

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3)) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    def cmc_curve(self):
        pass

    def Gram(self, model, loader, layer_name, print_freq=10):

        model = model.eval()
        model_dict = dict(type='STAM', arch = model, layer_name = layer_name, input_size = [256, 128])

        model_gradcam = GradCAM(model_dict)
        model_gradcammp = GradCAMpp(model_dict)
        mean, var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # cam_dict = [model_gradcam, model_gradcammp]

        for batch_idx, data in enumerate(loader):
            imgs, paths = data[0], data[3]
            id_name  = paths[0][0].split('/')[-2]
            gram_dir = osp.join(self.save_dir, 'Gram' + model_dict['layer_name'])
            mkdir_if_missing(gram_dir)
            imgs = imgs.cuda()
            mask, _ = model_gradcam(imgs)

            mask = mask.cpu()
            heat_map, result_list, _ = visualize_cam(mask, paths)

            mask_pp, _ = model_gradcammp(imgs)
            mask_pp = mask_pp.cpu()
            heatmap_pp, result_pp_list, imgs_list = visualize_cam(mask_pp, paths)

            # define the save path
            imgs_path = osp.join(gram_dir, id_name)
            mkdir_if_missing(imgs_path)

            # save imgage
            # images = []
            t = len(imgs_list)
            s = heat_map.size(0)
            k = int(t / s)
            num = 1
            for i in range(len(imgs_list)):
                if i >= num * k:
                    num = num + 1
                picture_name = paths[i][0].split('/')[-1]
                path = osp.join(imgs_path, picture_name)
                image = torch.stack([imgs_list[i], heat_map[num-1, :, :, :],
                                     result_list[i], heatmap_pp[num-1, :, :, :], result_pp_list[i]], 0)
                images = make_grid(image, nrow=5)
                save_image(images, path)
                PIL.Image.open(path)

        if (batch_idx + 1) % print_freq == 0:
            print('- done batch {}/{}'.format(batch_idx + 1, len(loader)))

    def SNE(self):
        pass


def visualize_cam(mask, paths):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    t = len(paths)
    imgs = []
    for i in range(t):
        img = PIL.Image.open(paths[i][0])
        torch_img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
        torch_img = F.interpolate(torch_img, size=(256, 128), mode='bilinear', align_corners=False).squeeze(0)
        imgs.append(torch_img)

    s = mask.size(0)
    heatmap_list = []
    for i in range(s):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask[i, :, :, :].squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        b, g, r = heatmap.split(1)
        heatmap = torch.cat([r, g, b])
        heatmap_list.append(heatmap)

    result_list = []
    k = int(t / s)
    num = 1
    for i in range(t):
        if i >= num * k :
            num = num + 1
        result = heatmap_list[num - 1] + imgs[i]
        result = result.div(result.max()).squeeze()
        result_list.append(result)

    return torch.stack(heatmap_list, 0), result_list, imgs


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)

def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)

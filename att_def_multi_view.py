import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np
import random

from kornia.geometry import transform_points
from shapely import Polygon
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from math import ceil

from AttDefMVDeTr import arguments
from AttDefMVDeTr.multiview_detector.datasets import Wildtrack, frameDataset
from AttDefMVDeTr.multiview_detector.loss import FocalLoss
from AttDefMVDeTr.multiview_detector.models.mvdetr import MVDeTr
from AttDefMVDeTr.multiview_detector.trainer import PerspectiveTrainer
from AttDefMVDeTr.multiview_detector.utils.image_utils import img_color_denormalize
from AttDefMVDeTr.multiview_detector.utils.projection import get_worldcoord_from_imgcoord_mat
from AttDefMVDeTr.pcgrad.pcgrad import PCGrad


Normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
Denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class TradLoss(FocalLoss):
    def __init__(self):
        super(TradLoss, self).__init__()

    def forward(self, outputs, targets, mask=None):
        """
        This method is a wrapper method to maximize focal loss, originally used by MVDeTr
        Args:
            - output: the outputs generated using adversarial data
            - target: the ground-truth targets
        Returns:
            - model loss
        """
        loss = super().forward(outputs, targets, mask)
        return -loss


class InPtrLoss(nn.Module):
    def __init__(self, dataset, tar_patches_info, n_heads=8, n_points=4, downsample=2, smoother=False):
        """
        Args:
            - dataset: the dataset object (required for getting some attributes of the dataset)
            - tar_patches_info (x, y, a): information of target patches mapped with cam id as key (for IP and OP
              only location (x, y) is used)
            - n_heads: number of heads in MVDeTr
            - n_points: number of pointers in each head in MVDeTr
            - downsample: scaling factor used on the reduced world
            - smoother (True/False): whether to use continuous smoother functions (i.e. exp and log)
        """
        super(InPtrLoss, self).__init__()
        self.dataset = dataset
        self.H, self.W = dataset.Rworld_shape
        self.H, self.W = int(self.H / downsample), int(self.W / downsample)
        self.N = dataset.num_cam
        self.tar_patches_info = deepcopy(tar_patches_info)
        self.n_heads, self.n_points = n_heads, n_points
        self.downsample = downsample
        self.smoother = smoother
        self.affine_mats = torch.eye(3).unsqueeze(0).unsqueeze(1).repeat(1, 7, 1, 1)
        self.proj = self.get_proj()
        self.attack_locations = self.generate_attack_locations()
        self.mse_loss = MSELoss()

    def get_proj(self):
        """
        This method returns the projection matrices for points in each cam to the reduced world
        Returns:
            - projection matrices from cams to world space
        """
        img_reduce, world_reduce = self.dataset.img_reduce, self.dataset.world_reduce
        mat1 = self.dataset.base.worldcoord_from_worldgrid_mat
        mat2 = np.diag([world_reduce * self.downsample, world_reduce * self.downsample, 1])
        mat3 = self.dataset.base.world_indexing_from_xy_mat
        mat4 = np.linalg.inv(mat1 @ mat2 @ mat3)
        intr = self.dataset.base.intrinsic_matrices
        extr = self.dataset.base.extrinsic_matrices
        mat5 = [get_worldcoord_from_imgcoord_mat(intr[cam], extr[cam], 0) for cam in range(self.N)]
        proj = torch.stack([torch.from_numpy(mat4 @ mat5[cam]) for cam in range(self.N)])
        mat6 = proj.repeat(1, 1, 1, 1).view(self.N, 3, 3).float()
        mat7 = torch.inverse(self.affine_mats.view([self.N, 3, 3]))
        # TODO: replace 8
        mat8 = torch.diag(torch.tensor([img_reduce / 8, img_reduce / 8, 1])).view(1, 3, 3).repeat(self.N, 1, 1).float()
        proj = mat6 @ mat7 @ mat8
        return proj

    @staticmethod
    def corner_pnts(patches_info):
        """
        This method returns the corner points of a patch given the patch information (x, y, a).
        Where (x, y) is the top-left location of the patch and a is the size of the patch.
        Args:
            - patches_info (x, y, a): information of patches mapped with cam id as key
        Returns:
            - information of corner points (x, y) of patches mapped with cam id as key
        """
        for i in range(len(patches_info)):
            for j in range(len(patches_info[i])):
                x, y, a = patches_info[i][j]
                patches_info[i][j] = [[x, y], [x, y + a], [x + a, y + a], [x + a, y]]
        return patches_info

    def clamp_pnts(self, patches_info):
        """
        This method clamps any point projected outside the frame to inside the frame
        Args:
            - patches_info (x, y): information of corner points of patches mapped with cam id as key
        Returns:
            - information of clamped corner points (x, y) of patches mapped with cam id as key
        """
        for i in range(len(patches_info)):
            for j in range(len(patches_info[i])):
                for k in range(len(patches_info[i][j])):
                    x, y = patches_info[i][j][k]
                    x, y = min(max(x, 0), self.W), min(max(y, 0), self.H)
                    patches_info[i][j][k] = [x, y]
        return patches_info

    @staticmethod
    def pad(lst, ref):
        """
        This method pads a list of list (so that each list has equal sized lists) so that it can be converted
        into a tensor
        """
        max_l = max(len(v) for v in ref.values())
        pad_v = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        for k in ref.keys():
            dis_l = len(lst[k])
            lst[k] = lst[k] + pad_v * (max_l - dis_l)
        return lst

    @staticmethod
    def unpad(lst, ref):
        """
        This method reverses the action of pad()
        """
        for k, v in ref.items():
            lst[k] = lst[k][:len(v)]
        return lst

    def convert_pnts(self, patches_info, proj):
        """
        This method takes the information of target patches, get their corner points, converts them into a tensor,
        transforms them into the world space, the clamps them within the world frame
        Args:
            - patches_info: information about the target patches mapped with cam id as key
            - proj: projection matrices
        Returns:
            - the transformed locations of the target patches in the world space
        """
        proj = proj.cuda()
        cams = list(patches_info.keys())
        pnts = list(patches_info.values())
        pnts = OutPtrLoss.corner_pnts(pnts)
        pnts = OutPtrLoss.pad(pnts, patches_info)
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        pnts = transform_points(proj[cams], pnts).tolist()
        pnts = OutPtrLoss.unpad(pnts, patches_info)
        pnts = self.clamp_pnts(pnts)
        pnts = dict(zip(cams, pnts))
        return pnts

    @staticmethod
    def get_centroid(patches_info):
        """
        This method takes the locations of the target patches in the world space and returns its centroid,
        which is the preferred attack location
        Args:
            - patches_info: the transformed locations of the target patches in the world space
        Returns:
            - the centroids of the target patches in the world space
        """
        for cam, pnts in patches_info.items():
            for i in range(len(pnts)):
                polygon = Polygon(pnts[i])
                centroid = polygon.centroid
                patches_info[cam][i] = [centroid.x, centroid.y]
        return patches_info

    def get_n_points(self, patches_info):
        """
        This method makes the size of the list of target patches equal to the size (number of pointers) of the heads
        Args:
            - patches_info: list of target patches having size less than or equal to the head
        Returns:
            - list of target patches having equal size to the head
        """
        for cam, pnts in patches_info.items():
            repeat = int(ceil(self.n_points / len(pnts)))
            pnts_rep = pnts * repeat
            pnts_rep = pnts_rep[:self.n_points]
            patches_info[cam] = pnts_rep
        return patches_info

    def generate_attack_locations(self):
        """
        This method projects the target patches on to the world space, get the centroid of the patches in the world
        space, makes the size of the list of target patches equal to the head, and then repeats it along
        the dimensions, to generate a corresponding tensor of preferred attack locations to redirect the pointers
        in a layer
        Returns:
            - preferred attack locations for pointers in a layer
        """
        pnts = self.convert_pnts(self.tar_patches_info, self.proj)
        pnts = OutPtrLoss.get_centroid(pnts)
        pnts = self.get_n_points(pnts)
        pnts = list(pnts.values())
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        pnts[:, :, 0] /= self.W
        pnts[:, :, 1] /= self.H
        pnts = pnts.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1, self.H * self.W * self.N, self.n_heads, 1, 1, 1)
        return pnts

    def forward(self, ptrs):
        """
        This forward method minimizes the MSE loss between the attack locations and the tip of the pointers
        Args:
            - ptrs: list of pointer tensors
        Returns:
            - cumulative pointer losses
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log
        ptrs_loss = 0
        for ptr in ptrs:
            head_loss = 0
            for head in range(ptr.shape[2]):
                loss = self.mse_loss(ptr[:, :, head, list(self.tar_patches_info.keys()), ...], self.attack_locations[:, :, head, ...])
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            ptrs_loss += f1(head_loss)
        ptrs_loss = f2(ptrs_loss)
        return ptrs_loss


class OutPtrLoss(nn.Module):
    def __init__(self, dataset, tar_patches_info, n_heads=8, n_points=4, downsample=2, smoother=False):
        super(OutPtrLoss, self).__init__()
        self.dataset = dataset
        self.H, self.W = dataset.Rworld_shape
        self.H, self.W = int(self.H / downsample), int(self.W / downsample)
        self.N = dataset.num_cam
        self.tar_patches_info = deepcopy(tar_patches_info)
        self.n_heads, self.n_points = n_heads, n_points
        self.downsample = downsample
        self.smoother = smoother
        self.affine_mats = torch.eye(3).unsqueeze(0).unsqueeze(1).repeat(1, 7, 1, 1)
        self.proj = self.get_proj()
        self.attack_locations = self.generate_attack_locations()
        self.mse_loss = MSELoss()

    def get_proj(self):
        img_reduce, world_reduce = self.dataset.img_reduce, self.dataset.world_reduce
        mat1 = self.dataset.base.worldcoord_from_worldgrid_mat
        mat2 = np.diag([world_reduce * self.downsample, world_reduce * self.downsample, 1])
        mat3 = self.dataset.base.world_indexing_from_xy_mat
        mat4 = np.linalg.inv(mat1 @ mat2 @ mat3)
        intr = self.dataset.base.intrinsic_matrices
        extr = self.dataset.base.extrinsic_matrices
        mat5 = [get_worldcoord_from_imgcoord_mat(intr[cam], extr[cam], 0) for cam in range(self.N)]
        proj = torch.stack([torch.from_numpy(mat4 @ mat5[cam]) for cam in range(self.N)])
        mat6 = proj.repeat(1, 1, 1, 1).view(self.N, 3, 3).float()
        mat7 = torch.inverse(self.affine_mats.view([self.N, 3, 3]))
        # TODO: replace 8
        mat8 = torch.diag(torch.tensor([img_reduce / 8, img_reduce / 8, 1])).view(1, 3, 3).repeat(self.N, 1, 1).float()
        proj = mat6 @ mat7 @ mat8
        return proj

    @staticmethod
    def corner_pnts(patches_info):
        for i in range(len(patches_info)):
            for j in range(len(patches_info[i])):
                x, y, a = patches_info[i][j]
                patches_info[i][j] = [[x, y], [x, y + a], [x + a, y + a], [x + a, y]]
        return patches_info

    def clamp_pnts(self, patches_info):
        for i in range(len(patches_info)):
            for j in range(len(patches_info[i])):
                for k in range(len(patches_info[i][j])):
                    x, y = patches_info[i][j][k]
                    x, y = min(max(x, 0), self.W), min(max(y, 0), self.H)
                    patches_info[i][j][k] = [x, y]
        return patches_info

    @staticmethod
    def pad(lst, ref):
        max_l = max(len(v) for v in ref.values())
        pad_v = [[[0, 0], [0, 0], [0, 0], [0, 0]]]
        for k in range(len(lst)):
            dis_l = len(lst[k])
            lst[k] = lst[k] + pad_v * (max_l - dis_l)
        return lst

    @staticmethod
    def unpad(lst, ref):
        for i, (k, v) in enumerate(ref.items()):
            lst[i] = lst[i][:len(v)]
        return lst

    def convert_pnts(self, patches_info, proj):
        proj = proj.cuda()
        cams = list(patches_info.keys())
        pnts = list(patches_info.values())
        pnts = OutPtrLoss.corner_pnts(pnts)
        pnts = OutPtrLoss.pad(pnts, patches_info)
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        pnts = transform_points(proj[cams], pnts).tolist()
        pnts = OutPtrLoss.unpad(pnts, patches_info)
        pnts = self.clamp_pnts(pnts)
        pnts = dict(zip(cams, pnts))
        return pnts

    @staticmethod
    def get_centroid(patches_info):
        for cam, pnts in patches_info.items():
            for i in range(len(pnts)):
                polygon = Polygon(pnts[i])
                centroid = polygon.centroid
                patches_info[cam][i] = [centroid.x, centroid.y]
        return patches_info

    def get_n_points(self, patches_info):
        for cam, pnts in patches_info.items():
            repeat = int(ceil(self.n_points / len(pnts)))
            pnts_rep = pnts * repeat
            pnts_rep = pnts_rep[:self.n_points]
            patches_info[cam] = pnts_rep
        return patches_info

    def generate_attack_locations(self):
        pnts = self.convert_pnts(self.tar_patches_info, self.proj)
        pnts = OutPtrLoss.get_centroid(pnts)
        pnts = self.get_n_points(pnts)
        pnts = list(pnts.values())
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        pnts[:, :, 0] /= self.W
        pnts[:, :, 1] /= self.H
        pnts = pnts.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(1, self.H * self.W * self.N, self.n_heads, 1, 1, 1)
        return pnts

    def forward(self, ptrs):
        """
        This forward method maximizes the MSE loss between the attack locations and the tip of the pointers
        Args:
            - ptrs: list of pointer tensors
        Returns:
            - cumulative pointer losses
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log
        ptrs_loss = 0
        for ptr in ptrs:
            head_loss = 0
            for head in range(ptr.shape[2]):
                loss = self.mse_loss(ptr[:, :, head, list(self.tar_patches_info.keys()), ...], self.attack_locations[:, :, head, ...])
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            ptrs_loss += f1(head_loss)
        ptrs_loss = f2(ptrs_loss)
        return -ptrs_loss


class AttLoss(nn.Module):
    def __init__(self, tar_patches_info, smoother):
        """
        Args:
            - tar_patches_info (x, y, a): information of target patches mapped with cam id as key (for IP and OP
              only location (x, y) is used)
            - smoother (True/False): whether to use continuous smoother functions (i.e. exp and log)
        """
        super(AttLoss, self).__init__()
        self.tar_patches_info = tar_patches_info
        self.smoother = smoother

    def forward(self, atts):
        """
        This forward method maximizes the attentions in the attention tensors
        Args:
            - atts: list of attention tensors
        Returns:
            - cumulative attention losses
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log
        atts_loss = 0
        for att in atts:
            head_loss = 0
            for head in range(att.shape[2]):
                loss = att[:, :, head, list(self.tar_patches_info.keys()), ...]
                loss = loss.mean()
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            atts_loss += f1(head_loss)
        atts_loss = f2(atts_loss)
        return -atts_loss


class AttDefMultiView:
    def __init__(self, output,
                 src_cams, src_nums, src_size, tar_cams, tar_nums, tar_size,
                 src_sampler, tar_sampler,
                 smoother, aggregator,
                 epsilon, constrainer,
                 layers,
                 dataset='wildtrack', learning_rate=0.22, step_size=10, gamma=0.95, downsample=1.5, batch_size=1):
        """
        Args:
            - output: name of the output folder
            - src_cams: list of cams hosting source patches
            - src_nums: list of number of source patches in each cam
            - src_size: size of source patches
            - tar_cams: list of cams hosting target patches
            - tar_nums: list of number of target patches in each cam
            - tar_size: size of target patches
            - src_sampler: location sampler for source patches
              (options:
                - uniforml (uniformly distributed on a line)
                - uniformg (uniformly distributed on a grid)
                - random)
            - tar sampler: location sampler for target patches
            - smoother (True/False): whether to use continuous smoother functions (i.e. exp and log)
            - aggregator: aggregator function used for source gradient aggregation
              (options:
                - mean (averaging the gradients)
                - norm (taking the gradient with the maximum l2 norm)
                - None (no aggregator, each source patch will be different))
            - epsilon: noise budget for the patches
            - constrainer: constrainer function used for patches
              (options:
                - l2 (l2 norm)
                - linf (l-infinity norm))
            - layers: number of layers used in the encoder and decoder
            - dataset: name of the dataset used
            - learning_rate: learning rate of the optimizer
            - step_size: step size of the scheduler
            - gamma: decay rate of the scheduler
            - downsample: scaling factor used on the original image (1920x1080 -> 1280x720)
            - batch_size: batch_size for training and testing
        """
        base = Wildtrack(os.path.expanduser(f'data/{dataset}'))

        self.num_cams = 7
        self.H, self.W = base.img_shape
        # optionally, images are downsampled from 1920x1080 to 1280x720 for memory constraints
        self.H, self.W = int(self.H / downsample), int(self.W / downsample)

        self.train_set = frameDataset(base)
        self.test_set = frameDataset(base, train=False)

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)

        # for 'wildtrack' dataset, model should be placed at model/wildtrack
        # number of layers should be put as a suffix to the model name (e.g. mvdetr_3)
        self.model_state_path = Path(f'model/{dataset}/mvdetr_{layers}.pth')
        self.model = MVDeTr(self.train_set, num_layers=layers, n_heads=8, n_points=4).cuda()
        self.model.load_state_dict(torch.load(self.model_state_path))

        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        self.downsample = downsample
        self.batch_size = batch_size

        self.src_cams, self.src_nums, self.src_size = src_cams, src_nums, src_size
        self.tar_cams, self.tar_nums, self.tar_size = tar_cams, tar_nums, tar_size
        self.smoother, self.aggregator = smoother, aggregator
        self.src_sampler, self.tar_sampler = src_sampler, tar_sampler
        self.epsilon, self.constrainer = epsilon, constrainer
        self.num_layers = layers

        self.output_dir = Path(f'logs/{dataset}/{output}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.src_info_dir = self.output_dir / Path('src_info.json')
        self.src_mask_dir = self.output_dir / Path('src_mask.pth')
        self.tar_info_dir = self.output_dir / Path('tar_info.json')
        self.tar_mask_dir = self.output_dir / Path('tar_mask.pth')

        # initialize/retrieve mask
        self.src_patches_info = self.get_patches_info((0, self.W), (0, self.H), self.src_cams, self.src_nums, self.src_size,
                                                      self.src_info_dir, sampler=self.src_sampler)
        self.tar_patches_info = self.get_patches_info((0, self.W), (0, self.H), self.tar_cams, self.tar_nums, self.tar_size,
                                                      self.tar_info_dir, sampler=self.tar_sampler)
        self.src_mask = self.get_mask(self.src_patches_info, self.src_mask_dir)
        self.tar_mask = self.get_mask(self.tar_patches_info, self.tar_mask_dir)
        print('!', end='')
        with open(self.src_info_dir, 'w') as file:
            json.dump(self.src_patches_info, file)
        with open(self.tar_info_dir, 'w') as file:
            json.dump(self.tar_patches_info, file)
        torch.save(self.src_mask, self.src_mask_dir)
        torch.save(self.tar_mask, self.tar_mask_dir)
        print('.', end='')

        self.in_ptr_loss = InPtrLoss(self.train_set, self.tar_patches_info, n_heads=8, n_points=4, smoother=self.smoother)
        self.out_ptr_loss = OutPtrLoss(self.train_set, self.tar_patches_info, n_heads=8, n_points=4, smoother=self.smoother)
        self.att_loss = AttLoss(self.tar_patches_info, smoother=self.smoother)
        self.trad_loss = TradLoss()

    @staticmethod
    def is_stop(modas, th):
        """
        This method stops the training if the Multi-object Detection Accuracy (MODA) reduces to 0.0 or
        the change in MODA is below a provided threshold value (eps) in the last 10 epochs
        Args:
            - aps: list of MODA in previous epochs
            - th: threshold value
        Returns:
            - True/False
        """
        if modas[-1] == 0:
            return True
        if modas[-1] == 0:
            return True
        if len(modas) < 10:
            return False
        for i in range(-10, 0):
            if abs(modas[i] - modas[-1]) > th:
                return False
        return True

    def train_adversarial(self, data, affine_mats):
        """
        This method takes in the adversarial data and generated outputs required for optimizing
        the adversarial patches. Wrapper method to test(), ensures eval() is called before.
        The MVDeTr model has been modified to return attentions and pointers in addition to the outputs...
        Args:
            - data: adversarial data injected with source and/or target patches
            - affine_mats: identity matrices
        Returns:
            - world_heatmap: the output of the model on adversarial data
            - atts: list of attention tensors
            - ptrs: list of pointer tensors
        """
        self.model.eval()
        (world_heatmap, _), _, (atts, ptrs) = self.model(data, affine_mats)
        return world_heatmap, atts, ptrs

    def test_adversarial(self, attack_fn, epoch, masked_delta, visualize):
        """
        This method takes in the masked perturbation, calls the modified test() method using the perturbation.
        Wrapper method to test(), ensures eval() is called before. Also, it appends the MODA to a file named
        after the attack method upto three decimal points...
        Args:
            - attack_fn: name of the attack
            - masked_delta (B x cams x C x H x W): masked perturbation
        Returns:
            - Multi-object Detection Accuracy (MODA)
        """
        attack_dir = self.output_dir / Path(f'{attack_fn}')
        attack_dir.mkdir(parents=True, exist_ok=True)

        imgs_dir = attack_dir / Path('imgs')
        imgs_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        trainer = PerspectiveTrainer(self.model, imgs_dir)

        tests_dir = attack_dir / Path(f'tests.txt')
        stats_dir = attack_dir / Path(f'stats.txt')

        _, moda = trainer.test(epoch, self.test_loader, tests_dir, args=[masked_delta, stats_dir], visualize=visualize)

        return moda

    @staticmethod
    def get_uniform_linear_pnts(x_rng, y_rng, n, sz):
        """
        This method generates uniform points in a line
        Args:
            - x_rng: range of x coordinates
            - y_rng: range of y coordinates
            - n: number of points
            - sz: padding size (so that all corners are within the above-mentioned ranges)
        Returns:
            - list of points as their top-left locations (x, y)
        """
        assert n <= 4, 'Can not return more than 4 points'
        pnts_x, pnts_y = torch.meshgrid(torch.linspace(x_rng[0] + sz // 2, x_rng[1] - sz // 2, n + 2, dtype=torch.int),
                                        torch.linspace(y_rng[0] + sz // 2, y_rng[1] - sz // 2, 7, dtype=torch.int))
        pnts_x, pnts_y = pnts_x - sz // 2, pnts_y - sz // 2
        pnts = torch.stack((pnts_x, pnts_y), -1)[1:-1, 1].reshape([-1, 2])
        pnts = pnts.tolist()
        return pnts

    @staticmethod
    def get_uniform_grid_pnts(x_rng, y_rng, n, sz):
        """
        This method generates uniform points in a grid
        Args:
            - x_rng: range of x coordinates
            - y_rng: range of y coordinates
            - n: number of points
            - sz: padding size (so that all corners are within the above-mentioned ranges)
        Returns:
            - list of points as their top-left locations (x, y)
        """
        assert n <= 4, 'Can not return more than 4 points'
        n_dict = {1: (3, 3), 2: (4, 3), 3: (5, 3), 4: (4, 4)}
        pnts_x, pnts_y = torch.meshgrid(torch.linspace(x_rng[0] + sz // 2, x_rng[1] - sz // 2, n_dict[n][0], dtype=torch.int),
                                        torch.linspace(y_rng[0] + sz // 2, y_rng[1] - sz // 2, n_dict[n][1], dtype=torch.int))
        pnts_x, pnts_y = pnts_x - sz // 2, pnts_y - sz // 2
        pnts = torch.stack((pnts_x, pnts_y), -1)[1:-1, 1:-1].reshape([-1, 2])
        pnts = pnts.tolist()
        return pnts

    @staticmethod
    def get_random_pnts(x_rng, y_rng, n, sz):
        """
        This method generates random points in a grid
        Args:
            - x_rng: range of x coordinates
            - y_rng: range of y coordinates
            - n: number of points
            - sz: padding size (so that all corners are within the above-mentioned ranges)
        Returns:
            - list of points as their top-left locations (x, y)
        """
        assert n <= 4, 'Can not return more than 4 points'
        pnts = []
        for i in range(n):
            pnt_x = random.randint(x_rng[0] + sz // 2, x_rng[1] - sz // 2 - 1) - sz // 2
            pnt_y = random.randint(y_rng[0] + sz // 2, y_rng[1] - sz // 2 - 1) - sz // 2
            pnts.append((pnt_x, pnt_y))
        return pnts

    @staticmethod
    def get_patches_info(x_rng, y_rng, cams, n, sz, path, sampler):
        """
        This method servers as an organizer that returns points based on the sampler. If previous points exists
        loads them instead.
        Args:
            - x_rng: range of x coordinates
            - y_rng: range of y coordinates
            - cams: list of cams
            - n: number of points
            - sz: patch size
            - path: load points from a file at this path if exists
            - sampler: location sampler for patches
              (options:
                - uniforml (uniformly distributed on a line)
                - uniformg (uniformly distributed on a grid)
                - random)
        Returns:
            - list of points as their top-left locations and patch sizes (x, y, a)
        """
        if path is not None and path.exists():
            with open(path, 'r') as file:
                patches_info = json.load(file)
                patches_info = {int(key): value for key, value in patches_info.items()}
            return patches_info
        patches_info = dict()
        for i, cam in enumerate(cams):
            if sampler == 'random':
                pnts = AttDefMultiView.get_random_pnts(x_rng, y_rng, n[i], sz)
            elif sampler == 'uniformg':
                pnts = AttDefMultiView.get_uniform_grid_pnts(x_rng, y_rng, n[i], sz)
            elif sampler == 'uniforml':
                pnts = AttDefMultiView.get_uniform_linear_pnts(x_rng, y_rng, n[i], sz)
            elif sampler == 'custom':
                pnts = [(100, 100)]
            else:
                raise ValueError('Sampler method not valid')
            pnts = [(x, y, sz) for x, y in pnts]
            patches_info[cam] = pnts
        return patches_info

    def get_mask(self, patches_info, path):
        """
        This method uses the patch information to generate a mask having equal shape of the samples that has
        1s where there is a patch and 0s elsewhere. If a previous mask exists loads it instead.
        Args:
            - patches_info (x, y, a): information of patches
            _ path: load patch from a file at this path if exists
        Returns:
            - mask tensor (B x cams x C x H x W)
        """
        if path is not None and path.exists():
            mask = torch.load(path)
            return mask.cuda()
        mask = torch.zeros((1, self.num_cams, 3, self.H, self.W))
        for cam, pnts in patches_info.items():
            for x, y, a in pnts:
                mask[:, cam, :, y:y+a, x:x+a] = torch.ones((1, 3, a, a))
        return mask.cuda()

    def get_opt(self, delta, opt_dir):
        """
        This method returns an optimizer for the perturbation frame delta. If the state dictionary of
        a previous optimizer exists loads it instead.
        Args:
            - delta: perturbation frame
            - opt_dir: load optimizer state from a file at this path if exists
        Returns:
            - optimizer for delta
        """
        opt = torch.optim.Adam([delta], lr=self.learning_rate)
        if opt_dir is not None and opt_dir.exists():
            opt.load_state_dict(torch.load(opt_dir))
        return opt

    def get_slr(self, opt, slr_dir):
        """
        This method returns a scheduler for the optimizer. If the state dictionary of a previous scheduler
        exists loads it instead.
        Args:
            - opt: optimizer
            - slr_dir: load scheduler state from a file at this path if exists
        Returns:
            - scheduler for opt
        """
        slr = torch.optim.lr_scheduler.StepLR(opt, step_size=self.step_size, gamma=self.gamma)
        if slr_dir is not None and slr_dir.exists():
            slr.load_state_dict(torch.load(slr_dir))
        return slr

    def get_delta(self, delta_dir):
        """
        This method returns a N(0, 1) tensor equal to the shape of samples. If a previous delta exists loads
        it instead.
        Args:
            delta_dir: load delta from a file at this path if exists
        Returns:
            - the perturbation frame (B x cams x C x H x W)
        """
        if delta_dir is not None and delta_dir.exists():
            delta = torch.load(delta_dir)
            return delta.cuda()
        delta = Normalize(torch.randn((1, self.num_cams, 3, self.H, self.W)))
        return delta.cuda()

    @staticmethod
    def constrain_delta(masked_delta, epsilon, constrainer):
        """
        This method servers as an organizer that constrains the l2 or l-infinity norm of the masked perturbation
        within the noise budget
        Args:
            - masked_delta: unconstrained masked perturbation
            - epsilon: noise budge
            - constrainer: constrainer function used for patches
              (options:
                - l2 (l2 norm)
                - linf (l-infinity norm))
        Returns:
            - constrained masked perturbation
        """
        if constrainer is None:
            return masked_delta
        elif constrainer == 'l2':
            norm = masked_delta.norm(p=2)
            if norm > epsilon:
                masked_delta = masked_delta * (epsilon / norm)
            return masked_delta
        elif constrainer == 'linf':
            masked_delta = torch.clamp(masked_delta, -epsilon, epsilon)
            return masked_delta
        else:
            raise ValueError('Constrainer method is not valid')

    @staticmethod
    def modify_grad(grad, src_patches_info, aggregator):
        """
        This method servers as an organizer that modifies the gradients at the source patches based on the aggregator
        Args:
            - grad: gradients with respect to the masked perturbation
            - src_patches_info (x, y, a): source patch information
            - aggregator: aggregator function used for source gradient aggregation
              (options:
                - mean (averaging the gradients)
                - norm (taking the gradient with the maximum l2 norm)
                - None (no aggregator, each source patch will be different))
        Returns:
            - modified gradients
        """
        if aggregator is None:
            return grad
        tnsr = []
        for cam, pnts in src_patches_info.items():
            for x, y, a in pnts:
                tnsr.append(grad[:, cam, :, y:y+a, x:x+a])
        if aggregator == 'mean':
            tnsr = torch.mean(torch.stack(tnsr), dim=0)
        elif aggregator == 'norm':
            tnsr = max(tnsr, key=lambda k: torch.norm(k, p=2).item())
        else:
            raise ValueError('Aggregator method not valid')
        for cam, pnts in src_patches_info.items():
            for x, y, a in pnts:
                grad[:, cam, :, y:y+a, x:x+a] = deepcopy(tnsr)
        return grad

    def IPAttack(self, attack_fn, epoch, visualize):
        """
        This method implements the inward pointer IP attack. In this attack, the source patches redirects all
        pointers to the target patches; except there is no patch (patch with all 0s). Effectively, converging
        the pointers of all the tokens/pixels to the target.
        Args:
            - attack_fn: name of the attack
            - epoch: running epoch
        """
        modas = []

        attack_dir = self.output_dir / Path(f'{attack_fn}')
        attack_dir.mkdir(parents=True, exist_ok=True)
        delta_ptr_att_dir = attack_dir / Path('delta_ptr_att.pth')
        opt_ptr_att_dir = attack_dir / Path('opt_ptr_att.pth')
        slr_ptr_att_dir = attack_dir / Path('slr_ptr_att.pth')

        # initialize/retrieve delta
        delta_ptr_att = self.get_delta(delta_ptr_att_dir)
        delta_ptr_att.requires_grad = True

        # initialize/retrieve optimizers
        opt_ptr_att = self.get_opt(delta_ptr_att, opt_ptr_att_dir)

        # initialize/retrieve schedulers
        slr_ptr_att = self.get_slr(opt_ptr_att, slr_ptr_att_dir)

        # gradient engineering
        pcgrad_ptr_att = PCGrad(opt_ptr_att)

        for ep in range(epoch):
            # save data
            data = None
            masked_delta = None
            for data, world_gt, imgs_gt, affine_mats, frame in tqdm(self.train_loader):
                # mode data to cuda
                data = data.cuda()

                # pre-process deltas
                masked_delta = delta_ptr_att * self.src_mask
                masked_delta = AttDefMultiView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                # add deltas to 'cams' views
                data = data + masked_delta

                # apply model
                world_heatmap, atts, ptrs = self.train_adversarial(data, affine_mats)

                # clear gradients
                self.model.zero_grad()
                opt_ptr_att.zero_grad()

                # calculate losses
                pl = self.in_ptr_loss(ptrs)
                al = self.att_loss(atts)

                # calculate gradients
                grad = pcgrad_ptr_att.pc_backward([pl, al])
                grad = AttDefMultiView.modify_grad(grad, self.src_patches_info, self.aggregator)
                pcgrad_ptr_att.update_grad(grad)

                # optimizers step
                opt_ptr_att.step()

            # schedulers step
            slr_ptr_att.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_ptr_att, delta_ptr_att_dir)
            torch.save(opt_ptr_att.state_dict(), opt_ptr_att_dir)
            torch.save(slr_ptr_att.state_dict(), slr_ptr_att_dir)
            print('.', end='')

            moda = self.test_adversarial(attack_fn, ep, masked_delta.clone(), visualize)
            modas.append(moda)

            if AttDefMultiView.is_stop(modas, 0.3):
                return

    def OPAttack(self, attack_fn, epoch, visualize):
        """
        This method implements the outward pointer OP attack. In this attack, the source patches redirects all
        pointers away from the target patches; except there is no patch (patch with all 0s). Effectively, diverging
        the pointers of all the tokens/pixels away from the target.
        """
        modas = []

        attack_dir = self.output_dir / Path(f'{attack_fn}')
        attack_dir.mkdir(parents=True, exist_ok=True)
        delta_ptr_att_dir = attack_dir / Path('delta_ptr_att.pth')
        opt_ptr_att_dir = attack_dir / Path('opt_ptr_att.pth')
        slr_ptr_att_dir = attack_dir / Path('slr_ptr_att.pth')

        # initialize/retrieve delta
        delta_ptr_att = self.get_delta(delta_ptr_att_dir)
        delta_ptr_att.requires_grad = True

        # initialize/retrieve optimizers
        opt_ptr_att = self.get_opt(delta_ptr_att, opt_ptr_att_dir)

        # initialize/retrieve schedulers
        slr_ptr_att = self.get_slr(opt_ptr_att, slr_ptr_att_dir)

        # gradient engineering
        pcgrad_ptr_att = PCGrad(opt_ptr_att)

        for ep in range(epoch):
            # save data
            data = None
            masked_delta = None
            for data, world_gt, imgs_gt, affine_mats, frame in tqdm(self.train_loader):
                # mode data to cuda
                data = data.cuda()

                # pre-process deltas
                masked_delta = delta_ptr_att * self.src_mask
                masked_delta = AttDefMultiView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                # add deltas to 'cams' views
                data = data + masked_delta

                # apply model
                world_heatmap, atts, ptrs = self.train_adversarial(data, affine_mats)

                # clear gradients
                self.model.zero_grad()
                opt_ptr_att.zero_grad()

                # calculate losses
                pl = self.out_ptr_loss(ptrs)
                al = self.att_loss(atts)

                # calculate gradients
                grad = pcgrad_ptr_att.pc_backward([pl, al])
                grad = AttDefMultiView.modify_grad(grad, self.src_patches_info, self.aggregator)
                pcgrad_ptr_att.update_grad(grad)

                # optimizers step
                opt_ptr_att.step()

            # schedulers step
            slr_ptr_att.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_ptr_att, delta_ptr_att_dir)
            torch.save(opt_ptr_att.state_dict(), opt_ptr_att_dir)
            torch.save(slr_ptr_att.state_dict(), slr_ptr_att_dir)
            print('.', end='')

            moda = self.test_adversarial(attack_fn, ep, masked_delta.clone(), visualize)
            modas.append(moda)

            if AttDefMultiView.is_stop(modas, 0.3):
                return

    def CPAttack(self, attack_fn, epoch, visualize):
        """
        This method implements the collaborative patch CP attack. In this attack, the source patches redirect all
        pointers to the target patches. Effectively, converging the pointers of all token/pixels to the target patches.
        The gradients from pointer and attention loss is propagated to the source patches and more traditional
        model loss is propagated to the target patches...
        """
        modas = []

        attack_dir = self.output_dir / Path(f'{attack_fn}')
        attack_dir.mkdir(parents=True, exist_ok=True)
        delta_ptr_att_dir = attack_dir / Path('delta_ptr_att.pth')
        opt_ptr_att_dir = attack_dir / Path('opt_ptr_att.pth')
        slr_ptr_att_dir = attack_dir / Path('slr_ptr_att.pth')
        delta_trad_dir = attack_dir / Path('delta_trad.pth')
        opt_trad_dir = attack_dir / Path('opt_trad.pth')
        slr_trad_dir = attack_dir / Path('slr_trad.pth')

        # initialize/retrieve delta
        delta_ptr_att = self.get_delta(delta_ptr_att_dir)
        delta_focal = self.get_delta(delta_trad_dir)
        delta_ptr_att.requires_grad, delta_focal.requires_grad = True, True

        # initialize/retrieve optimizers
        opt_ptr_att = self.get_opt(delta_ptr_att, opt_ptr_att_dir)
        opt_focal = self.get_opt(delta_focal, opt_trad_dir)

        # initialize/retrieve schedulers
        slr_ptr_att = self.get_slr(opt_ptr_att, slr_ptr_att_dir)
        slr_focal = self.get_slr(opt_focal, slr_trad_dir)

        # gradient engineering
        pcgrad_ptr_att, pcgrad_focal = PCGrad(opt_ptr_att), PCGrad(opt_focal)

        for ep in range(epoch):
            # save data
            data = None
            masked_delta = None
            for data, world_gt, imgs_gt, affine_mats, frame in tqdm(self.train_loader):
                # mode data to cuda
                data = data.cuda()

                # pre-process deltas
                masked_delta = delta_ptr_att * self.src_mask + delta_focal * self.tar_mask
                masked_delta = AttDefMultiView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                # add deltas to 'cams' views
                data = data + masked_delta

                # apply model
                world_heatmap, atts, ptrs = self.train_adversarial(data, affine_mats)

                # clear gradients
                self.model.zero_grad()
                opt_ptr_att.zero_grad()
                opt_focal.zero_grad()

                # calculate losses
                pl = self.in_ptr_loss(ptrs)
                al = self.att_loss(atts)
                tl = self.trad_loss(world_heatmap, world_gt['heatmap'])

                # calculate gradients
                grad = pcgrad_ptr_att.pc_backward([pl, al])
                grad = AttDefMultiView.modify_grad(grad, self.src_patches_info, self.aggregator)
                pcgrad_ptr_att.update_grad(grad)
                pcgrad_focal.update_grad(pcgrad_focal.pc_backward([tl]))

                # optimizers step
                opt_ptr_att.step()
                opt_focal.step()

            # schedulers step
            slr_ptr_att.step()
            slr_focal.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_ptr_att, delta_ptr_att_dir)
            torch.save(delta_focal, delta_trad_dir)
            torch.save(opt_ptr_att.state_dict(), opt_ptr_att_dir)
            torch.save(opt_focal.state_dict(), opt_trad_dir)
            torch.save(slr_ptr_att.state_dict(), slr_ptr_att_dir)
            torch.save(slr_focal.state_dict(), slr_trad_dir)
            print('.', end='')

            moda = self.test_adversarial(attack_fn, ep, masked_delta.clone(), visualize)
            modas.append(moda)

            if AttDefMultiView.is_stop(modas, 0.3):
                return

    def SPAttack(self, attack_fn, epoch, visualize):
        """
        This method implements the standalone patch SP attack. In this attack, the source patches move and merge
        with the target patches. Effectively, converging the pointers of all the token/pixels to itself.
        """
        modas = []

        attack_dir = self.output_dir / Path(f'{attack_fn}')
        attack_dir.mkdir(parents=True, exist_ok=True)
        delta_all_dir = attack_dir / Path('delta_all.pth')
        opt_all_dir = attack_dir / Path('opt_all.pth')
        slr_all_dir = attack_dir / Path('slr_all.pth')

        # initialize/retrieve delta
        delta_all = self.get_delta(delta_all_dir)
        delta_all.requires_grad = True

        # initialize/retrieve optimizers
        opt_all = self.get_opt(delta_all, opt_all_dir)

        # initialize/retrieve schedulers
        slr_all = self.get_slr(opt_all, slr_all_dir)

        # gradient engineering
        pcgrad_all = PCGrad(opt_all)

        for ep in range(epoch):
            # save data
            data = None
            masked_delta = None
            for data, world_gt, imgs_gt, affine_mats, frame in tqdm(self.train_loader):
                # mode data to cuda
                data = data.cuda()

                # pre-process deltas
                masked_delta = delta_all * self.tar_mask
                masked_delta = AttDefMultiView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                # add deltas to 'cams' views
                data = data + masked_delta

                # apply model
                world_heatmap, atts, ptrs = self.train_adversarial(data, affine_mats)

                # clear gradients
                self.model.zero_grad()
                opt_all.zero_grad()

                # calculate losses
                pl = self.in_ptr_loss(ptrs)
                al = self.att_loss(atts)
                tl = self.trad_loss(world_heatmap, world_gt['heatmap'])

                # calculate gradients
                grad = pcgrad_all.pc_backward([pl, al, tl])
                grad = AttDefMultiView.modify_grad(grad, self.tar_patches_info, self.aggregator)
                pcgrad_all.update_grad(grad)

                # optimizers step
                opt_all.step()

            # schedulers step
            slr_all.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_all, delta_all_dir)
            torch.save(opt_all.state_dict(), opt_all_dir)
            torch.save(slr_all.state_dict(), slr_all_dir)
            print('.', end='')

            moda = self.test_adversarial(attack_fn, ep, masked_delta.clone(), visualize)
            modas.append(moda)

            if AttDefMultiView.is_stop(modas, 0.3):
                return

    def clean(self, attack_fn, visualize):
        self.test_adversarial(attack_fn, 0, torch.zeros_like(self.get_delta(None)), visualize)


args = arguments.build_args()
att_def_single_view = AttDefMultiView(args.output,
                                      args.src_cams, args.src_nums, args.src_size,
                                      args.tar_cams, args.tar_nums, args.tar_size,
                                      args.src_sampler, args.tar_sampler,
                                      args.smoother, args.aggregator,
                                      args.epsilon, args.constrainer,
                                      args.layers)

if args.attack == 'IP':
    att_def_single_view.IPAttack('IP', args.epoch, args.visualize)
elif args.attack == 'OP':
    att_def_single_view.OPAttack('OP', args.epoch, args.visualize)
elif args.attack == 'CP':
    att_def_single_view.CPAttack('CP', args.epoch, args.visualize)
elif args.attack == 'SP':
    att_def_single_view.CPAttack('SP', args.epoch, args.visualize)
elif args.attack == 'Clean':
    att_def_single_view.clean('Clean', args.visualize)

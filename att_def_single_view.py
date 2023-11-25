import json
import random
from copy import deepcopy
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.nn import MSELoss
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, SequentialSampler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from AttDefDeTr import arguments
import AttDefDeTr.util.misc as utils
from AttDefDeTr.datasets import build_dataset, get_coco_api_from_dataset
from AttDefDeTr.datasets.coco_eval import CocoEvaluator
from AttDefDeTr.models import build_model
from AttDefDeTr.pcgrad.pcgrad import PCGrad
from PIL import ImageDraw
import colorsys


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.FloatTensor(mean).view([1, -1, 1, 1])
        self.std = torch.FloatTensor(std).view([1, -1, 1, 1])

    def __call__(self, tensor):
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


Norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
Denorm = Denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class InPtrLoss(nn.Module):
    def __init__(self, tar_resolutions, tar_patches_info, H, W, scaling_factors, num_queries, smoother):
        """
        Args:
            - tar_resolutions: index of the resolutions that will host target patches/points
            - tar_patches_info (x, y, a): information of target patches (for IP and OP only (x, y) is used)
            - H: height of the frame
            - W: width of the frame
            - scaling_factors: scaling factor for the resolutions
            - num_queries: number of object queries
            - smoother (True/False): whether to use continuous smoother functions (i.e. exp and log)
        """
        super(InPtrLoss, self).__init__()
        self.tar_resolutions = tar_resolutions
        self.tar_patches_info = tar_patches_info
        self.H, self.W = H, W
        self.scaling_factors = scaling_factors
        self.num_queries = num_queries
        self.smoother = smoother
        self.attack_locations = self.generate_attack_locations(tar_patches_info)
        self.mse_loss = MSELoss()

    @staticmethod
    def center_pnts(patches_info):
        """
        This method is used to get the center points of the patches, whether it is an IP/OP attack,
        or SP/CP attack pointers are redirected to the center points of the target patches.
        In the cases of IP/OP attack, one can think of placing all-0 patches...
        Args:
            - patches_info (x, y, a): information of patches
        Returns:
            - center points of the patches
        """
        ret = []
        for x, y, a in patches_info:
            ret.append(((2 * x + a) / 2, (2 * y + a) / 2))
        return ret

    def generate_attack_locations(self, patches_info):
        """
        This method receives the information of the patches, expands it to match the size of a head,
        and then repeats it along the dimensions, to generate a corresponding tensor of preferred
        attack locations to redirect the pointers in a layer
        Args:
            - patches_info (x, y, a): information of patches
        Returns:
            - preferred attack locations for pointers in a layer
        """
        pnts = InPtrLoss.center_pnts(patches_info)
        pnts = pnts * int(ceil(4 / len(pnts)))
        pnts = pnts[:4]
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        pnts[:, 0] /= self.W
        pnts[:, 1] /= self.H
        pnts = pnts.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        d1 = sum([(int(ceil(self.H / sf)) * int(ceil(self.W / sf))) for sf in self.scaling_factors])
        pnts = pnts.repeat(1, d1 + self.num_queries, 8, 4, 1, 1)
        return pnts

    def forward(self, ptrs_en, ptrs_de):
        """
        This forward method minimizes the MSE loss between the attack locations and the tip of the pointers
        in both the encoder and the decoder
        Args:
            - ptrs_en: list of pointer tensors of the encoder
            - ptrs_de: list of pointer tensor of the decoder
        Returns:
            - cumulative pointer losses of both the encoder and the decoder
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log

        ptrs_en_loss = 0
        for ptr in ptrs_en:
            head_loss = 0
            for head in range(ptr.shape[2]):
                loss = self.mse_loss(ptr[:, :, head, self.tar_resolutions, ...],
                                     self.attack_locations[:, :-self.num_queries, head, self.tar_resolutions, ...])
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            ptrs_en_loss += f1(head_loss)
        ptrs_en_loss = f2(ptrs_en_loss)

        ptrs_de_loss = 0
        for ptr in ptrs_de:
            head_loss = 0
            for head in range(ptr.shape[2]):
                loss = self.mse_loss(ptr[:, :, head, self.tar_resolutions, ...],
                                     self.attack_locations[:, -self.num_queries:, head, self.tar_resolutions, ...])
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            ptrs_de_loss += f1(head_loss)
        ptrs_de_loss = f2(ptrs_de_loss)

        return ptrs_en_loss + ptrs_de_loss


class OutPtrLoss(nn.Module):
    def __init__(self, tar_resolutions, tar_patches_info, H, W, scaling_factors, num_queries, smoother):
        super(OutPtrLoss, self).__init__()
        self.tar_resolutions = tar_resolutions
        self.tar_patches_info = tar_patches_info
        self.H, self.W = H, W
        self.scaling_factors = scaling_factors
        self.num_queries = num_queries
        self.smoother = smoother
        self.attack_locations = self.generate_attack_locations(tar_patches_info)
        self.mse_loss = MSELoss()

    @staticmethod
    def center_pnts(patches_info):
        ret = []
        for x, y, a in patches_info:
            ret.append(((2 * x + a) / 2, (2 * y + a) / 2))
        return ret

    def generate_attack_locations(self, patches_info):
        pnts = OutPtrLoss.center_pnts(patches_info)
        pnts = pnts * int(ceil(4 / len(pnts)))
        pnts = pnts[:4]
        pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
        pnts[:, 0] /= self.W
        pnts[:, 1] /= self.H
        pnts = pnts.unsqueeze(0).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        d1 = sum([(int(ceil(self.H / sf)) * int(ceil(self.W / sf))) for sf in self.scaling_factors])
        pnts = pnts.repeat(1, d1 + self.num_queries, 8, 4, 1, 1)
        return pnts

    def forward(self, ptrs_en, ptrs_de):
        """
        This forward method maximizes the MSE loss between the attack locations and the tip of the pointers
        in both the encoder and decoder
        Args:
            - ptrs_en: list of pointer tensors of the encoder
            - ptrs_de: list of pointer tensor of the decoder
        Returns:
            - cumulative pointer losses of both the encoder and the decoder
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log

        ptrs_en_loss = 0
        for ptr in ptrs_en:
            head_loss = 0
            for head in range(ptr.shape[2]):
                loss = self.mse_loss(ptr[:, :, head, self.tar_resolutions, ...],
                                     self.attack_locations[:, :-self.num_queries, head, self.tar_resolutions, ...])
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            ptrs_en_loss += f1(head_loss)
        ptrs_en_loss = f2(ptrs_en_loss)

        ptrs_de_loss = 0
        for ptr in ptrs_de:
            head_loss = 0
            for head in range(ptr.shape[2]):
                loss = self.mse_loss(ptr[:, :, head, self.tar_resolutions, ...],
                                     self.attack_locations[:, -self.num_queries:, head, self.tar_resolutions, ...])
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            ptrs_de_loss += f1(head_loss)
        ptrs_de_loss = f2(ptrs_de_loss)

        return -(ptrs_en_loss + ptrs_de_loss)


class DenseAttLoss(nn.Module):
    def __init__(self, tar_patches_info, H, W, scaling_factors, smoother):
        """
        Args:
        - tar_patches (x, y, a): information of target patches (for IP and OP only location (x, y) is used)
        - H: height of the frame
        - W: width of the frame
        - scaling_factors: scaling factor for the resolutions
        - smoother (True/False): whether to use continuous smoother functions (i.e. exp and log)
        """
        super(DenseAttLoss, self).__init__()
        self.tar_patches_info = tar_patches_info
        self.H = H
        self.W = W
        self.scaling_factors = scaling_factors
        self.smoother = smoother

    def get_idx(self, ptrs):
        """
        This method returns the indices of the pointer tensor of a layer that has its tip within one of
        the target patches
        Args:
            - ptrs: pointer tensor of a layer
        Returns:
            - indices of the pointer tensor that are within the target patches
        """
        d1 = sum([(int(ceil(self.H / sf)) * int(ceil(self.W / sf))) for sf in self.scaling_factors])
        idx = torch.zeros((1, d1, 4, 4), dtype=torch.bool).cuda()
        for x, y, a in self.tar_patches_info:
            x1 = x / self.W
            y1 = y / self.H
            x2 = (x + a) / self.W
            y2 = (y + a) / self.H
            pnts = [(x1, y1), (x2, y2)]
            pnts = torch.tensor(pnts, dtype=torch.float32).cuda()
            idx_i = (ptrs >= pnts[0]) & (ptrs <= pnts[1])
            # noinspection PyTypeChecker
            idx_i = torch.all(idx_i, dim=-1)
            idx |= idx_i
        return idx

    def forward(self, ptrs_en, atts_en):
        """
        This forward method determines which pointers in the pointer tensor have their tips within the
        target patches and maximizes their corresponding attentions in the attention tensor.
        Only the pointer and attention tensors in the encoder in considered as described in the previous literature...
        Args:
            - ptrs_en: list of pointer tensors of the encoder
            - atts_de: list of pointer tensor of the encoder
        Returns:
            - cumulative attention losses of the encoder
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log

        atts_loss = 0
        for ptr, att in zip(ptrs_en, atts_en):
            head_loss = 0
            for head in range(att.shape[2]):
                loss = att[:, :, head, ...][self.get_idx(ptr[:, :, head, ...])]
                loss = loss.mean()
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            atts_loss += f1(head_loss)
        atts_loss = f2(atts_loss)
        return -atts_loss


class SparseAttLoss(nn.Module):
    def __init__(self, tar_resolutions, smoother):
        """
        Args:
            - tar_resolutions: index of the resolutions that will host target patches/points
            - smoother (True/False): whether to use continuous smoother functions (i.e. exp and log)
        """
        super(SparseAttLoss, self).__init__()
        self.tar_resolutions = tar_resolutions
        self.smoother = smoother

    def forward(self, atts_en, atts_de):
        """
        This forward method maximizes the attentions in the attention tensors of both encoder and decoder
        Args:
            - atts_en: list of attention tensors of the encoder
            - atts_de: list of attention tensors of the decoder
        Returns:
            - cumulative attention losses of the encoder and decoder
        """
        f1, f2 = lambda x: x, lambda x: x
        if self.smoother:
            f1, f2 = torch.exp, torch.log

        atts_en_loss = 0
        for att in atts_en:
            head_loss = 0
            for head in range(att.shape[2]):
                loss = att[:, :, head, self.tar_resolutions, ...]
                loss = loss.mean()
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            atts_en_loss += f1(head_loss)
        atts_en_loss = f2(atts_en_loss)

        atts_de_loss = 0
        for att in atts_de:
            head_loss = 0
            for head in range(att.shape[2]):
                loss = att[:, :, head, self.tar_resolutions, ...]
                loss = loss.mean()
                head_loss += f1(loss)
            head_loss = f2(head_loss)
            atts_de_loss += f1(head_loss)
        atts_de_loss = f2(atts_de_loss)

        return -(atts_en_loss + atts_de_loss)


class AttDefSingleView:
    def __init__(self, output, args,
                 src_num, src_size, tar_num, tar_size,
                 src_sampler, tar_sampler,
                 smoother, aggregator,
                 epsilon, constrainer,
                 layers,
                 dataset='coco', learning_rate=0.22, step_size=10, gamma=0.95, num_categories=91):
        """
        Args:
            - output: name of the output folder
            - args: command line arguments
            - src_num: number of source patches
            - src_size: size of source patches
            - tar_num: number of target patches
            - tar_size: size of target patches
            - src_sampler: location sampler for source patches
              (options:
                - uniforml (uniformly distributed on a line)
                - uniformg (uniformly distributed on a grid)
                - random)
            - tar_sampler: location sampler for target patches
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
            - num_categories: total number of object categories
        """
        args.enc_layers, args.dec_layers = layers, layers
        utils.init_distributed_mode(args)

        self.dataset_train = build_dataset(image_set='train', args=args)
        self.dataset_val = build_dataset(image_set='val', args=args)

        sampler_train = RandomSampler(self.dataset_train)
        batch_sampler_train = BatchSampler(sampler_train, args.batch_size, drop_last=True)
        self.data_loader_train = DataLoader(self.dataset_train, batch_sampler=batch_sampler_train,
                                            collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

        sampler_val = SequentialSampler(self.dataset_val)
        batch_sampler_val = BatchSampler(sampler_val, args.batch_size, drop_last=True)
        self.data_loader_val = DataLoader(self.dataset_val, batch_sampler=batch_sampler_val,
                                          collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True)

        self.model, self.criterion, self.postprocessors = build_model(args)
        self.model.cuda()

        # for 'ms coco' dataset, model should be placed at model/coco/
        # number of layers should be put as suffix to the model name (e.g, detr_6)
        self.model_state_path = Path(f'model/{dataset}/detr_{layers}.pth')
        checkpoint = torch.load(self.model_state_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)

        self.learning_rate, self.step_size, self.gamma = learning_rate, step_size, gamma

        # images should be resized to a particular height x width
        self.B, self.C, self.H, self.W = args.batch_size, 3, args.img_size, args.img_size
        # resolution are indexed as 0, 1, 2, 3...
        # scaling factor starts from 2^3, reducing by half in lower resolutions
        # this is with respect to the model configuration at training time
        self.scaling_factors = [pow(2, 3), pow(2, 4), pow(2, 5), pow(2, 6)]
        self.num_queries = args.num_queries

        self.src_num, self.src_size = src_num, src_size
        self.src_sampler = src_sampler
        self.tar_num, self.tar_size = tar_num, tar_size
        self.tar_sampler = tar_sampler
        self.smoother, self.aggregator = smoother, aggregator
        self.epsilon, self.constrainer = epsilon, constrainer

        self.output_dir = Path(f'logs/{dataset}/{output}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.src_info_dir = self.output_dir / Path('src_info.json')
        self.src_mask_dir = self.output_dir / Path('src_mask.pth')
        self.tar_info_dir = self.output_dir / Path('tar_info.json')
        self.tar_mask_dir = self.output_dir / Path('tar_mask.pth')

        self.src_patches_info = self.get_patches_info((0, self.W), (0, self.H), self.src_num, self.src_size,
                                                      self.src_info_dir, self.src_sampler)
        self.src_mask = self.get_mask(self.src_patches_info, self.src_mask_dir)
        self.tar_patches_info = self.get_patches_info((0, self.W), (0, self.H), self.tar_num, self.tar_size,
                                                      self.tar_info_dir, self.tar_sampler)
        self.tar_mask = self.get_mask(self.tar_patches_info, self.tar_mask_dir)

        # patch information (locations, sizes and masks are saved for restarting training
        print('!', end='')
        with open(self.src_info_dir, 'w') as file:
            json.dump(self.src_patches_info, file)
        with open(self.tar_info_dir, 'w') as file:
            json.dump(self.tar_patches_info, file)
        torch.save(self.src_mask, self.src_mask_dir)
        torch.save(self.tar_mask, self.tar_mask_dir)
        print('.', end='')

        self.colors = self.generate_colors(num_categories)

        self.in_ptr_loss = InPtrLoss(range(4), self.tar_patches_info, self.H, self.W, self.scaling_factors,
                                     self.num_queries, self.smoother)
        self.out_ptr_loss = OutPtrLoss(range(4), self.tar_patches_info, self.H, self.W, self.scaling_factors,
                                       self.num_queries, self.smoother)
        self.dense_att_loss = DenseAttLoss(self.tar_patches_info, self.H, self.W, self.scaling_factors, self.smoother)
        self.sparse_att_loss = SparseAttLoss(range(4), self.smoother)

    @staticmethod
    def is_stop(aps, th):
        """
        This method stops the training if the Average Precision (AP) reduces to 0.0 or the change in AP is below
        a provided threshold value (eps) in the last 10 epochs
        Args:
            - aps: list of AP in previous epochs
            - th: threshold value
        Returns:
            - True/False
        """
        if aps[-1] == 0:
            return True
        if len(aps) < 10:
            return False
        for i in range(-10, 0):
            if abs(aps[i] - aps[-1]) > th:
                return False
        return True

    @staticmethod
    def generate_colors(num_categories):
        """
        Generate colors of bounding-boxes for different object categoreis
        Args:
            - num_categories: total number of object categoreis
        Returns:
            - list of RGB colors
        """
        hsv_tuples = [(x / num_categories, 1.0, 1.0) for x in range(num_categories)]
        rgb_tuples = list(map(lambda x: tuple(int(y * 255) for y in colorsys.hsv_to_rgb(*x)), hsv_tuples))
        return rgb_tuples

    def train_adversarial(self, samples):
        """
        This method accepts the adversarial samples and generates outputs required for optimizing
        the adversarial patches. Wrapper method for model(), ensures eval() is called before.
        The DeTr model has been modified to return attentions and pointers in addition to the outputs...
        Args:
            - samples: adversarial samples injected with source and/or target patches
        Returns:
            - outputs: the output of the model on adversarial samples
            - atts_en: list of attention tensors of the encoder
            - ptrs_en: list of pointer tensors of the encoder
            - atts_de: list of attention tensors of the decoder
            - ptrs_de: list of pointer tensors of the decoder
        """
        self.model.eval()
        self.criterion.eval()
        outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.model(samples)
        return outputs, atts_en, ptrs_en, atts_de, ptrs_de

    @staticmethod
    def save_model_parameters(epoch, ptrs_en, ptrs_de, atts_en, atts_de, attack_dir, samples_id):
        attack_dir_it_ep = attack_dir / Path(f'parameters/{epoch}/{samples_id}')
        attack_dir_it_ep.mkdir(parents=True, exist_ok=True)
        ptrs_en_dir = attack_dir_it_ep / Path('ptrs_en.pth')
        ptrs_de_dir = attack_dir_it_ep / Path('ptrs_de.pth')
        atts_en_dir = attack_dir_it_ep / Path('atts_en.pth')
        atts_de_dir = attack_dir_it_ep / Path('atts_de.pth')
        torch.save(ptrs_en, ptrs_en_dir)
        torch.save(ptrs_de, ptrs_de_dir)
        torch.save(atts_en, atts_en_dir)
        torch.save(atts_de, atts_de_dir)

    def save_annotated_imgs(self, epoch, coco_evaluator, targets, before, after, attack_dir, samples_id):
        attack_dir_imgs = attack_dir / Path('imgs/annotations/')
        attack_dir_imgs.mkdir(parents=True, exist_ok=True)
        before_dir = attack_dir / Path('before')
        after_dir = attack_dir / Path('after')
        gt_dir = attack_dir_imgs / Path('gt')
        dt_dir = attack_dir_imgs / Path('dt')
        before_dir.mkdir(parents=True, exist_ok=True)
        after_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        dt_dir.mkdir(parents=True, exist_ok=True)

        img_id = targets[0]['image_id'].item()

        size = targets[0]['size']
        orig_size = targets[0]['orig_size']
        y_scale = size[0] / orig_size[0]
        x_scale = size[1] / orig_size[1]

        bef = to_pil_image(Denorm(before).squeeze(0))
        img_gt = to_pil_image(Denorm(before).squeeze(0))
        draw_gt = ImageDraw.Draw(img_gt)
        coco_gt = coco_evaluator.coco_eval['bbox'].cocoGt
        ann_gt = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))

        for annotation in ann_gt:
            bbox = annotation['bbox']
            color = self.colors[annotation['category_id']]
            x0, y0 = bbox[0], bbox[1]
            x1, y1 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            x0, x1 = x0 * x_scale, x1 * x_scale
            y0, y1 = y0 * y_scale, y1 * y_scale
            draw_gt.rectangle((x0, y0, x1, y1), outline=color, width=5)

        aft = to_pil_image(Denorm(after).squeeze(0))
        img_dt = to_pil_image(Denorm(after).squeeze(0))
        draw_dt = ImageDraw.Draw(img_dt)
        coco_dt = coco_evaluator.coco_eval['bbox'].cocoDt
        ann_dt = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))

        for annotation in ann_dt:
            if annotation['score'] > 0.35:
                bbox = annotation['bbox']
                color = self.colors[annotation['category_id']]
                x0, y0 = bbox[0], bbox[1]
                x1, y1 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                x0, x1 = x0 * x_scale, x1 * x_scale
                y0, y1 = y0 * y_scale, y1 * y_scale
                draw_dt.rectangle((x0, y0, x1, y1), outline=color, width=5)

        if epoch == 0:
            bef.save(before_dir / Path(f'{samples_id}.png'))
            aft.save(after_dir / Path(f'{samples_id}.png'))
        img_gt.save(gt_dir / Path(f'{samples_id}.png'))
        img_dt.save(dt_dir / Path(f'{samples_id}.png'))

    def test_adversarial(self, epoch, attack_fn, masked_delta, num_batches_val, visualize):
        """
        This method takes in the masked perturbation, adds it to the samples, and generates the
        Average Precision (AP) achieved using the perturbed samples. Also, it appends the AP to
        a file named after attack method upto three decimal points.
        If visualize is True, it also saves the original image, and the attention and pointer tensors
        to a folder named after the attack method...
        Args:
            - epoch: running epoch
            - attack_fn: name of the attack
            - masked_delta (3 x height x width): masked perturbation
            - num_batches_val: batch size used for testing
            - visualize (True/False): whether to save pointer and attention tensors for visualization
        Returns:
            - Average Precision (AP)
        """
        attack_dir = self.output_dir / Path(f'{attack_fn}')
        attack_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self.criterion.eval()

        iou_types = tuple(k for k in ('segm', 'bbox') if k in self.postprocessors.keys())
        base_ds = get_coco_api_from_dataset(self.dataset_val)
        coco_evaluator = CocoEvaluator(base_ds, iou_types)

        with tqdm(total=num_batches_val) as progress_bar:
            for samples_id, (samples, targets) in enumerate(self.data_loader_val):
                # move data to device
                samples = samples.to(torch.device('cuda'))
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                # apply delta to data
                before = samples.tensors.clone()
                samples.tensors = samples.tensors + masked_delta
                after = samples.tensors.clone()

                # apply model
                outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.model(samples)

                # visualize
                if visualize:
                    ep = epoch if epoch in (0, 1) else 2
                    self.save_model_parameters(ep, ptrs_en, ptrs_de, atts_en, atts_de, attack_dir, samples_id)

                # update coco evaluator
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = self.postprocessors['bbox'](outputs, orig_target_sizes)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                coco_evaluator.update(res)

                # visualize
                if visualize:
                    self.save_annotated_imgs(epoch, coco_evaluator, targets, before, after, attack_dir, samples_id)

                # update progress_bar
                progress_bar.update(1)

                if samples_id + 1 >= num_batches_val:
                    break

        # accumulate results
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()

        # print ap info
        coco_evaluator.summarize()

        # write in file
        stats_dir = self.output_dir / Path(f'{attack_fn}/stats.txt')
        stats = coco_evaluator.coco_eval['bbox'].stats
        with open(stats_dir, 'a') as file:
            np.savetxt(file, stats, fmt='%.3f', newline=' ')
            file.write('\n')
        ap = stats[0]
        return ap

    def traditional_loss(self, outputs, targets):
        """
        This loss function maximizes the model loss by comparing the outputs generated using the adversarial samples
        and the ground-truth targets
        Args:
            - outputs: the outputs generated using adversarial samples
            - targets: the ground-truth targets
        Returns:
            - model loss
        """
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        return -losses

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
        pnts_x, pnts_y = torch.meshgrid(
            torch.linspace(x_rng[0] + sz // 2, x_rng[1] - sz // 2, n + 2, dtype=torch.int),
            torch.linspace(y_rng[0] + sz // 2, y_rng[1] - sz // 2, 6, dtype=torch.int))
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
        pnts_x, pnts_y = torch.meshgrid(
            torch.linspace(x_rng[0] + sz // 2, x_rng[1] - sz // 2, n_dict[n][0], dtype=torch.int),
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
    def get_patches_info(x_rng, y_rng, n, sz, path, sampler):
        """
        This method servers as an organizer that returns points based on the sampler. If previous points exists
        loads them instead.
        Args:
            - x_rng: range of x coordinates
            - y_rng: range of y coordinates
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
            return patches_info
        if sampler == 'random':
            patches_info = AttDefSingleView.get_random_pnts(x_rng, y_rng, n, sz)
        elif sampler == 'uniformg':
            patches_info = AttDefSingleView.get_uniform_grid_pnts(x_rng, y_rng, n, sz)
        elif sampler == 'uniforml':
            patches_info = AttDefSingleView.get_uniform_linear_pnts(x_rng, y_rng, n, sz)
        elif sampler == 'custom':
            patches_info = [(100, 100)]
        else:
            raise ValueError('Sampler function not valid')
        patches_info = [(x, y, sz) for x, y in patches_info]
        return patches_info

    def get_mask(self, patches_info, path):
        """
        This method uses the patch information to generate a mask having equal shape of the samples that has
        1s where there is a patch and 0s elsewhere. If a previous mask exists loads it instead.
        Args:
            - patches info (x, y, a): information of patches
            _ path: load patch from a file at this path if exists
        Returns:
            - mask tensor (B x C x H x W)
        """
        if path is not None and path.exists():
            mask = torch.load(path)
            return mask.cuda()
        mask = torch.zeros((self.B, self.C, self.H, self.W))
        for x, y, a in patches_info:
            mask[:, :, y:y+a, x:x+a] = torch.ones((self.B, self.C, a, a))
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
            - the perturbation frame (B x C x H x W)
        """
        if delta_dir is not None and delta_dir.exists():
            delta = torch.load(delta_dir)
            return delta.cuda()
        delta = Norm(torch.randn((self.B, self.C, self.H, self.W)))
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
            raise ValueError('Constrainer function is not valid')

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
        for x, y, a in src_patches_info:
            tnsr.append(grad[:, :, y:y+a, x:x+a])
        if aggregator == 'mean':
            tnsr = torch.mean(torch.stack(tnsr), dim=0)
        elif aggregator == 'norm':
            tnsr = max(tnsr, key=lambda k: torch.norm(k, p=2).item())
        else:
            raise ValueError('Aggregator function not valid')
        for x, y, a in src_patches_info:
            grad[:, :, y:y+a, x:x+a] = deepcopy(tnsr)
        return grad

    def IPAttack(self, attack_fn, epoch, num_batches_train, num_batches_val, visualize=False):
        """
        This method implements the inward pointer IP attack. In this attack, the source patches redirects all
        pointers to the target patches; except there is no patch (patch with all 0s). Effectively, converging
        the pointers of all the tokens/pixels to the target.
        Args:
            - attack_fn: name of the attack
            - epoch: running epoch
            - num_batches_train: batch size used for training
            - num_batches_val: batch size used for testing
            - visualize (True/False): whether to save pointer and attention tensors for visualization
        """
        aps = []

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
            masked_delta = None
            with tqdm(total=num_batches_train) as progress_bar:
                for iteration, (samples, targets) in enumerate(self.data_loader_train):
                    # move data to device
                    samples = samples.to(torch.device('cuda'))

                    # pre-process deltas
                    masked_delta = delta_ptr_att * self.src_mask
                    masked_delta = AttDefSingleView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                    # apply delta to data
                    samples.tensors = samples.tensors + masked_delta

                    # apply model
                    outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.train_adversarial(samples)

                    # clear gradients
                    self.model.zero_grad()
                    opt_ptr_att.zero_grad()

                    # calculate losses
                    pl = self.in_ptr_loss(ptrs_en, ptrs_de)
                    al = self.sparse_att_loss(atts_en, atts_de)

                    # calculate gradients
                    grad = pcgrad_ptr_att.pc_backward([pl, al])
                    grad = AttDefSingleView.modify_grad(grad, self.src_patches_info, self.aggregator)
                    pcgrad_ptr_att.update_grad(grad)

                    # optimizer step
                    opt_ptr_att.step()

                    # update progress bar
                    progress_bar.update(1)

                    if iteration + 1 >= num_batches_train:
                        break

            # scheduler step
            slr_ptr_att.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_ptr_att, delta_ptr_att_dir)
            torch.save(opt_ptr_att.state_dict(), opt_ptr_att_dir)
            torch.save(slr_ptr_att.state_dict(), slr_ptr_att_dir)
            print('.', end='')

            ap = self.test_adversarial(ep, attack_fn, masked_delta.clone(), num_batches_val, visualize)
            aps.append(ap)

            if AttDefSingleView.is_stop(aps, 0.003):
                return

    def OPAttack(self, attack_fn, epoch, num_batches_train, num_batches_val, visualize=False):
        """
        This method implements the outward pointer OP attack. In this attack, the source patches redirects all
        pointers away from the target patches; except there is no patch (patch with all 0s). Effectively, diverging
        the pointers of all the tokens/pixels away from the target.
        """
        aps = []

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
            masked_delta = None
            with tqdm(total=num_batches_train) as progress_bar:
                for iteration, (samples, targets) in enumerate(self.data_loader_train):
                    # move data to device
                    samples = samples.to(torch.device('cuda'))

                    # pre=process deltas
                    masked_delta = delta_ptr_att * self.src_mask
                    masked_delta = AttDefSingleView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                    # apply delta to data
                    samples.tensors = samples.tensors + masked_delta

                    # apply model
                    outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.train_adversarial(samples)

                    # clear gradients
                    self.model.zero_grad()
                    opt_ptr_att.zero_grad()

                    # calculate losses
                    pl = self.out_ptr_loss(ptrs_en, ptrs_de)
                    al = self.sparse_att_loss(atts_en, atts_de)

                    # calculate gradients
                    grad = pcgrad_ptr_att.pc_backward([pl, al])
                    grad = AttDefSingleView.modify_grad(grad, self.src_patches_info, self.aggregator)
                    pcgrad_ptr_att.update_grad(grad)

                    # optimizer step
                    opt_ptr_att.step()

                    # update progress bar
                    progress_bar.update(1)

                    if iteration + 1 >= num_batches_train:
                        break

            # scheduler step
            slr_ptr_att.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_ptr_att, delta_ptr_att_dir)
            torch.save(opt_ptr_att.state_dict(), opt_ptr_att_dir)
            torch.save(slr_ptr_att.state_dict(), slr_ptr_att_dir)
            print('.', end='')

            ap = self.test_adversarial(ep, attack_fn, masked_delta.clone(), num_batches_val, visualize)
            aps.append(ap)

            if AttDefSingleView.is_stop(aps, 0.003):
                return

    def CPAttack(self, attack_fn, epoch, num_batches_train, num_batches_val, visualize=False):
        """
        This method implements the collaborative patch CP attack. In this attack, the source patches redirect all
        pointers to the target patches. Effectively, converging the pointers of all token/pixels to the target patches.
        The gradients from pointer and attention loss is propagated to the source patches and more traditional
        model loss is propagated to the target patches...
        """
        aps = []

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
        delta_trad = self.get_delta(delta_trad_dir)
        delta_ptr_att.requires_grad, delta_trad.requires_grad = True, True

        # initialize/retrieve optimizers
        opt_ptr_att = self.get_opt(delta_ptr_att, opt_ptr_att_dir)
        opt_trad = self.get_opt(delta_trad, opt_trad_dir)

        # initialize/retrieve schedulers
        slr_ptr_att = self.get_slr(opt_ptr_att, slr_ptr_att_dir)
        slr_trad = self.get_slr(opt_trad, slr_trad_dir)

        # gradient engineering
        pcgrad_ptr_att, pcgrad_trad = PCGrad(opt_ptr_att), PCGrad(opt_trad)

        for ep in range(epoch):
            masked_delta = None
            with tqdm(total=num_batches_train) as progress_bar:
                for iteration, (samples, targets) in enumerate(self.data_loader_train):
                    # move data to device
                    samples = samples.to(torch.device('cuda'))
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                    # pre=process deltas
                    masked_delta = delta_ptr_att * self.src_mask + delta_trad * self.tar_mask
                    masked_delta = AttDefSingleView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                    # apply delta to data
                    samples.tensors = samples.tensors + masked_delta

                    # apply model
                    outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.train_adversarial(samples)

                    # clear gradients
                    self.model.zero_grad()
                    opt_ptr_att.zero_grad()
                    opt_trad.zero_grad()

                    # calculate losses
                    pl = self.in_ptr_loss(ptrs_en, ptrs_de)
                    al = self.sparse_att_loss(atts_en, atts_de)
                    tl = self.traditional_loss(outputs, targets)

                    # calculate gradients
                    grad = pcgrad_ptr_att.pc_backward([pl, al])
                    grad = AttDefSingleView.modify_grad(grad, self.src_patches_info, self.aggregator)
                    pcgrad_ptr_att.update_grad(grad)
                    pcgrad_trad.update_grad(pcgrad_trad.pc_backward([tl]))

                    # optimizer step
                    opt_ptr_att.step()
                    opt_trad.step()

                    # update progress bar
                    progress_bar.update(1)

                    if iteration + 1 >= num_batches_train:
                        break

            # scheduler step
            slr_ptr_att.step()
            slr_trad.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_ptr_att, delta_ptr_att_dir)
            torch.save(opt_ptr_att.state_dict(), opt_ptr_att_dir)
            torch.save(slr_ptr_att.state_dict(), slr_ptr_att_dir)
            torch.save(delta_trad, delta_trad_dir)
            torch.save(opt_trad.state_dict(), opt_trad_dir)
            torch.save(slr_trad.state_dict(), slr_trad_dir)
            print('.', end='')

            ap = self.test_adversarial(ep, attack_fn, masked_delta.clone(), num_batches_val, visualize)
            aps.append(ap)

            if AttDefSingleView.is_stop(aps, 0.003):
                return

    def SPAttack(self, attack_fn, epoch, num_batches_train, num_batches_val, visualize=False):
        """
        This method implements the standalone patch SP attack. In this attack, the source patches more and merge
        with the target patches. Effectively, converging the pointers of all the tokens/pixels to itself.
        """
        aps = []

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
            masked_delta = None
            with tqdm(total=num_batches_train) as progress_bar:
                for iteration, (samples, targets) in enumerate(self.data_loader_train):
                    # move data to device
                    samples = samples.to(torch.device('cuda'))
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                    # pre=process deltas
                    masked_delta = delta_all * self.tar_mask
                    masked_delta = AttDefSingleView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                    # apply delta to data
                    samples.tensors = samples.tensors + masked_delta

                    # apply model
                    outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.train_adversarial(samples)

                    # clear gradients
                    self.model.zero_grad()
                    opt_all.zero_grad()

                    # calculate losses
                    pl = self.in_ptr_loss(ptrs_en, ptrs_de)
                    al = self.sparse_att_loss(atts_en, atts_de)
                    tl = self.traditional_loss(outputs, targets)

                    # calculate gradients
                    grad = pcgrad_all.pc_backward([pl, al, tl])
                    grad = AttDefSingleView.modify_grad(grad, self.tar_patches_info, self.aggregator)
                    pcgrad_all.update_grad(grad)

                    # optimizer step
                    opt_all.step()

                    # update progress bar
                    progress_bar.update(1)

                    if iteration + 1 >= num_batches_train:
                        break

            # scheduler step
            slr_all.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_all, delta_all_dir)
            torch.save(opt_all.state_dict(), opt_all_dir)
            torch.save(slr_all.state_dict(), slr_all_dir)
            print('.', end='')

            ap = self.test_adversarial(ep, attack_fn, masked_delta.clone(), num_batches_val, visualize)
            aps.append(ap)

            if AttDefSingleView.is_stop(aps, 0.003):
                return

    def AttAttack(self, attack_fn, epoch, num_batches_train, num_batches_val, visualize=False):
        """
        This method implements the previous attention based attacks (Attention-Fool and Patch-Fool)
        """
        aps = []

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
            masked_delta = None
            with tqdm(total=num_batches_train) as progress_bar:
                for iteration, (samples, targets) in enumerate(self.data_loader_train):
                    # move data to device
                    samples = samples.to(torch.device('cuda'))
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

                    # pre=process deltas
                    masked_delta = delta_all * self.tar_mask
                    masked_delta = AttDefSingleView.constrain_delta(masked_delta, self.epsilon, self.constrainer)

                    # apply delta to data
                    samples.tensors = samples.tensors + masked_delta

                    # apply model
                    outputs, atts_en, ptrs_en, atts_de, ptrs_de = self.train_adversarial(samples)

                    # clear gradients
                    self.model.zero_grad()
                    opt_all.zero_grad()

                    # calculate losses
                    al = self.dense_att_loss(ptrs_en, atts_en)
                    tl = self.traditional_loss(outputs, targets)

                    # calculate gradients
                    grad = pcgrad_all.pc_backward([al, tl])
                    grad = AttDefSingleView.modify_grad(grad, self.tar_patches_info, self.aggregator)
                    pcgrad_all.update_grad(grad)

                    # optimizer step
                    opt_all.step()

                    # update progress bar
                    progress_bar.update(1)

                    if iteration + 1 >= num_batches_train:
                        break

            # scheduler step
            slr_all.step()

            # save data, delta, opt, slr
            print('!', end='')
            torch.save(delta_all, delta_all_dir)
            torch.save(opt_all, opt_all_dir)
            torch.save(slr_all, slr_all_dir)
            print('.', end='')

            ap = self.test_adversarial(ep, attack_fn, masked_delta.clone(), num_batches_val, visualize)
            aps.append(ap)

            if AttDefSingleView.is_stop(aps, 0.003):
                return

    def clean(self, attack_fn, num_batches_val, visualize=False):
        """
        This method implements the clean case (no adversarial patch applied)
        """
        self.test_adversarial(0, attack_fn, torch.zeros_like(self.get_delta(None)), num_batches_val, visualize)


args = arguments.build_args()
att_def_single_view = AttDefSingleView(args.output, args,
                                       args.src_num, args.src_size,
                                       args.tar_num, args.tar_size,
                                       args.src_sampler, args.tar_sampler,
                                       args.smoother, args.aggregator,
                                       args.epsilon, args.constrainer,
                                       args.layers)

if args.attack == 'IP':
    att_def_single_view.IPAttack('IP', args.epoch, args.num_batches_train, args.num_batches_val, args.visualize)
elif args.attack == 'OP':
    att_def_single_view.OPAttack('OP', args.epoch, args.num_batches_train, args.num_batches_val, args.visualize)
elif args.attack == 'CP':
    att_def_single_view.CPAttack('CP', args.epoch, args.num_batches_train, args.num_batches_val, args.visualize)
elif args.attack == 'SP':
    att_def_single_view.CPAttack('SP', args.epoch, args.num_batches_train, args.num_batches_val, args.visualize)
elif args.attack == 'Att':
    att_def_single_view.AttAttack('Att', args.epoch, args.num_batches_train, args.num_batches_val, args.visualize)
elif args.attack == 'Clean':
    att_def_single_view.clean('Clean', args.num_batches_val, args.visualize)

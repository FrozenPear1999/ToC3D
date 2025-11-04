# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------

import csv
import json
import os

import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
from PIL import Image


def _load_lidar_points(filename):
    points = np.fromfile(filename, dtype=np.float32)
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    points = points.reshape(-1, 5)[:, :3]
    return points


@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
    
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()


        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.training and self.with_2d: # sync_2d bbox labels
                gt_bboxes = results['gt_bboxes'][i]
                centers2d = results['centers2d'][i]
                gt_labels = results['gt_labels'][i]
                depths = results['depths'][i]
                if len(gt_bboxes) != 0:
                    gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                        gt_bboxes, 
                        centers2d,
                        gt_labels,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                    )
                if len(gt_bboxes) != 0 and self.filter_invisible:
                    gt_bboxes, centers2d, gt_labels, depths =  self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths)

                new_gt_bboxes.append(gt_bboxes)
                new_centers2d.append(centers2d)
                new_gt_labels.append(gt_labels)
                new_depths.append(depths)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]

        return results

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths,resize, crop, flip):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths



    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )  

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results['ego_pose_inv'] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()


    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()

class CoverageAccumulator:
    """Running statistics for LiDAR token coverage."""

    def __init__(self, view_count, num_bins):
        self.view_count = view_count
        self.num_bins = num_bins
        self.sample_count = 0
        self.view_sum = np.zeros(view_count, dtype=np.float64)
        self.view_sq_sum = np.zeros(view_count, dtype=np.float64)
        self.total_sum = 0.0
        self.total_sq_sum = 0.0
        self.view_hist_sum = np.zeros((view_count, num_bins), dtype=np.float64)
        self.total_hist_sum = np.zeros(num_bins, dtype=np.float64)
        self.view_token_sum = np.zeros(view_count, dtype=np.float64)
        self.total_token_sum = 0.0

    def update(self, coverages, total_coverage, view_hist_counts,
               total_hist_counts, view_token_counts, total_tokens):
        coverages = np.asarray(coverages, dtype=np.float64)
        view_hist_counts = np.asarray(view_hist_counts, dtype=np.float64)
        self.sample_count += 1
        self.view_sum += coverages
        self.view_sq_sum += np.square(coverages)
        self.total_sum += float(total_coverage)
        self.total_sq_sum += float(total_coverage) ** 2
        self.view_hist_sum += view_hist_counts
        self.total_hist_sum += np.asarray(total_hist_counts, dtype=np.float64)
        self.view_token_sum += np.asarray(view_token_counts, dtype=np.float64)
        self.total_token_sum += float(total_tokens)

    def _mean_std(self, sum_vals, sq_sum_vals):
        if self.sample_count == 0:
            zeros = np.zeros_like(sum_vals)
            return zeros, zeros
        mean = sum_vals / self.sample_count
        var = np.maximum(sq_sum_vals / self.sample_count - np.square(mean), 0.0)
        return mean, np.sqrt(var)

    def summarise(self, bin_labels):
        mean_view, std_view = self._mean_std(self.view_sum, self.view_sq_sum)
        total_mean, total_std = self._mean_std(
            np.asarray([self.total_sum], dtype=np.float64),
            np.asarray([self.total_sq_sum], dtype=np.float64)
        )

        view_hist = []
        for idx in range(self.view_count):
            tokens = self.view_token_sum[idx]
            if tokens > 0:
                view_hist.append((self.view_hist_sum[idx] / tokens).tolist())
            else:
                view_hist.append([0.0] * self.num_bins)

        if self.total_token_sum > 0:
            total_hist = (self.total_hist_sum / self.total_token_sum).tolist()
        else:
            total_hist = [0.0] * self.num_bins

        return {
            'num_samples': int(self.sample_count),
            'coverage_per_view_mean': mean_view.tolist(),
            'coverage_per_view_std': std_view.tolist(),
            'coverage_total_mean': float(total_mean[0]),
            'coverage_total_std': float(total_std[0]),
            'histogram_bins': bin_labels,
            'histogram_per_view': view_hist,
            'histogram_total': total_hist,
            'tokens_per_view': self.view_token_sum.tolist(),
            'tokens_total': float(self.total_token_sum)
        }


class LidarCoverageMonitor:
    """Collect LiDAR token coverage metrics without affecting the pipeline."""

    BIN_LABELS = ['0', '1', '2-3', '4-7', '8-15', '16+']

    def __init__(self, config, patch_size):
        config = config or {}
        self.enabled = bool(config.get('enable', False))
        if not self.enabled:
            self.view_names = None
            return

        self.patch_size = patch_size
        self.log_every = int(config.get('log_every', 50))
        self.save_mask_debug = bool(config.get('save_mask_debug', False))
        self.debug_max_samples = int(config.get('debug_max_samples', 5))
        self.output_dir = config.get('output_dir', 'work_dirs/lidar_coverage')
        os.makedirs(self.output_dir, exist_ok=True)
        self.masks_dir = os.path.join(self.output_dir, 'masks')
        if self.save_mask_debug:
            os.makedirs(self.masks_dir, exist_ok=True)

        self.csv_path = os.path.join(self.output_dir, 'coverage.csv')
        self.csv_file = None
        self.csv_writer = None

        self.view_names = None
        self.global_accumulator = None
        self.resolution_accumulators = {}
        self.sample_counter = 0
        self.debug_saved = 0

    def _ensure_writer(self, view_count):
        if self.csv_writer is not None:
            return
        fieldnames = ['scene_token', 'sample_token', 'height', 'width', 'patch']
        fieldnames += [f'cov_v{i}' for i in range(view_count)]
        fieldnames.append('cov_total')
        fieldnames += [f'hit_v{i}' for i in range(view_count)]
        fieldnames += [f'total_v{i}' for i in range(view_count)]
        fieldnames.append('hit_total')
        fieldnames.append('total_total')
        file_exists = os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(self.csv_path) == 0:
            self.csv_writer.writeheader()

    def _compute_histogram(self, patch_counts):
        counts = np.asarray(patch_counts, dtype=np.float64)
        hist = np.zeros(len(self.BIN_LABELS), dtype=np.float64)
        hist[0] = np.sum(counts == 0)
        hist[1] = np.sum(counts == 1)
        hist[2] = np.sum((counts >= 2) & (counts <= 3))
        hist[3] = np.sum((counts >= 4) & (counts <= 7))
        hist[4] = np.sum((counts >= 8) & (counts <= 15))
        hist[5] = np.sum(counts >= 16)
        return hist

    def _ensure_accumulators(self, view_count):
        if self.global_accumulator is None:
            self.global_accumulator = CoverageAccumulator(
                view_count, len(self.BIN_LABELS))

    def _resolution_accumulator(self, view_count, height, width):
        key = (int(height), int(width), int(self.patch_size))
        if key not in self.resolution_accumulators:
            self.resolution_accumulators[key] = CoverageAccumulator(
                view_count, len(self.BIN_LABELS))
        return key, self.resolution_accumulators[key]

    def _maybe_save_masks(self, meta, view_stats, view_names):
        if not self.save_mask_debug or self.debug_saved >= self.debug_max_samples:
            return
        sample_token = meta.get('sample_token', f'sample_{self.sample_counter}')
        for idx, stats in enumerate(view_stats):
            filename = f"{sample_token}_view{view_names[idx]}.npz"
            np.savez_compressed(
                os.path.join(self.masks_dir, filename),
                hit_mask=stats['hit_mask'].astype(np.uint8),
                valid_mask=stats['valid_mask'].astype(np.uint8),
                patch_counts=stats['patch_counts'].astype(np.float32)
            )
        self.debug_saved += 1

    def record_sample(self, meta, view_stats, patch_size, height, width, view_names):
        if not self.enabled:
            return

        view_count = len(view_stats)
        self._ensure_writer(view_count)
        self._ensure_accumulators(view_count)
        _, res_acc = self._resolution_accumulator(view_count, height, width)
        if self.view_names is None:
            self.view_names = list(view_names)

        coverages = np.zeros(view_count, dtype=np.float64)
        hits = np.zeros(view_count, dtype=np.float64)
        totals = np.zeros(view_count, dtype=np.float64)
        view_hists = np.zeros((view_count, len(self.BIN_LABELS)), dtype=np.float64)
        total_hist = np.zeros(len(self.BIN_LABELS), dtype=np.float64)

        for idx, stats in enumerate(view_stats):
            valid_mask = stats['valid_mask']
            patch_counts = stats['patch_counts']
            hit_mask = stats['hit_mask'] & valid_mask
            valid_tokens = float(valid_mask.sum())
            hit_tokens = float(hit_mask.sum())
            coverages[idx] = hit_tokens / valid_tokens if valid_tokens > 0 else 0.0
            hits[idx] = hit_tokens
            totals[idx] = valid_tokens
            hist = self._compute_histogram(patch_counts[valid_mask])
            view_hists[idx] = hist
            total_hist += hist

        total_tokens = float(totals.sum())
        total_hits = float(hits.sum())
        total_coverage = total_hits / total_tokens if total_tokens > 0 else 0.0

        self.global_accumulator.update(coverages, total_coverage, view_hists,
                                       total_hist, totals, total_tokens)
        res_acc.update(coverages, total_coverage, view_hists,
                       total_hist, totals, total_tokens)

        row = {
            'scene_token': meta.get('scene_token'),
            'sample_token': meta.get('sample_token'),
            'height': int(height),
            'width': int(width),
            'patch': int(patch_size),
            'cov_total': total_coverage,
            'hit_total': int(total_hits),
            'total_total': int(total_tokens)
        }
        for idx in range(view_count):
            row[f'cov_v{idx}'] = coverages[idx]
            row[f'hit_v{idx}'] = int(hits[idx])
            row[f'total_v{idx}'] = int(totals[idx])
        self.csv_writer.writerow(row)
        self.csv_file.flush()

        self.sample_counter += 1
        if self.log_every > 0 and self.sample_counter % self.log_every == 0:
            coverage_str = ', '.join(f'{cov:.3f}' for cov in coverages.tolist())
            print(
                f"LidarCoverage | view={self.view_names}: [{coverage_str}] "
                f"total={total_coverage:.3f} | res={int(height)}x{int(width)} p={patch_size}"
            )

        self._maybe_save_masks(meta, view_stats, view_names)

    def finalize(self):
        if not self.enabled:
            return
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        if self.global_accumulator is None:
            return

        summary = {
            'view_names': self.view_names,
            'bin_labels': self.BIN_LABELS,
            'overall': self.global_accumulator.summarise(self.BIN_LABELS),
            'per_resolution': {}
        }
        for (height, width, patch), acc in self.resolution_accumulators.items():
            key = f'{height}x{width}_p{patch}'
            summary['per_resolution'][key] = acc.summarise(self.BIN_LABELS)

        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)


@PIPELINES.register_module()
class ComputeLidarTokenPrior:
    """Compute LiDAR-guided priors for image tokens.

    The pipeline projects LiDAR points into each camera view, rasterises
    per-pixel depth and density statistics, and finally aggregates them to the
    ViT patch grid. The resulting tensor follows the same ordering as the
    image tokens (view-major, row-major) and is normalised to :math:`[0, 1]` per
    camera.
    """
    def __init__(self,
                 patch_size=16,
                 depth_weight=1.0,
                 density_weight=1.0,
                 depth_epsilon=1e-3,
                 coverage=None):
        self.patch_size = patch_size
        self.depth_weight = depth_weight
        self.density_weight = density_weight
        self.depth_epsilon = depth_epsilon
        self.coverage_monitor = LidarCoverageMonitor(coverage, patch_size)

    def __call__(self, results):
        if 'img' not in results or 'lidar2img' not in results or 'pts_filename' not in results:
            return results

        points = _load_lidar_points(results['pts_filename'])
        if points.shape[0] > 0:
            points_hom = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        else:
            points_hom = None

        monitor_enabled = hasattr(self, 'coverage_monitor') and self.coverage_monitor.enabled
        orig_shapes = results.get('img_shape') if monitor_enabled else None
        view_names = self._infer_view_names(results.get('img_filename'), len(results['img']))
        sample_meta = {
            'scene_token': results.get('scene_token'),
            'sample_token': results.get('sample_idx') or results.get('token')
        }

        priors = []
        view_stats = []
        view_heights = []
        view_widths = []
        for view_idx, img in enumerate(results['img']):
            height, width = img.shape[0], img.shape[1]
            lidar2img = results['lidar2img'][view_idx]
            view_heights.append(height)
            view_widths.append(width)

            if points_hom is None:
                depth_map = np.zeros((height, width), dtype=np.float32)
                density_map = np.zeros((height, width), dtype=np.float32)
            else:
                depth_map, density_map = self._project(points_hom, lidar2img, height, width)

            valid_shape = None
            if orig_shapes is not None and view_idx < len(orig_shapes):
                orig_shape = orig_shapes[view_idx]
                if isinstance(orig_shape, (list, tuple)) and len(orig_shape) >= 2:
                    valid_shape = (int(orig_shape[0]), int(orig_shape[1]))
                elif hasattr(orig_shape, '__len__') and len(orig_shape) >= 2:
                    valid_shape = (int(orig_shape[0]), int(orig_shape[1]))

            prior, stats = self._aggregate(
                depth_map,
                density_map,
                valid_shape=valid_shape,
                collect_stats=monitor_enabled
            )
            priors.append(prior)
            if monitor_enabled and stats is not None:
                view_stats.append(stats)

        results['lidar_token_prior'] = np.stack(priors).astype(np.float32)

        if monitor_enabled and view_stats:
            height0 = view_heights[0] if view_heights else 0
            width0 = view_widths[0] if view_widths else 0
            self.coverage_monitor.record_sample(
                sample_meta,
                view_stats,
                self.patch_size,
                height0,
                width0,
                view_names
            )

        return results

    def _empty_prior(self, height, width):
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f'Image resolution ({height}, {width}) is not divisible by '
                f'patch_size={self.patch_size}. Please pad or resize inputs '
                'to align with the ViT patch embedding.'
            )

        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        return np.zeros((h_tokens, w_tokens), dtype=np.float32)

    def _project(self, points_hom, lidar2img, height, width):
        lidar2img = np.asarray(lidar2img)
        proj = points_hom @ lidar2img.T
        depth = proj[:, 2]
        valid = depth > self.depth_epsilon
        if not np.any(valid):
            depth_map = np.zeros((height, width), dtype=np.float32)
            density_map = np.zeros((height, width), dtype=np.float32)
            return depth_map, density_map

        proj = proj[valid]
        depth = depth[valid]
        u = proj[:, 0] / depth
        v = proj[:, 1] / depth

        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)

        valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not np.any(valid):
            depth_map = np.zeros((height, width), dtype=np.float32)
            density_map = np.zeros((height, width), dtype=np.float32)
            return depth_map, density_map

        u = u[valid]
        v = v[valid]
        depth = depth[valid]
        linear_idx = v * width + u

        depth_map = np.full((height * width,), np.inf, dtype=np.float32)
        density_map = np.zeros((height * width,), dtype=np.float32)

        np.minimum.at(depth_map, linear_idx, depth)
        np.add.at(density_map, linear_idx, 1)

        depth_map = depth_map.reshape(height, width)
        density_map = density_map.reshape(height, width)
        depth_map[~np.isfinite(depth_map)] = 0.0
        return depth_map, density_map

    def _aggregate(self, depth_map, density_map, valid_shape=None, collect_stats=False):
        height, width = depth_map.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f'Image resolution ({height}, {width}) is not divisible by '
                f'patch_size={self.patch_size}. Please pad inputs accordingly.'
            )

        h_tokens = height // self.patch_size
        w_tokens = width // self.patch_size
        priors = np.zeros((h_tokens, w_tokens), dtype=np.float32)
        density_vals = np.zeros_like(priors)
        inv_depth_vals = np.zeros_like(priors)
        patch_counts = np.zeros((h_tokens, w_tokens), dtype=np.float32)

        for h in range(h_tokens):
            h_start = h * self.patch_size
            h_end = h_start + self.patch_size
            for w in range(w_tokens):
                w_start = w * self.patch_size
                w_end = w_start + self.patch_size

                patch_depth = depth_map[h_start:h_end, w_start:w_end]
                patch_density = density_map[h_start:h_end, w_start:w_end]

                valid_depth = patch_depth[patch_depth > 0]
                if valid_depth.size > 0:
                    depth_med = np.median(valid_depth)
                    inv_depth_vals[h, w] = 1.0 / max(depth_med, self.depth_epsilon)
                else:
                    inv_depth_vals[h, w] = 0.0

                count = patch_density.sum()
                patch_counts[h, w] = count
                density_vals[h, w] = count / (self.patch_size * self.patch_size)

        density_min, density_max = density_vals.min(), density_vals.max()
        if density_max > density_min:
            density_norm = (density_vals - density_min) / (density_max - density_min)
        else:
            density_norm = np.zeros_like(density_vals)

        raw_score = self.depth_weight * inv_depth_vals + self.density_weight * density_norm
        raw_min, raw_max = raw_score.min(), raw_score.max()
        if raw_max > raw_min:
            priors = (raw_score - raw_min) / (raw_max - raw_min)
        else:
            priors = np.zeros_like(raw_score)

        stats = None
        if collect_stats:
            valid_mask = self._valid_patch_mask(h_tokens, w_tokens, valid_shape)
            stats = {
                'hit_mask': patch_counts > 0,
                'patch_counts': patch_counts,
                'valid_mask': valid_mask
            }

        return priors.astype(np.float32), stats

    def _valid_patch_mask(self, h_tokens, w_tokens, valid_shape):
        if valid_shape is None:
            return np.ones((h_tokens, w_tokens), dtype=bool)

        valid_h, valid_w = valid_shape
        mask = np.zeros((h_tokens, w_tokens), dtype=bool)
        for h in range(h_tokens):
            h_start = h * self.patch_size
            if h_start >= valid_h:
                break
            for w in range(w_tokens):
                w_start = w * self.patch_size
                if w_start >= valid_w:
                    break
                mask[h, w] = True
        return mask

    def _infer_view_names(self, filenames, count):
        if not isinstance(filenames, (list, tuple)) or len(filenames) != count:
            return [f'v{idx}' for idx in range(count)]

        names = []
        for idx, path in enumerate(filenames):
            name = f'v{idx}'
            if isinstance(path, str):
                norm = os.path.normpath(path)
                parts = norm.split(os.sep)
                candidate = None
                for part in reversed(parts):
                    upper = part.upper()
                    if upper.startswith('CAM_'):
                        candidate = part
                        break
                if candidate is None and parts:
                    candidate = os.path.splitext(parts[-1])[0]
                if candidate:
                    name = candidate
            names.append(name)
        return names

    def close(self):
        if hasattr(self, 'coverage_monitor') and self.coverage_monitor.enabled:
            self.coverage_monitor.finalize()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

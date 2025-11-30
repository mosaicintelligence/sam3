# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import numpy as np
import pycocotools.mask as mask_utils
import torch

import torchvision.transforms.functional as F
from PIL import Image as PILImage

from sam3.model.box_ops import masks_to_boxes

from sam3.train.data.sam3_image_dataset import Datapoint


class InstanceToSemantic(object):
    """Convert instance segmentation to semantic segmentation."""

    def __init__(self, delete_instance=True, use_rle=False):
        self.delete_instance = delete_instance
        self.use_rle = use_rle

    def __call__(self, datapoint: Datapoint, **kwargs):
        for fquery in datapoint.find_queries:
            h, w = datapoint.images[fquery.image_id].size

            if self.use_rle:
                all_segs = [
                    datapoint.images[fquery.image_id].objects[obj_id].segment
                    for obj_id in fquery.object_ids_output
                ]
                if len(all_segs) > 0:
                    # we need to double check that all rles are the correct size
                    # Otherwise cocotools will fail silently to an empty [0,0] mask
                    for seg in all_segs:
                        assert seg["size"] == all_segs[0]["size"], (
                            "Instance segments have inconsistent sizes. "
                            f"Found sizes {seg['size']} and {all_segs[0]['size']}"
                        )
                    fquery.semantic_target = mask_utils.merge(all_segs)
                else:
                    # There is no good way to create an empty RLE of the correct size
                    # We resort to converting an empty box to RLE
                    fquery.semantic_target = mask_utils.frPyObjects(
                        np.array([[0, 0, 0, 0]], dtype=np.float64), h, w
                    )[0]

            else:
                # `semantic_target` is uint8 and remains uint8 throughout the transforms
                # (it contains binary 0 and 1 values just like `segment` for each object)
                fquery.semantic_target = torch.zeros((h, w), dtype=torch.uint8)
                for obj_id in fquery.object_ids_output:
                    segment = datapoint.images[fquery.image_id].objects[obj_id].segment
                    if segment is not None:
                        assert (
                            isinstance(segment, torch.Tensor)
                            and segment.dtype == torch.uint8
                        )
                        fquery.semantic_target |= segment

        if self.delete_instance:
            for img in datapoint.images:
                for obj in img.objects:
                    del obj.segment
                    obj.segment = None

        return datapoint


class RecomputeBoxesFromMasks:
    """Recompute bounding boxes from masks."""

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            for obj in img.objects:
                # Note: if the mask is empty, the bounding box will be undefined
                # The empty targets should be subsequently filtered
                obj.bbox = masks_to_boxes(obj.segment)
                obj.area = obj.segment.sum().item()

        return datapoint


class DecodeRle:
    """This transform decodes RLEs into binary segments.
    Implementing it as a transform allows lazy loading. Some transforms (e.g., query filters)
    may be deleting masks, so decoding them from the beginning is wasteful.

    This transform needs to be called before any kind of geometric manipulation.
    """

    def __call__(self, datapoint: Datapoint, **kwargs):
        imgId2size = {}
        warning_shown = False
        for imgId, img in enumerate(datapoint.images):
            if isinstance(img.data, PILImage.Image):
                img_w, img_h = img.data.size
            elif isinstance(img.data, torch.Tensor):
                img_w, img_h = img.data.shape[-2:]
            else:
                raise RuntimeError(f"Unexpected image type {type(img.data)}")

            imgId2size[imgId] = (img_h, img_w)

            for obj in img.objects:
                if obj.segment is not None and not isinstance(obj.segment, torch.Tensor):
                    # Handle multiple input formats:
                    # - COCO RLE dict (compressed or uncompressed)
                    # - COCO polygons list (each polygon is [x0,y0,...])
                    # - numpy uint8 array
                    seg = obj.segment
                    decoded_np = None

                    try:
                        if isinstance(seg, dict):
                            # RLE dict
                            if mask_utils.area(seg) == 0:
                                print("Warning, empty mask found, approximating from box")
                                decoded_np = np.zeros((img_h, img_w), dtype=np.uint8)
                                x1, y1, x2, y2 = obj.bbox.int().tolist()
                                decoded_np[y1 : max(y2, y1 + 1), x1 : max(x2, x1 + 1)] = 1
                            else:
                                decoded_np = mask_utils.decode(seg)
                                if decoded_np.ndim == 3:
                                    decoded_np = decoded_np[:, :, 0]
                        elif isinstance(seg, list):
                            # Polygons list -> RLE -> decode
                            rles = mask_utils.frPyObjects(seg, img_h, img_w)
                            rle = mask_utils.merge(rles)
                            decoded_np = mask_utils.decode(rle)
                        elif isinstance(seg, np.ndarray):
                            # Already a binary mask
                            decoded_np = (seg > 0).astype(np.uint8)
                        else:
                            raise RuntimeError(f"Unsupported segment type: {type(seg)}")
                    except Exception as e:
                        # Fallback to an empty mask if decoding fails
                        print(f"Warning, failed to decode segment ({type(seg)}): {e}")
                        decoded_np = np.zeros((img_h, img_w), dtype=np.uint8)

                    # Convert to torch uint8
                    obj.segment = torch.as_tensor(decoded_np, dtype=torch.uint8)

                    if list(obj.segment.shape) != [img_h, img_w]:
                        # Should not happen often, but adding for security
                        if not warning_shown:
                            print(
                                f"Warning expected instance segmentation size to be {[img_h, img_w]} but found {list(obj.segment.shape)}"
                            )
                            # Printing only once per datapoint to avoid spam
                            warning_shown = True

                        obj.segment = F.resize(
                            obj.segment[None], (img_h, img_w)
                        ).squeeze(0)

                    assert list(obj.segment.shape) == [img_h, img_w]

        warning_shown = False
        for query in datapoint.find_queries:
            if query.semantic_target is not None and not isinstance(query.semantic_target, torch.Tensor):
                seg = query.semantic_target
                decoded_np = None
                h, w = imgId2size[query.image_id]

                try:
                    if isinstance(seg, dict):
                        decoded_np = mask_utils.decode(seg)
                        if decoded_np.ndim == 3:
                            decoded_np = decoded_np[:, :, 0]
                    elif isinstance(seg, list):
                        # Polygons list -> RLE -> decode
                        rles = mask_utils.frPyObjects(seg, h, w)
                        rle = mask_utils.merge(rles)
                        decoded_np = mask_utils.decode(rle)
                    elif isinstance(seg, np.ndarray):
                        decoded_np = (seg > 0).astype(np.uint8)
                    else:
                        raise RuntimeError(f"Unsupported semantic_target type: {type(seg)}")
                except Exception as e:
                    print(f"Warning, failed to decode semantic_target ({type(seg)}): {e}")
                    decoded_np = np.zeros((h, w), dtype=np.uint8)

                query.semantic_target = torch.as_tensor(decoded_np).to(torch.uint8)

                if tuple(query.semantic_target.shape) != imgId2size[query.image_id]:
                    if not warning_shown:
                        print(
                            f"Warning expected semantic segmentation size to be {imgId2size[query.image_id]} but found {tuple(query.semantic_target.shape)}"
                        )
                        # Printing only once per datapoint to avoid spam
                        warning_shown = True

                    query.semantic_target = F.resize(
                        query.semantic_target[None], imgId2size[query.image_id]
                    ).squeeze(0)

                assert tuple(query.semantic_target.shape) == imgId2size[query.image_id]

        return datapoint

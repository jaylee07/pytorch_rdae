import numpy as np
import torch

def plot_tile_images(tensor, img_shapes, tile_shapes, tile_spacings=(0,0)):
    assert len(img_shapes) == 2
    assert len(tile_shapes) == 2
    assert len(tile_spacings) == 2
    if tensor.requires_grad:
        tensor = tensor.detach()

    out_shape = [(img_shape + tile_space) * tile_shape - tile_space
                 for img_shape, tile_shape, tile_space in zip(img_shapes, tile_shapes, tile_spacings)]

    tensor = torch.transpose(tensor, 1, 2)
    tensor = torch.transpose(tensor, 2, 3)

    (n_data, w, h, n_channel) = tensor.size()
    if tile_shapes[0] * tile_shapes[1] < n_data:
        raise NotImplementedError('tile_shapes[0] * tile_shapes[1] should be larger than n_data')

    if n_channel == 1:
        tile_images = np.zeros(shape=out_shape)
        for idx in range(n_data):
            row_idx, col_idx = int(idx / tile_shapes[1]), idx % tile_shapes[1]
            row_start = img_shapes[0] * row_idx
            row_end   = img_shapes[0] * (row_idx+1)
            col_start = img_shapes[1] * col_idx
            col_end   = img_shapes[1] * (col_idx+1)
            tile_images[row_start:row_end, col_start:col_end] = tensor[idx].cpu().squeeze().numpy()

    else:
        out_shape.append(n_channel)
        tile_images = np.zeros(shape = out_shape)
        for idx, in range(n_data):
            row_idx, col_idx = int(idx / tile_shapes[1]), idx % tile_shapes[1]
            row_start = img_shapes[0] * row_idx
            row_end = img_shapes[0] * (row_idx + 1)
            col_start = img_shapes[1] * col_idx
            col_end = img_shapes[1] * (col_idx + 1)
            tile_images[row_start:row_end, col_start:col_end, :] = tensor[idx].cpu().numpy()

    return tile_images

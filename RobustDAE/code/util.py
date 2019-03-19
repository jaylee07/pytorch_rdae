import os


def conv1d_output_size(h, kernel_h, stride, padding):
    h_out = int(((h - kernel_h + 2*padding) / stride) + 1)
    return h_out


def conv2d_output_size(w, h, kernel_w, kernel_h, stride, padding):
    w_out = int(((w - kernel_w + 2*padding) / stride) + 1)
    h_out = int(((h - kernel_h + 2*padding) / stride) + 1)
    return w_out, h_out


def convtr1d_output_size(h, kernel_h, stride, padding):
    h_out = stride*(h-1) + kernel_h - 2*padding
    return h_out


def convtr2d_output_size(w, h, kernel_w, kernel_h, stride, padding):
    w_out = stride*(w-1) + kernel_w - 2*padding
    h_out = stride*(h-1) + kernel_h - 2*padding
    return w_out, h_out


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_expconfig(args):
    noise_config = 'noise_{}_nr_{}_nc_{}_'.format(args.noise_method,
                                                  args.n_corrupt_rows,
                                                  args.n_corrupt_cols)
    train_config = 'lamb_{}_inepoch_{}_outepoch_{}'.format(args.lamb,
                                                           args.inner_epochs,
                                                           args.outer_epochs)
    class_config = 'class_{}_'.format(args.class_ratio)
    if 'cdae' in args.model:
        model_config = 'ch_{}_k_{}_s_{}_p_{}_fc_{}_embed_dim_{}_'.format(args.n_ch,
                                                                         args.kernels,
                                                                         args.strides,
                                                                         args.paddings,
                                                                         args.use_fc,
                                                                         args.embed_dim)
    else:
        model_config = 'dim_{}_'.format(args.dims)

    config = model_config + noise_config + class_config + train_config
    return config


def get_paths(args, exp_config):
    result_dir = os.path.join(args.result_dir, args.data, args.model, exp_config)
    save_dir = os.path.join(args.save_dir, args.data, args.model, exp_config)
    log_dir = os.path.join(args.log_dir, args.data, args.model, exp_config)
    make_dir(result_dir)
    make_dir(save_dir)
    make_dir(log_dir)

    return result_dir, save_dir, log_dir

#
# def add_noise(tensor, corrupt_col_idx=[1, 2, 3, 4, 5], corrupt_row_idx=[x for x in range(28)], method='fixed'):
#     """
#     Produce structured corrupted pixels
#     -input
#       - tensor: Input tensor. shape = [batch, channel, row, col]
#       - corrupted_col_idx: Structly corrupted column index
#       - corrupted_row_idx: Structly corrupted row index
#       - method: Noise method
#     """
#     if method == 'fixed':
#         for col in corrupt_col_idx:
#             tensor[:, :, :, col] = np.random.random()
#     elif method == 'uniform':
#         for col in corrupt_col_idx:
#             for row in corrupt_row_idx:
#                 tensor[:, :, row, col] = np.random.uniform()
#     elif method == 'gaussian':
#         for col in corrupt_col_idx:
#             for row in corrupt_row_idx:
#                 tensor[:, :, row, col] = np.random.normal()
#     else:
#         raise ValueError('Enter the proper noise method')
#
#     return tensor

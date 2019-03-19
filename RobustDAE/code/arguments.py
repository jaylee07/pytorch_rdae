import argparse

def get_arguments():

    parser = argparse.ArgumentParser(description='rdae')
    # Data arguments
    parser.add_argument('--datadir', type=str, default='/home/jehyuk/PycharmProjects/RobustDAE')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist','fashion-mnist','cifar'])
    parser.add_argument('--image_ch', type=int, default=1)
    parser.add_argument('--is_partial', default=False, action='store_true')
    parser.add_argument('--class_ratio', type=int, nargs='+', default=[5923,6742,5958,6131,5842,
                                                                       5421,5918,6265,5851,5949])
    # Model arguments
    parser.add_argument('--model', type=str, default='dae',
                        choices=['dae', 'cdae', 'l1_rdae', 'l1_rcdae', 'l21_rdae', 'l21_rcdae'])
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--n_ch', type=int, default=64)
    parser.add_argument('--dims', type=int, nargs='+', default=[784,200,20])
    parser.add_argument('--kernels', type=int, nargs='+', default=[4,4,4])
    parser.add_argument('--strides', type=int, nargs='+', default=[2,2,1])
    parser.add_argument('--paddings', type=int, nargs='+', default=[1,1,0])
    parser.add_argument('--out_act_fn', type=str, default='tanh')
    parser.add_argument('--act_fn', type=str, default='relu')
    parser.add_argument('--use_fc', default=False, action='store_true')
    parser.add_argument('--embed_dim', type=int, default=20)
    #Train arguments
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_corrupt_rows', type=int, default=5)
    parser.add_argument('--n_corrupt_cols', type=int, default=5)
    parser.add_argument('--noise_method', type=str, default='none',
                        choices=['none', 'rowwise', 'columnwise', 'rowcol', 'random'])
    parser.add_argument('--lamb', type=float, default=50)
    parser.add_argument('--n_workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--device_num', type=int, default=0)
    parser.add_argument('--multi_gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--inner_epochs', type=int, default=200)
    parser.add_argument('--outer_epochs', type=int, default=20)
    parser.add_argument('--log_dir', type=str, default='/home/jehyuk/PycharmProjects/RobustDAE/logs')
    parser.add_argument('--save_dir', type=str, default='/home/jehyuk/PycharmProjects/RobustDAE/models')
    parser.add_argument('--result_dir', type=str, default='/home/jehyuk/PycharmProjects/RobustDAE/results')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'infer'])
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--load', default=False, action='store_true')
    args = parser.parse_args()

    return args

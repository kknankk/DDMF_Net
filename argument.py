import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('--fusion_type', type=str, default='ecg', help='train or eval for [fusion,cxr,ecg,deeper_frequency_fusion ]')
    parser.add_argument('--save_dir', type=str, help='Directory relative which all output files are stored',
                    default='/home/ec2-user/yuan/dataset/multimodal/code/mimic2025_code/checkpoints')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of chunks to train')
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers (default: [64, 128, 196, 256, 320]).')
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16],
                        help='number of samples per resnet layer (default: [4096, 1024, 256, 64, 16]).')
    # parser.add_argument(
    #     "--gpus",
    #     type=str,
    #     default="0", 
    #     help="Comma-separated list of GPU device IDs to use, e.g. '0,1,2' or '1,2,3,4,5,6,7'"
    # )

    # parser.add_argument(
    #     "--num_fold",
    #     type=int,
    #     required=True,
    #     help="Fold 编号，例如 0 到 9"
    # )
    
    parser.add_argument('--mod_net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320],
                        help='filter size in resnet layers .')
    parser.add_argument('--mod_net_seq_lengh', type=int, nargs='+', default=[4096, 2048, 1024, 512, 256],
                        help='number of samples per resnet layer .')

    # parser.add_argument('--model', type=str,default='wavevit_s', required=False, choices=['ResNet1d', 'spectrogram', 'ECGModel', 'wavevit_s', 'CXRModels'], help='Specify the model to train')
    parser.add_argument('--ecg_model', type=str, default='none', help='ECG model name')
    parser.add_argument('--cxr_model', type=str, default='none', help='CXR model name')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--domain', type=str, default='frequency', choices=['frequency' , 'S_T','ecg_fusion'])
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for L2 regularization')
    parser.add_argument('--b', type=float, default=1, help='flooding')
    parser.add_argument('--patience', type=int, default=30, help='number of epoch to wait for best')
    parser.add_argument('--fusion_model', type=str, default=None, 
                        help='Specify the fusion model (e.g., FSRU). Default is None.')
    parser.add_argument('--name', type=str, required=True, help='Specify your name')
    parser.add_argument('--threshold', type=float, default=0.3, 
                        help='Threshold for binary classification (default: 0.5)')
# self.args.pretrained
    parser.add_argument('--pretrained', type=str, default='False', choices=['True' , 'False'],
                        help='pretrained')
    parser.add_argument('--jsd_loss', type=str, default='False', choices=['True' , 'False'],
                        help='pretrained')
    parser.add_argument('--ehr_n_layers', type=int, default=1)
    parser.add_argument('--ehr_n_head', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--lambda_attn_aux', type=float, default=1)

    parser.add_argument("--num_gpus", type=int, default=7, help="Number of GPUs per machine")
    parser.add_argument("--num_machines", type=int, default=1, help="Total number of machines")
    parser.add_argument("--machine_rank", type=int, default=0, help="Rank of the current machine")
    parser.add_argument("--dist_url", type=str, default="tcp://ip-172-31-44-55.us-east-2.compute.internal:23456", help="URL for initializing distributed training")
    
    
    return parser
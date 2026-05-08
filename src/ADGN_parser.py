import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ADGN.", allow_abbrev=False)

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')

    parser.add_argument('--data_path', default='./Data/',
                        help='Input data path')
    parser.add_argument('--save_path', help='Input save path')
    parser.add_argument('--ADG_path', default='./Data/data_info/',
                        help='Input ADGdata path')

    parser.add_argument('--domain', choices=['Rest-TA', 'Hotel-TA'],
                        default='Rest-TA', help='Choose the domain: restuarant or hotel')

    parser.add_argument('--is_preprocess', type=int, choices=[
                        0, 1], default=0, help='Whether to apply preprocessing: 1 (yes), 0 (no)')
    parser.add_argument('--is_train', type=int, choices=[
                        0, 1], default=1, help='Whether to train or test: 1 (train), 0 (test)')
    parser.add_argument('--is_pretrained', type=int, choices=[
                        0, 1], default=0, help='Whether to use pretrained model : 1 (yes), 0 (no)')

    parser.add_argument('--n_epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of Early Stopping Count')
    parser.add_argument("--batch_size", type=int,
                        default=32, help="batch size")
    parser.add_argument("--max_token", type=int, default=512,
                        help="Max token for tokeninzation")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="dropout rate for sentimental features")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="learning rate for sentimental features")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="weight decay for sentimental features")
    parser.add_argument("--alpha", type=float,
                        default=0.5, help="Fusion weight")

    return parser.parse_known_args()[0]

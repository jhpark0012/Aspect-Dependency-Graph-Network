import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa: E402

import warnings
warnings.filterwarnings('ignore')  # noqa: E402

import gc
import random
import pandas as pd
import numpy as np
import torch
import torch.multiprocessing as mp  # noqa: E402
mp.set_sharing_strategy("file_system")  # noqa: E402

from torch.utils.data import DataLoader

from pathlib import Path
from dotenv import load_dotenv
from datasets import load_from_disk

import ADGN_parser
from ADGN_model import ADGN
import loss_func
import modeling
import preprocessing

BASE_DIR = Path("/workspace/")
DEVICE = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

load_dotenv(BASE_DIR / ".env")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_preprocessing(args):
    raw_data_path = BASE_DIR / \
        f"Data/{args.domain}/clean_data"

    raw_data = load_from_disk(raw_data_path)

    train_clean = raw_data['train']
    val_clean = raw_data['val']
    test_clean = raw_data['test']

    print(f"=== Start Preprocessing {args.domain} Dataset ===")
    preprocessing.get_prep(BASE_DIR, args, train_clean, val_clean, test_clean)

    print(f"=== End Preprocessing {args.domain} Dataset ===")


def run_train(args, model, criterion, optimizer, device):
    train_data = torch.load(f"{args.data_path}/final_data/train_data.pt")
    val_data = torch.load(f"{args.data_path}/final_data/val_data.pt")

    if args.is_pretrained == 1:
        model.load_state_dict(torch.load(
            f"{args.save_path}/best_model_{args.best_model}.pt"
        ))

    train_dataset = preprocessing.LoadDataset(train_data)
    val_dataset = preprocessing.LoadDataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=preprocessing.custom_collate,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=preprocessing.custom_collate,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train_result = modeling.training(
        args, model, train_loader, val_loader, optimizer, criterion, device
    )

    return train_result


def run_test(args, model, criterion, device):
    test_data = torch.load(
        f"{args.data_path}/final_data/test_data.pt")

    test_dataset = preprocessing.LoadDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=preprocessing.custom_collate,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    model.load_state_dict(torch.load(
        f"{args.save_path}/best_model.pt"))

    # For eval M/N
    df_review_type = pd.read_csv(
        f"{args.ADG_path}/df_review_type.csv"
    )

    test_loss, eval_res, df_pred = modeling.test_epoch(
        args, model, df_review_type, test_loader, criterion, device
    )

    df_pred.to_csv(
        f'{args.save_path}/df_pred_{args.domain}.csv',
        index=False
    )

    print(f'=== Test {args.domain} Result ===')
    for key, val in eval_res.items():
        if key in ['EMR', 'HL'] or 'F1' in key:
            print(f'{key} : {val}')

    test_result = {
        "test_loss": test_loss,
        **eval_res,
    }

    return test_result


if __name__ == "__main__":
    args = ADGN_parser.parse_args()
    set_seed()

    aspects_dict = {
        'Rest-TA': ['Overall', 'Value', 'Service', 'Food', 'Atmosphere'],
        'Hotel-TA': ['Overall', 'Value', 'Service', 'Location',
                     'Room', 'Cleanliness', 'Sleep'],
    }
    aspects = aspects_dict[args.domain]
    n_aspect = len(aspects)

    train_result = {}
    test_result = {}

    if args.is_preprocess == 1:
        run_preprocessing(args)

    model = ADGN(args=args, device=DEVICE, aspects=aspects)
    model = model.to(DEVICE)

    criterion = loss_func.MaskedBCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.is_train == 1:
        train_result = run_train(args, model, criterion, optimizer, DEVICE)

    elif args.is_train == 0:
        test_result = run_test(args, model, criterion, DEVICE)

        final_result = {
            "domain": args.domain,
            "batch_size": args.batch_size,
            **test_result
        }

        result_df = pd.DataFrame([final_result])

        os.makedirs(args.save_path, exist_ok=True)

        result_df.to_csv(
            f"{args.save_path}/ADGN_results_{args.domain}.csv",
            index=False
        )

    gc.collect()
    torch.cuda.empty_cache()

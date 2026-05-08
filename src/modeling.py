import numpy as np
import pandas as pd
import torch

import utils
from tqdm.auto import tqdm


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc='Train Model..'):

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        syntax_graph = torch.stack([a.to_dense() for a in batch["syntax_graph"]]).to(
            device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask, syntax_graph)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def training(args, model, train_loader, val_loader, optimizer, criterion, device):
    best_epoch = 0
    best_loss = float('inf')
    patience = args.patience
    counter = 0
    actual_epochs = 0
    best_model_path = f"{args.save_path}/best_model.pt"

    df_review_type = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for epoch in tqdm(range(args.n_epochs), desc='Training model in Epochs'):
        print(f"Epoch [{epoch+1}/{args.n_epochs}]")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, _, _ = test_epoch(
            args, model, df_review_type, val_loader, criterion, device
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        actual_epochs += 1

        print(
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
        )

        # Early Stopping
        if best_loss > val_loss:
            best_loss = val_loss
            best_epoch = epoch + 1

            print(
                f'Best Loss in [{epoch+1}/{args.n_epochs}] : {best_loss}!')

            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} / {patience}")
            if counter >= patience:
                print("Early stopping triggered!")
                break

    if best_epoch > 0:
        model.load_state_dict(torch.load(best_model_path))

    training_result = {
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "best_loss": best_loss,
    }

    return training_result


def test_epoch(args, model, df_review_type, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Test Model..'):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=True)
            syntax_graph = torch.stack(
                [a.to_dense() for a in batch["syntax_graph"]]
            ).to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask, syntax_graph)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            labels_np = labels.cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels_np)

    val_loss = total_loss / len(loader)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    df_pred = pd.DataFrame({
        "y_pred": all_preds.tolist(),
        "y_true": all_labels.tolist()
    })

    eval_res = utils.get_final_scores(
        args, model_id='ADGN', task_name='test',
        df_pred=df_pred, df_review_type=df_review_type
    )

    return val_loss, eval_res, df_pred

import ast
from tqdm.auto import tqdm
import pandas as pd
from datasets import DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# Check polarity distriution
def get_polarity_dist(data_dict: DatasetDict, aspects: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:

    result = {aspect: {'0': [], '1': [], '-1': []} for aspect in aspects}

    for split in ['train', 'val', 'test']:
        cat_dict = {aspect: [0, 0, 0] for aspect in aspects}  # [neg, pos, nan]
        data = data_dict[split]

        for d in data:
            for pol, name in zip(d['final_polarity'], d['aspect_name']):
                if pol == 0:
                    cat_dict[name][0] += 1
                elif pol == 1:
                    cat_dict[name][1] += 1
                else:
                    cat_dict[name][2] += 1

        for aspect in aspects:
            result[aspect]['0'].append(cat_dict[aspect][0])
            result[aspect]['1'].append(cat_dict[aspect][1])
            result[aspect]['-1'].append(cat_dict[aspect][2])

    table_data = {}
    for aspect in aspects:
        table_data[(aspect, '0')] = result[aspect]['0']
        table_data[(aspect, '1')] = result[aspect]['1']
        table_data[(aspect, '-1')] = result[aspect]['-1']

    df = pd.DataFrame(table_data, index=['Train', 'Val', 'Test'])
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    total = df.groupby(level=0, axis=1).sum()

    ratio_df = df.copy()
    for aspect in df.columns.levels[0]:
        ratio_df[aspect] = df[aspect].div(total[aspect], axis=0)

    ratio_df = ratio_df.round(4)

    return df, ratio_df


def check_data_dist(data_dict: DatasetDict, aspects: list[str]):

    polarity_dist_res, polarity_dist_ratio_res = get_polarity_dist(
        data_dict, aspects)

    print('## Number of Data ##')
    print(
        f"Train : {len(data_dict['train'])}, Val : {len(data_dict['val'])}, Test:{len(data_dict['test'])}")
    print('-'*20)

    print('## Polarity Dist ##')
    print(polarity_dist_res)
    print('-'*20)
    print('## Polarity Ratio Dist ##')
    print(polarity_dist_ratio_res)
    print('-'*20)


def get_aspect_scores(df_pred, pred, label, aspect_list):

    metric_list = ['Acc', 'Prec', 'Rec', 'F1']

    # Scores by aspects
    aspect_col = [
        f"{aspect}_{metric}" for aspect in aspect_list for metric in metric_list]
    aspect_res = []

    macro_col = [f"{'Macro'}_{metric}" for metric in metric_list]

    micro_col = [f"{'Micro'}_{metric}" for metric in metric_list]
    micro_pred = []
    micro_label = []

    n_aspects = pred.shape[1]
    aspect_accuracy = []
    aspect_precision = []
    aspect_recall = []
    aspect_f1 = []

    EMR = len(df_pred[df_pred['y_pred'] == df_pred['y_true']]) / len(df_pred)

    ## Macro Score ##
    for i in tqdm(range(n_aspects)):
        pred_col = pred[:, i]
        label_col = label[:, i]

        p = np.array(pred_col, dtype=int)
        l = np.array(label_col, dtype=int)

        acc = accuracy_score(l, p)
        prec = precision_score(l, p, zero_division=0)
        rec = recall_score(l, p, zero_division=0)
        f1 = f1_score(l, p, zero_division=0)

        aspect_accuracy.append(acc)
        aspect_precision.append(prec)
        aspect_recall.append(rec)
        aspect_f1.append(f1)

        aspect_res += [acc, prec, rec, f1]

        micro_pred += list(p)
        micro_label += list(l)

    macro_acc = np.mean(aspect_accuracy)
    macro_prec = np.mean(aspect_precision)
    macro_rec = np.mean(aspect_recall)
    macro_f1 = np.mean(aspect_f1)

    ## Micro Score ##
    micro_acc = accuracy_score(micro_label, micro_pred)
    micro_prec = precision_score(micro_label, micro_pred, zero_division=0)
    micro_rec = recall_score(micro_label, micro_pred, zero_division=0)
    micro_f1 = f1_score(micro_label, micro_pred, zero_division=0)

    macro_res = [macro_acc, macro_prec, macro_rec, macro_f1]
    micro_res = [micro_acc, micro_prec, micro_rec, micro_f1]

    eval_res = {'EMR': EMR, 'HL': 1-micro_acc}

    for k, v in zip(aspect_col, aspect_res):
        eval_res[k] = v

    for k, v in zip(macro_col, macro_res):
        eval_res[k] = v
    for k, v in zip(micro_col, micro_res):
        eval_res[k] = v

    return eval_res


def get_mn_aspect_scores(df_pred, df_review_type, aspect_list):
    mn_res = {}

    df_review_type = df_review_type[df_review_type['data_type'] == 'test'].reset_index(
        drop=True)

    assert len(df_pred) == len(
        df_review_type), "The shapes of pred and df_review_type must be the same."

    all_y_true_mentioned = []
    all_y_pred_mentioned = []
    all_y_true_nonmentioned = []
    all_y_pred_nonmentioned = []

    for idx, aspect in enumerate(tqdm(aspect_list)):
        if aspect == "Overall":
            continue

        y_true_mentioned = []
        y_pred_mentioned = []
        y_true_nonmentioned = []
        y_pred_nonmentioned = []

        for row_idx in range(len(df_pred)):
            pred = df_pred.loc[row_idx, "y_pred"][idx]
            label = df_pred.loc[row_idx, "y_true"][idx]

            if pred is None or pred == "None" or pred not in [0, 1, "0", "1"]:
                pred = 1 - int(label)
            else:
                pred = int(pred)

            label = int(label)

            is_mentioned = df_review_type.loc[row_idx, aspect] == 1

            if is_mentioned:
                y_true_mentioned.append(label)
                y_pred_mentioned.append(pred)
                all_y_true_mentioned.append(label)
                all_y_pred_mentioned.append(pred)
            else:
                y_true_nonmentioned.append(label)
                y_pred_nonmentioned.append(pred)
                all_y_true_nonmentioned.append(label)
                all_y_pred_nonmentioned.append(pred)

        # Per-aspect F1
        if len(set(y_true_mentioned)) > 1:
            f1_mentioned = f1_score(
                y_true_mentioned, y_pred_mentioned, average="macro")
        else:
            f1_mentioned = None

        if len(set(y_true_nonmentioned)) > 1:

            f1_nonmentioned = f1_score(
                y_true_nonmentioned, y_pred_nonmentioned, average="macro")

        else:
            f1_nonmentioned = None

        mn_res[f"{aspect.lower()}_mentioned_f1"] = round(
            f1_mentioned, 3) if f1_mentioned is not None else None
        mn_res[f"{aspect.lower()}_nonmentioned_f1"] = round(
            f1_nonmentioned, 3) if f1_nonmentioned is not None else None

    # Micro/Macro f1 (Mentioned)
    if len(set(all_y_true_mentioned)) > 1:
        mn_res["mentioned_macro_f1"] = round(
            f1_score(all_y_true_mentioned, all_y_pred_mentioned, average="macro"), 4)
        mn_res["mentioned_micro_f1"] = round(
            f1_score(all_y_true_mentioned, all_y_pred_mentioned, average="micro"), 4)
    else:
        mn_res["mentioned_macro_f1"] = None
        mn_res["mentioned_micro_f1"] = None

    # Micro/Macro f1 (Non-Mentioned)
    if len(set(all_y_true_nonmentioned)) > 1:
        mn_res["nonmentioned_macro_f1"] = round(
            f1_score(all_y_true_nonmentioned, all_y_pred_nonmentioned, average="macro"), 4)
        mn_res["nonmentioned_micro_f1"] = round(
            f1_score(all_y_true_nonmentioned, all_y_pred_nonmentioned, average="micro"), 4)
    else:
        mn_res["nonmentioned_macro_f1"] = None
        mn_res["nonmentioned_micro_f1"] = None

    return mn_res


def get_final_scores(args, model_id, task_name, df_pred, df_review_type) -> dict[str]:

    df_pred['y_pred'] = df_pred['y_pred'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df_pred['y_true'] = df_pred['y_true'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    for i in range(len(df_pred)):
        for j in range(len(df_pred.loc[i, 'y_pred'])):
            if df_pred.loc[i, 'y_pred'][j] not in [0, 1]:
                df_pred.loc[i, 'y_pred'][j] = 1 - \
                    int(df_pred.loc[i, 'y_true'][j])
            else:
                df_pred.loc[i, 'y_pred'][j] = int(df_pred.loc[i, 'y_pred'][j])

            df_pred.loc[i, 'y_true'][j] = int(df_pred.loc[i, 'y_true'][j])

    pred = np.array(df_pred['y_pred'].tolist())
    label = np.array(df_pred['y_true'].tolist())

    assert pred.shape == label.shape, "The shapes of pred and label must be the same."

    if args.domain == 'Rest-TA':
        aspect_list = ['Overall', 'Value', 'Service', 'Food', 'Atmosphere']

    elif args.domain == 'Hotel-TA':
        aspect_list = ['Overall', 'Value', 'Service',
                       'Location', 'Rooms', 'Cleanliness', 'Sleep']

    ## Aspects Scores ##
    try:
        eval_res = get_aspect_scores(df_pred, pred, label, aspect_list)

    except Exception as e:
        print(
            f'err in aspect score calculation {model_id}, {task_name}, {args.domain} | error: {e}')
        eval_res = {'EMR': -1, 'HL': -1}
        eval_res = {f"{aspect}_{metric}": -
                    1 for aspect in aspect_list for metric in ['Acc', 'Prec', 'Rec', 'F1']}

    ## M/N Score ##
    try:
        mn_res = get_mn_aspect_scores(df_pred, df_review_type, aspect_list)
    except Exception as e:
        print(
            f'err in M/N score calculation {model_id}, {task_name}, {args.domain} | error: {e}')
        mn_res = {'mentioned_macro_f1': -1, 'mentioned_micro_f1': -1,
                  'nonmentioned_macro_f1': -1, 'nonmentioned_micro_f1': -1}
        mn_res = {f"{aspect}_{metric}": -
                  1 for aspect in aspect_list for metric in ['mentioned_f1', 'nonmentioned_f1']}

    total_res = {'model_id': model_id, 'task': task_name,
                 'domain': args.domain, **eval_res, **mn_res}

    return total_res

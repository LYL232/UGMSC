import argparse
import os
import os.path
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Bert_PLOME
from ugmodel import UG_Bert_PLOME
from tagging_eval import score_f_py, score_f, score_f_sent
from transformers import AdamW, WarmupLinearSchedule
from utils import read_train_ds, collate_fn, MyData  # , convert_example
from utils import get_zi_py_matrix, convert_single_example
from pinyin_tool import PinyinTool
import tokenization
import logging

logger = logging.getLogger(__name__)


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type to run.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--model_path", type=str, default="./datas/pretrained_plome/",
                        help="Pretraining model name or path.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="The maximum total input sequence length after SentencePiece tokenization.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout ratio.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay ratio.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=100, help="logs steps.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--corrector_layers", type=int, default=3,
                        help="Transformer layers of the correctors of uncertainty-guided model")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Linear warmup proption over the training process.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
    parser.add_argument("--pinyin_dim", type=int, default=32, help="pinyin dim.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--output_dir", type=str, default="output", help="output file path.")
    args = parser.parse_args()

    logger.info("Training/evaluation parameters %s", args)

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_model_output(outputs, label_ids, py_labels, _lmask):
    active_loss = _lmask.view(-1) == 1
    active_hanzi_labels = label_ids.view(-1)[active_loss]
    active_pinyin_labels = py_labels.view(-1)[active_loss]

    logits_pinyin = outputs['logits_pinyin']
    logits_hanzi = outputs['logits_hanzi']

    logits_hanzi = logits_hanzi.view((-1, logits_hanzi.shape[2]))[active_loss]
    logits_pinyin = logits_pinyin.view((-1, logits_pinyin.shape[2]))[active_loss]

    closs = torch.nn.functional.cross_entropy(logits_hanzi, active_hanzi_labels)
    ploss = torch.nn.functional.cross_entropy(logits_pinyin, active_pinyin_labels)

    auxiliary_loss = {}
    for k, v in outputs.items():
        if 'zi_logits' in k:
            active_logits = v.view((-1, v.shape[2]))[active_loss]
            auxiliary_loss[k.replace('_logits', '')] = torch.nn.functional.cross_entropy(
                active_logits, active_hanzi_labels)
        elif 'py_logits' in k:
            active_logits = v.view((-1, v.shape[2]))[active_loss]
            auxiliary_loss[k.replace('_logits', '')] = torch.nn.functional.cross_entropy(
                active_logits, active_pinyin_labels)

    return closs, ploss, auxiliary_loss


def trainer(model, train_data_loader, eval_data_loader, args, device, label_list, py_label_list):
    num_training_steps = args.max_steps \
        if args.max_steps > 0 else len(train_data_loader) * args.epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'modal_scale']

    # different lr
    different_lr_params_config = {
        'modal_scale': ([], 1e-3),
    }
    default_lr_params = []
    for n, p in param_optimizer:
        lr_set = False
        for k, (prams_list, new_lr) in different_lr_params_config.items():
            if k in n:
                prams_list.append((n, p))
                logger.info(f'the max lr of parameter {n} is {new_lr}')
                lr_set = True
                break
        if lr_set:
            continue
        default_lr_params.append((n, p))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in default_lr_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in default_lr_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate},
    ]
    for prams_list, new_lr in different_lr_params_config.values():
        optimizer_grouped_parameters.extend([
            {'params': [p for n, p in prams_list if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': new_lr},
            {'params': [p for n, p in prams_list if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': new_lr},
        ])

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    warmup_steps = int(num_training_steps * args.warmup_proportion)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps,
        t_total=num_training_steps
    )

    global_steps = 0
    best_score = 0.0

    model = model.to(device)

    accumulated_closs = 0
    logging_closs = 0
    accumulated_ploss = 0
    logging_ploss = 0
    last_logging_step = 0
    tr_aloss, logging_aloss = {}, {}

    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch + 1}')
        for step, batch in enumerate(train_data_loader, start=1):
            model.train()
            input_ids, input_mask, pinyin_ids, stroke_ids, _lmask, label_ids, py_labels = tuple(
                t.to(device) for t in batch)

            if isinstance(model, Bert_PLOME):
                outputs = model(
                    input_ids, pinyin_ids, stroke_ids, device, None, input_mask, True
                )
            else:
                outputs = model(
                    input_ids, pinyin_ids, stroke_ids, device, None, input_mask,
                    True
                )

            closs, ploss, aloss = process_model_output(outputs, label_ids, py_labels, _lmask)

            accumulated_closs += closs.item()
            accumulated_ploss += ploss.item()

            loss = closs + ploss

            for k, v in aloss.items():
                loss += v
                tr_aloss[k] = tr_aloss.get(k, 0.0) + v.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (global_steps + 1) % args.logging_steps == 0:
                steps = global_steps - last_logging_step
                closs_value = (accumulated_closs - logging_closs) / steps
                logging_closs = accumulated_closs
                ploss_value = (accumulated_ploss - logging_ploss) / steps
                logging_ploss = accumulated_ploss
                logger.info(
                    "global step %d, epoch: %d, batch: %d, closs: %f, ploss: %f"
                    % (global_steps + 1, epoch + 1, step, closs_value, ploss_value)
                )
                this_aloss = {}
                for k, v in aloss.items():
                    this_aloss[k] = (tr_aloss.get(k, 0.0) - logging_aloss.get(k, 0.0)) / args.logging_steps
                    logging_aloss[k] = tr_aloss.get(k, 0.0)
                if len(this_aloss) > 0:
                    logger.info(
                        "Scloss: %f, Pcloss: %f, Gcloss: %f, Jcloss: %f"
                        % (
                            this_aloss['semantic_zi'],
                            this_aloss['phonetic_zi'],
                            this_aloss['graphic_zi'],
                            this_aloss['joint_zi'],
                        )
                    )
                last_logging_step = global_steps

            if global_steps % args.save_steps == 0:
                for k, value in model._modules.items():
                    if 'scale' not in k:
                        continue
                    weight = value.weight.cpu().detach().numpy()
                    w = np.log(np.exp(weight) + 1)
                    logger.info(f'{k}.scale: {w.tolist()}')
                logger.info('*' * 30 + f"Eval: step {global_steps}" + '*' * 30)
                f1 = evaluate(model, eval_data_loader, device, label_list, py_label_list, args.output_dir)
                if f1 > best_score:
                    # save best model
                    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
                    logger.info("Save best model at %d step.", global_steps)
                    best_score = f1
            if 0 < args.max_steps <= global_steps:
                break
            global_steps += 1
        if 0 < args.max_steps <= global_steps:
            break
    logger.info('*' * 30 + f"Eval: step {global_steps}" + '*' * 30)
    f1 = evaluate(model, eval_data_loader, device, label_list, py_label_list, args.output_dir)
    if f1 > best_score:
        # save best model
        torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
        logger.info("Save best model at %d step.", global_steps)


def evaluate(model, eval_data_loader, device, label_list, py_label_list, output_dir):
    model.eval()
    all_inputs, all_golds, all_preds = [], [], []
    all_py_golds, all_py_preds = [], []
    all_fusino_preds = []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    all_py_pred_sent, all_py_gold_sent, all_fusion_sent = [], [], []
    all_py_inputs, all_py_inputs_sent = [], []
    for step, batch in enumerate(eval_data_loader, start=1):
        input_ids, input_mask, pinyin_ids, stroke_ids, _lmask, label_ids, py_labels = tuple(t.to(device) for t in batch)
        batch_size = len(input_ids)
        max_sen_len = len(input_ids[0])

        with torch.no_grad():
            outputs = model(input_ids, pinyin_ids, stroke_ids, device, None, input_mask, False)

            prob_hanzi = outputs['logits_hanzi']
            prob_pinyin = outputs['logits_pinyin']
            prob_fusion = outputs['prob_fusion']

            prob_hanzi = np.argmax(prob_hanzi.cpu().numpy(), axis=2)
            prob_pinyin = np.argmax(prob_pinyin.cpu().numpy(), axis=2)
            prob_fusion = np.argmax(prob_fusion.cpu().numpy(), axis=2)
            label_ids = label_ids.cpu().numpy()
            py_labels = py_labels.cpu().numpy()
            input_ids = input_ids.cpu().numpy()
            pinyin_ids = pinyin_ids.cpu().numpy()

        for k in range(batch_size):
            tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
            for j in range(max_sen_len):
                if _lmask[k][j] == 0:
                    continue
                all_golds.append(label_ids[k][j])
                all_preds.append(prob_hanzi[k][j])
                all_inputs.append(input_ids[k][j])
                tmp1.append(label_list[label_ids[k][j]])
                tmp2.append(label_list[prob_hanzi[k][j]])
                tmp3.append(label_list[input_ids[k][j]])
                # 拼音
                all_py_inputs.append(pinyin_ids[k][j])
                all_py_golds.append(py_labels[k][j])
                all_py_preds.append(prob_pinyin[k][j])
                all_fusino_preds.append(prob_fusion[k][j])
                tmps4.append(str(py_labels[k][j]))
                tmps5.append(str(prob_pinyin[k][j]))
                tmps6.append(label_list[prob_fusion[k][j]])
                tmps7.append(str(pinyin_ids[k][j]))
            all_golds_sent.append(tmp1)
            all_preds_sent.append(tmp2)
            all_inputs_sent.append(tmp3)
            all_py_pred_sent.append(tmps4)
            all_py_gold_sent.append(tmps5)
            all_fusion_sent.append(tmps6)
            all_py_inputs_sent.append(tmps7)
    all_golds = [label_list[k] for k in all_golds]
    all_preds = [label_list[k] for k in all_preds]
    all_inputs = [label_list[k] for k in all_inputs]
    all_fusino_preds = [label_list[k] for k in all_fusino_preds]
    all_py_inputs = [py_label_list.get(int(k), k) for k in all_py_inputs]
    all_py_golds = [py_label_list.get(int(k), k) for k in all_py_golds]
    all_py_preds = [py_label_list.get(int(k), k) for k in all_py_preds]
    logger.info('pinyin result:')
    score_f_py((all_py_inputs, all_py_golds, all_py_preds), (all_inputs, all_golds, all_preds), output_dir, False)
    logger.info('fusion result:')
    p, r, f = score_f((all_inputs, all_golds, all_fusino_preds), out_dir=output_dir)
    score_f_sent(all_inputs_sent, all_golds_sent, all_fusion_sent)
    return f


def main():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    args = config()
    setup_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据读取部分
    train_list = list(read_train_ds('datas/train.txt'))
    test_list = list(read_train_ds('datas/test_1100.txt'))

    pytool = PinyinTool(py_dict_path='pinyin_data/zi_py.txt', py_vocab_path='pinyin_data/py_vocab.txt', py_or_sk='py')
    sktool = PinyinTool(py_dict_path='stroke_data/zi_sk.txt', py_vocab_path='stroke_data/sk_vocab.txt', py_or_sk='sk')
    pyid2_seq = pytool.get_pyid2seq_matrix()  # pinyin_id到zimu_id的映射
    skid2_seq = sktool.get_pyid2seq_matrix()  # 汉字到笔画的映射
    vocab_file = args.model_path + 'vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    label_list = {}
    for key in tokenizer.vocab:
        label_list[tokenizer.vocab[key]] = key
    py_label_list = {v: k for k, v in pytool.vocab.items()}
    # 这里偷懒了，有些地方直接复用了原作者的代码
    zi_py_matrix = get_zi_py_matrix(pytool, tokenizer)
    # input_ids, input_mask, segment_ids(pinyin), stroke_ids, _lmask, label_ids
    train_ids = [convert_single_example(example, args.max_seq_length, tokenizer, pytool, sktool) for example in
                 train_list]
    test_ids = [convert_single_example(example, args.max_seq_length, tokenizer, pytool, sktool) for example in
                test_list]

    train_x = MyData(train_ids)
    test_x = MyData(test_ids)
    collate = partial(collate_fn, tokenizer=tokenizer, pytool=pytool)
    train_data_loader = DataLoader(train_x, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
    test_data_loader = DataLoader(test_x, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

    # 模型建立
    if args.model_type == 'ug':
        model = UG_Bert_PLOME(
            args.model_path,
            num_class=len(label_list),
            pyid2seq=pyid2_seq,
            skid2seq=skid2_seq,
            py_dim=args.pinyin_dim,
            max_sen_len=args.max_seq_length,
            zi_py_matrix=zi_py_matrix,
            dropout=args.dropout,
            corrector_layers=args.corrector_layers
        )
    elif args.model_type == 'plome':
        model = Bert_PLOME(
            args.model_path,
            num_class=len(label_list),
            pyid2seq=pyid2_seq,
            skid2seq=skid2_seq,
            py_dim=args.pinyin_dim,
            max_sen_len=args.max_seq_length,
            zi_py_matrix=zi_py_matrix,
            dropout=args.dropout
        )
    else:
        raise NotImplementedError

    # 加载预训练权重
    logger.info("start to load weight")
    model.load_state_dict(torch.load(args.model_path + 'pytorch_model.bin'), strict=False)
    # 训练
    logger.info('start to train')

    trainer(model, train_data_loader, test_data_loader, args, device, label_list, py_label_list)


if __name__ == "__main__":
    main()

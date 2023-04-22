import os
import sys
import json
from argparse import ArgumentParser
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import classification_report

from custom_bert.module import file_reader, custom_dataset
from custom_bert.bert import bert

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
modBERT = bert.ConcatedBERT.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=2)

def main(args):
    dataset = args.dataset
    target = args.target
    size = args.docsize
    
    target_dir = f'/data/realive333/kakuyomu-dataset/tsv/{dataset}/{target}/'
    save_dir = f'./cat_saves/{target}'
    result_dir = f'./cat_results/{target}'
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    train_df = file_reader.getDataframeByPath(f'{target_dir}/train.json', size)
    eval_df = file_reader.getDataframeByPath(f'{target_dir}/dev.json', size)
    
    train_ds = custom_dataset.getDatasetByDataframe(train_df, tokenizer)
    eval_ds = custom_dataset.getDatasetByDataframe(eval_df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=save_dir,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        learning_rate=8e-6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=100,
        weight_decay=0.00001,
        no_cuda=False,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=modBERT,
        args=training_args,
        compute_metrics=bert.compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    train_result = trainer.train()
    
    test_df = file_reader.getDataframeByPath(f'{target_dir}/test.json', size)
    test_ds = custom_dataset.getDatasetByDataframe(test_df, tokenizer)
    actl_labels = test_ds[:]['label']
    test_result =  trainer.predict(test_ds, ignore_keys=['loss', 'last_hidden_state', 'hidden_states', 'attentions'])
    pred_label= test_result.predictions.argmax(axis=1).tolist()
    
    report = classification_report(actl_labels, pred_label, target_names=['False', 'True'], output_dict=True)
    
    
    with open(f'{result_dir}/eval_record.jsonl', 'a+') as f:
        json.dump(train_result.metrics, f)
    with open(f'{result_dir}/test_record.jsonl', 'a+') as f:
        json.dump(test_result.metrics, f)
    with open(f'{save_dir}/record.json', 'w+') as f:
        json.dump(report, f)
        
if __name__ == '__main__':
    parser = ArgumentParser(description="Custom BERT Trainer")
    parser.add_argument('--target', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='first-match-scatter')
    parser.add_argument('--docsize', type=int, default=5)
                                
                                
    args = parser.parse_args()
    print(f'Train Custom BERT\n\tTarget: {args.dataset}/{args.target}\n\tDoc Size: {args.docsize}')
    
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    main(args)

import argparse, os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    return Dataset.from_pandas(df[['text','label']])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--model', default='distilbert-base-uncased')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    ds = load_data(args.data)
    label_names = sorted(list(set(ds['label'])))
    label2id = {l:i for i,l in enumerate(label_names)}
    id2label = {i:l for l,i in label2id.items()}
    ds = ds.map(lambda x: {'labels': label2id[x['label']]} )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    def tok(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    ds = ds.map(tok, batched=True)
    ds = ds.rename_column('labels','labels')
    ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    split = ds.train_test_split(test_size=0.1, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split['train'],
        eval_dataset=split['test']
    )

    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    with open(os.path.join(args.output,'labels.txt'),'w') as f:
        for l in label_names:
            f.write(str(l)+'\n')

if __name__ == '__main__':
    main()

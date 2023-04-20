from torch.utils.data import Dataset

class ModDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length=512):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        texts = self.X[index]
        label = self.y[index]
        ids, msk, tkn = [], [], []
        for seq in texts:
            input = self.tokenizer.encode_plus(
                seq,
                add_special_tokens = True,
                max_length = self.max_length,
                pad_to_max_length = True,
                truncation=True,
                padding='max_length'
            )
            ids.append(input['input_ids'])
            msk.append(input['attention_mask'])
            tkn.append(input['token_type_ids'])
            
        return {
            'input_ids': ids,
            'attention_mask': msk,
            'token_type_ids': tkn,
            'label': label
        }
    
def getDatasetByDataframe(dataframe, tokenizer):
    x = dataframe['text'].tolist()
    y = dataframe['label'].tolist()
    return ModDataset(x, y, tokenizer)
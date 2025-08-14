def get_data_df(
    data_path='data', 
    data_type: str='train', 
    nlp_columns: list=['UniqueID', 'nlp_feature_vector']
):
    import pandas as pd
    from pathlib import Path
    from functools import reduce
    """
    Combine all provided files in a single dataset.
    Convenient for analysis and modeling downstream
    """
    # create a path to append to later
    data_path = Path(f'{data_path}/{data_type}')
    
    # add type to get the corresponding files - easier if the format is the same than passing a new list everytime
    validation_file = f'y_{data_type}.csv' if data_type=='train' else f'y_{data_type}_blank.csv'
    files=[
        f'x_{data_type}_nlp.csv', 
        f'x_{data_type}_tab.csv', 
        validation_file
    ] 
     
    #nlp train does not have column ids or name so adding it for convenience -  will also help with joining the data
    data_dfs = [
        pd.read_csv(data_path/fname, header=None, names=nlp_columns) 
        if 'nlp' in fname else pd.read_csv(data_path/fname)
        for fname in files 
    ] 
    data_df = reduce(lambda left, right: pd.merge(left, right, on='UniqueID'), data_dfs)
    return data_df

def get_padded_tokens(
    data_df, 
    pad_token_id: int=0, 
    feature_name: str='nlp_feature_vector'
):
    """
    Add padding to the tokens
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    
    nlp_tokens = []
    for vec in data_df[feature_name]: #change the indexing here to be more dynamic - put this all into a function
        nlp_tokens.append(torch.tensor(eval(vec)))

    padded_dataset = pad_sequence(
        nlp_tokens,
        batch_first=True,
        padding_value=pad_token_id
    )
    print(f"Shape of the token features after padding: {padded_dataset.shape}")
    return padded_dataset

def generate_embeddings_from_tokens(
    data_df, 
    pad_token_id: int=0,
    model_name: str="answerdotai/ModernBERT-base",
    embeddings_file_name: str='data/train/x_train_nlp_embeddings.csv'
):
    import torch
    import pandas as pd
    from transformers import AutoModel
    
    #get padded tokens from the nlp_feature_vector/tokens
    padded_tokens = get_padded_tokens(data_df=data_df)
    
    # Attention mask (1 where token != PAD_TOKEN_ID, else 0)
    attention_mask = (padded_tokens != pad_token_id).long()

    # Load model (no tokenizer needed since data is already tokenized)
    model = AutoModel.from_pretrained(model_name)

    # Get embeddings
    embeddings = []
    for batch, batch_attention_mask in zip(padded_tokens, attention_mask):
        batch, batch_attention_mask = batch.unsqueeze(dim=0), batch_attention_mask.unsqueeze(dim=0)
        with torch.no_grad():
            outputs = model(
                input_ids=batch, 
                attention_mask=batch_attention_mask
            )
        #pull the embeddings
        token_emb = outputs.last_hidden_state
        #compress dimensions to make an embeddings vector
        _emb = (token_emb * batch_attention_mask.unsqueeze(-1)).sum(1) / batch_attention_mask.sum(1, keepdim=True) 
        embeddings.append(_emb)
        del outputs, token_emb, _emb #clean up to save memory

    #clean up
    del model

    # combine all embeddings
    embeddings_df = pd.DataFrame(torch.cat(embeddings).numpy())
    
    #store embeddings for training
    try:
        embeddings_df['UniqueID'] = data_df['UniqueID'] # add IDs back in
        embeddings_df.to_csv(embeddings_file_name) #save the embeddings
    except FileNotFoundError:
        raise FileNotFoundError('Folder does not exist at this location. Create a folder to save the embeddings file.')
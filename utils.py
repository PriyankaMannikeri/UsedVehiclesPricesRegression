

def label_encoding(df, columns):

    encoding_dict = {}
    for col in columns:
        df[col] = df[col].astype('category')

        # get per column label to encoded feat dict
        # example:
        # {2: 'good', 0: 'excellent', 3: 'like new', 5: 'salvage', 4: 'new', 1: 'fair'}
        encoding_dict[col] = dict(zip(df[col].cat.codes, df[col]))

        df[col] = df[col].cat.codes

    return df, encoding_dict

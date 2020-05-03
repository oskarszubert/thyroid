
from DataSet import *

if __name__ == "__main__":


    dataset = DataSet(filename='config/referral_source.txt')
    columns = []
    separator = ','
    # separator = ' '

    if separator == ' ':
        file = 'data/ann-train.data'
    else:
        file = 'data/thyroid0387.data'
        columns = dataset.read_list_from_file(filename='config/columns.txt')

    df = dataset.prepare_dataframe(filename=file, separator=separator,columns=columns)

    # if separator == ' ':
    #     percent = 70
    #     test_train_border = int(df.shape[0] * percent/100)

    dataset.univariate_selection(data=df, k_best=(len(df.loc[0]) - 1) )
    dataset.f_importance(data=df, n_attrs =(len(df.loc[0]) - 1) )
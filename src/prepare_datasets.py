
from DataSet import *

if __name__ == "__main__":
    dataset = DataSet(filename='../config/referral_source.txt')
    dataset.change_unused_value_to_mean = False
    df = dataset.prepare_dataframe(filename='../data/ann_thyroid.data', separator=' ', columns=[])
    df = dataset.normalize_value(df)
    dataset.save_dataset_to_file(df, 'ann_thyroid')


    dataset.change_unused_value_to_mean = True
    columns = dataset.read_list_from_file(filename='../config/columns.txt')
    df = dataset.prepare_dataframe(filename='../data/thyroid0387.data', separator=',',columns=columns)
    filename='thyroid0387'
    if dataset.change_unused_value_to_mean:
        filename = 'mean_' + filename
    else:
        filename = 'solid_' + filename
    dataset.save_dataset_to_file(df, filename)

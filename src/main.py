
from DataSet import *
from CrossValidation import *

if __name__ == "__main__":
    gen_dir = '../gen'
    if not os.path.exists( gen_dir ):
        exit(-1)
    result_dir = '../gen/results'
    if not os.path.exists( result_dir ):
        os.makedirs( result_dir )

    dataset = DataSet(filename='../config/referral_source.txt')

    files_proposal = ['ann_thyroid', 'mean_thyroid0387', 'solid_thyroid0387']
    ann_proposal = [[1],[5],[7],[1,3],[3,5],[3,3],[1,2,3],[3,5,7]]
    epochs__proposal = [5,10,20]
    momentum_proposal = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    sf_proposal = ['univ', 'impot']

    for filename in files_proposal:
        path_to_file = os.path.join( gen_dir, filename+'.csv')

        df = pd.read_csv(path_to_file, sep=',')

        sf_univarate =  dataset.univariate_selection(data=df, k_best=(len(df.loc[0]) - 1) )
        sf_univarate.insert(0, 'value')

        sf_importance =  dataset.f_importance(data=df, n_attrs =(len(df.loc[0]) - 1) )
        sf_importance.insert(0, 'value')


        nm = NetworkModel()
        ds = CrossValidation()

        result_list = []
        for feature_selection_name in sf_proposal:
            tmp_row = []
            if feature_selection_name == 'univ':
                f_selection = sf_univarate
            elif feature_selection_name == 'impot':
                f_selection = sf_importance
            for number_of_features in range(17, len(f_selection) ):
                for layers in ann_proposal:
                    for momentum in momentum_proposal:
                        model = nm.create_model(layers=layers,
                                                input_size=number_of_features-1, 
                                                output_size=len(set(df['value'])) ,
                                                momentum=momentum)

                        for epochs in epochs__proposal:
                            df_tmp = df[f_selection[:number_of_features]]
                            mean_acc = ds.cross_valid_and_fit_model(df_tmp,model,epochs=epochs)
                            tmp_row = [ filename,
                                        mean_acc, 
                                        epochs,
                                        momentum,
                                        layers,
                                        feature_selection_name,
                                        number_of_features-1]
                            print(tmp_row)
                            result_list.append(tmp_row)


        result_columns=['filename','acc','epochs','momentum','layers','feature_selection_name','number_of_features']

        df = pd.DataFrame(data=result_list, columns=result_columns)
        df.to_excel( os.path.join( result_dir, filename+'_results.xlsx') )

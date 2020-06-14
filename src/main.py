
from DataSet import *
from CrossValidation import *
from datetime import datetime
from tensorflow.keras import backend as K

if __name__ == "__main__":
    gen_dir = '../gen'
    if not os.path.exists( gen_dir ):
        exit(-1)
    result_dir = '../gen/results'
    if not os.path.exists( result_dir ):
        os.makedirs( result_dir )   

    dataset = DataSet(filename='../config/referral_source.txt')

    files_proposal = ['ann_thyroid', 'mean_thyroid0387', 'solid_thyroid0387']

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
        if filename == 'ann_thyroid':
            feature_selection_name = 'univ'
            number_of_features = 12
            ann_proposal = [[7], [5,20]]
            
            epochs__proposal = [5,10,15,20]
            # momentum_proposal = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            momentum_proposal =  [0.4]

        elif filename == 'mean_thyroid0387':
            feature_selection_name = 'impot'
            number_of_features = 2
            ann_proposal = [[5]]

            epochs__proposal = [5,10,15,20]
            # momentum_proposal = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            momentum_proposal =  [0.4]

        elif filename == 'solid_thyroid0387':
            feature_selection_name = 'univ'
            number_of_features = 3
            ann_proposal = [[5]]

            epochs__proposal = [5,10,15,20]
            # momentum_proposal = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            momentum_proposal =  [0.4]

        if feature_selection_name == 'univ':
            f_selection = sf_univarate
        elif feature_selection_name == 'impot':
            f_selection = sf_importance

        tmp_row = []
        for layers in ann_proposal:
            for momentum in momentum_proposal:
                model = nm.create_model(layers=layers,
                                        input_size=number_of_features, 
                                        output_size=len(set(df['value'])) ,
                                        momentum=momentum)
                for epochs in epochs__proposal:
                    df_tmp = df[f_selection[:number_of_features+1]]
                    mean_acc, mean_f1 = ds.cross_valid_and_fit_model(df_tmp,model,epochs=epochs)
                    tmp_row = [ filename,
                                mean_acc, 
                                mean_f1, 
                                epochs,
                                momentum,
                                layers,
                                feature_selection_name,
                                number_of_features]
                    print(tmp_row)
                    result_list.append(tmp_row)

        result_columns=['filename','acc','f1_score','epochs','momentum','layers','feature_selection_name','number_of_features']

        df = pd.DataFrame(data=result_list, columns=result_columns)
        df.to_excel( os.path.join( result_dir, filename+'_results' + datetime.now().strftime('%H_%M_%S') + '.xlsx') )

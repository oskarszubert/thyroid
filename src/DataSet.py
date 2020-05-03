import re
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle

# import matplotlib.pyplot as plt

class DataSet:
    def __init__(self, filename):
        self.referral_source = self.create_dict_from_file(filename)
        self.delete_row_with_unused_data = False   # do not use it, just 4fun
        self.change_unused_value_to_mean = True


    def read_file(self, filename, separator):
        try:
            file = open(filename, 'r')
            dataframe = []
            for line in file:
                if separator == ',':
                    line = re.sub(r'\[\d*\]','', line )    # remove [number] from end
                    line = self.cleanup(line=line.rstrip(), separator=separator)
                else:
                    line = line.rstrip().split(separator)
                
                if line:
                    dataframe.append( line )  # remove newline symbol and split into single elements
        except Exception as e:
            print(e)
            exit(f'Error! while reading {filename}.')

        return dataframe


    def cleanup(self, line, separator):
        tmp_element_list = []
        if self.delete_row_with_unused_data:
            if '?' in line:
                return None
            

        for element in line.split(separator):
            if element == 't':
                tmp_element_list.append(1)
            elif element == 'f':
                tmp_element_list.append(0)
            elif element == 'M':
                tmp_element_list.append(0)
            elif element == 'F':
                tmp_element_list.append(1)
            elif element in self.referral_source:
                tmp_element_list.append( self.referral_source[ element ] )
            elif element == '?':
                if self.change_unused_value_to_mean:
                    tmp_element_list.append(np.NaN)
                else:
                    tmp_element_list.append(100)
            elif element == '-':
                tmp_element_list.append(0)
            elif len(element) == 1 and ( ord('A') <= ord(element) <= ord('T') ): 
                tmp_element_list.append( ord(element) - ord('A') + 1 ) # coz 0 is reserved for negative(-)
            elif not self.is_number(element):
                tmp_element_list.append( ord(element[0]) - ord('A') + 1 ) # coz 0 is reserved for negative(-)
            elif self.is_number(element):
                tmp_element_list.append( float(element) )
            else:
                tmp_element_list.append(element)

        if '?' in line:
            tmp_element_list.insert(-1, 1)
        else:
            tmp_element_list.insert(-1, 0)
        return tmp_element_list


    @staticmethod
    def is_number( element ):
        try:
            float(element)
            return True
        except:
            return False


    @staticmethod
    def create_name_of_columns(size_of_vector):
        attr = ['attr_'+str(i) for i in range( size_of_vector - 1 )]
        attr.append('value')
        return attr


    @staticmethod
    def read_list_from_file(filename):     # it can be used for reading columns name from file
        elements = []
        try:
            file = open(filename, 'r')
            for line in file:
                elements.append( line.rstrip() )
        except Exception as e:
            print(e)
            exit(f'Error! while reading {filename}.')
        return elements


    def create_dict_from_file(self, filename):
        elements_list = self.read_list_from_file(filename)
        tmp_dict = {}
        elements_list = set(elements_list)
        couter = 1
        for element in elements_list:
            if element not in tmp_dict:
                tmp_dict[ element ] = couter
                couter += 1
        return tmp_dict


    def prepare_dataframe(self, filename, separator, columns=[]):
        data = self.read_file(filename=filename, separator=separator)
        if not columns:
            columns = self.create_name_of_columns(len(data[0])) # take as argument first row and calculate size of vec, 
                                                        # [NOTE] last element is class of this vector 

        df = pd.DataFrame(data=data, columns=columns)

        if self.change_unused_value_to_mean:
            for single_column in df.columns:
                df[single_column] = df[single_column].replace(np.NaN, df[single_column].mean())
                
        return shuffle(df)     # shuffle df at the end



    def univariate_selection(self, data, k_best):
        X = data.iloc[:,0:k_best]  #independent columns
        y = data.iloc[:,-1]    #target column i.e price range

        #apply SelectKBest class to extract top 10 best features
        bestfeatures = SelectKBest(score_func=chi2, k=k_best)
        fit = bestfeatures.fit(X,y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization 
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Specs','Score']  #naming the dataframe columns
        print('Univariate Selection (chi2):')
        print(featureScores.nlargest(k_best,'Score'))


    def f_importance(self, data, n_attrs):
        X = data.iloc[:,0:n_attrs]  #independent columns
        y = data.iloc[:,-1]    #target column i.e price range
        model = ExtraTreesClassifier()
        model.fit(X,y)
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]

        data = []
        index = [i for i in indices]
        for i in indices:
            data.append([X.columns[i],  importance[i] ])

        df = pd.DataFrame(data=data, columns=['Specs','Score'], index=index)
        print('Feature Importance (Extra Tree Classifier):')
        print(df.to_string())

from numpy.core.fromnumeric import around
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

class Model():
    def __init__(self):
        shap.initjs()

        self.digital_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate',
         'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
         'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
         'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
         'YearsSinceLastPromotion', 'YearsWithCurrManager']

        self.category_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        
        self.train_col = ['OverTime Yes', 'YearsWithCurrManager', 'NumCompaniesWorked',
       'EnvironmentSatisfaction', 'Compare MaritalStatus with DailyRate',
       'Compare BusinessTravel with MonthlyIncome', 'EducationField Medical',
       'Compare JobInvolvement with MonthlyIncome', 'YearsSinceLastPromotion',
       'Compare Age with DailyRate', 'JobRole Laboratory Technician',
       'BusinessTravel Non-Travel',
       'Compare NumCompaniesWorked with DailyRate', 'WorkLifeBalance',
       'EducationField Life Sciences', 'Compare OverTime with DailyRate',
       'DailyRate', 'Compare EnvironmentSatisfaction with MonthlyIncome',
       'quantile of DailyRate', 'Compare JobRole with StockOptionLevel',
       'quantile of MonthlyIncome', 'Gender Female',
       'RelationshipSatisfaction', 'JobLevel', 'JobInvolvement',
       'EducationField Technical Degree', 'quantile of DistanceFromHome',
       'HourlyRate', 'MonthlyRate', 'EducationField Marketing',
       'StockOptionLevel', 'quantile of TotalWorkingYears',
       'quantile of YearsWithCurrManager', 'Department Sales', 'Age',
       'quantile of YearsInCurrentRole', 'Compare Age with StockOptionLevel',
       'JobSatisfaction', 'quantile of HourlyRate', 'YearsAtCompany']

        self.df = pd.read_csv('../train.csv')
        self.LogisticRegression = joblib.load('./model/Logistic_model')
        self.SVM = joblib.load('./model/SVM_model')
        self.MinMaxScaler = joblib.load('./model/minmax.save') 
        self.mask, self.encoder, self.decoder = self.__labelencode(self.df, self.category_cols)

        self.cols = -1

        self.mask = self.__preprocess(self.mask.drop(['Attrition'], axis=1))
    
    def predict(self, x, mode='nonplot'):
        user_id = x['user_id']

        x = self.__preprocess(x)

        lr_pred_y1 = self.LogisticRegression.predict_proba(x[self.train_col])

        svc_pred_y1 = self.SVM.predict_proba(x[self.train_col])

        # ensemble
        predict = pd.DataFrame((svc_pred_y1[:,0] + lr_pred_y1[:,0])/2, columns=['Attrition'])

        predict = (predict * 100).apply(around)

        output = pd.concat([user_id, predict], axis=1)

        output = output.to_dict('records')

        if mode == 'plot':
            img_base64 = self.__plot(self.LogisticRegression, x)
            return output, img_base64

        return output

    def __preprocess(self, x):
        
        user_id = x['user_id']
        x = x.drop(columns=['EmployeeCount', 'Over18', 'StandardHours', 'user_id', 'EmployeeNumber'])
        x = self.__AddCompareFeature(x, ['Age', 'BusinessTravel', 'Department', 'EnvironmentSatisfaction', 'EducationField', 'JobLevel', 'JobInvolvement', 'JobRole', 'WorkLifeBalance', 'OverTime', 'NumCompaniesWorked', 'MaritalStatus'], ['MonthlyIncome', 'DailyRate', 'StockOptionLevel'])
        x = self.__apply_quater(x, ['Age','DailyRate', 'DistanceFromHome','HourlyRate','MonthlyIncome','MonthlyRate','PercentSalaryHike','TotalWorkingYears','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager'])

        x = x.drop(columns=['MonthlyIncome','TotalWorkingYears'])

        dummies = self.__onehotencode(x, self.category_cols, self.encoder)
        x = pd.concat([x.drop(self.category_cols,axis=1), dummies],axis=1)

        self.cols = x.columns.to_list() if self.cols == -1 else self.cols

        x = self.MinMaxScaler.transform(x)

        x = pd.DataFrame(x, columns=self.cols)
        
        return x

    def __plot(self, model, x):
        explainer = shap.LinearExplainer(model, self.mask[self.train_col])
        shap_values = explainer.shap_values(x[self.train_col])
        x_array = x[self.train_col].to_numpy()
        ind = 0
        shap.force_plot(
            explainer.expected_value, shap_values[ind,:], x_array[ind,:], text_rotation=20,
            feature_names=x[self.train_col].columns.values, show=False, matplotlib=True
        )
        output = io.BytesIO()
        plt.savefig(output, transparent=True, bbox_inches="tight", format='png')
        output.seek(0)
        img_base64 = base64.b64encode(output.read())
        
        return img_base64

    # 比較薪資等數據資料
    def __AddCompareFeature(self, df, col1, col2):
        for i in col1:
            for j in col2:
                concat_df = pd.DataFrame()
                temp = pd.DataFrame()
                for k in df[i].unique():
                    
                    if(abs(self.df[self.df[i] == k][j].skew()) > 0.5):
                        mean_value = self.df[self.df[i] == k][j].median()
                    else:
                        mean_value = self.df[self.df[i] == k][j].mean()   

                    temp = df[df[i] == k][j] < mean_value
                    concat_df = pd.concat([concat_df, temp], axis=0)
                concat_df = concat_df.sort_index()
                df = pd.concat([df, concat_df], axis = 1)
                df = df.rename(columns = {0:'Compare ' + i + ' with ' + j})
                df = df.replace({True:1, False:0})
        return df
    
    def __apply_quater(self, df, col):
        for i in col:
            temp1 = pd.DataFrame()
            temp2 = pd.DataFrame()
            temp3 = pd.DataFrame()
            temp4 = pd.DataFrame()
            quantile = self.df.quantile([0.25, 0.5, 0.75], axis = 0)
            
            temp1['quantile of ' + i] = df[i] <= quantile.loc[0.25, i]
            temp2['quantile of ' + i] = (df[i] <= quantile.loc[0.5, i]).mul(df[i] > quantile.loc[0.25, i])
            temp3['quantile of ' + i] = (df[i] <= quantile.loc[0.75, i]).mul(df[i] > quantile.loc[0.5, i])
            temp4['quantile of ' + i] = df[i] > quantile.loc[0.75, i] 

            temp1 = temp1.replace({True:1, False:0})
            temp2 = temp2.replace({True:2, False:0})
            temp3 = temp3.replace({True:3, False:0})
            temp4 = temp4.replace({True:4, False:0})
            
            temp4 = temp4.add(temp3)
            temp4 = temp4.add(temp2)
            temp4 = temp4.add(temp1)

            df = pd.concat([df, temp4], axis=1)
        return df

    def __labelencode(self, df, col):
        encoder_nums = {}
        decoder_nums = {}
        for i in col:
            order = 1
            en_coldict = {}
            de_coldict = {}
            for j in df[i].unique():
                if j != ' NaN':
                    en_coldict[j] = order
                    de_coldict[order] = j
                    order = order + 1

            encoder_nums[i] = en_coldict
            decoder_nums[i] = de_coldict
            
        df = df.replace(encoder_nums)

        return df, encoder_nums, decoder_nums

    def __onehotencode(self, df, col, encoder):
        new_col = []
        for i in encoder.keys():
            for j in encoder[i].keys():
                new_col.append(i + ' ' + j)

        temp = pd.DataFrame(np.zeros((len(df), len(new_col))), columns=new_col)
        
        for i in encoder.keys():
            for j in encoder[i].keys():
                idx = df[df[i] == encoder[i][j]].index
                temp[i + ' ' + j].loc[idx] = 1 
            
            temp = temp.drop(columns=[i + ' ' + j])
        return temp

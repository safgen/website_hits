import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.feature_extraction import FeatureHasher


class DataAnalysis:
    def __init__(self, df):
        self.df = df
        self.df_clean = self.clean_df(df_unclean=df)
        self.df_encoded = self.feature_encode(df_cl=self.df_clean)
        self.df_hits, self.df_no_hits = self.separate_data(df_enc=self.df_encoded, col_name='hits')

    def remove_extreme_outliers(self, df):
        q_sd = df['session_duration'].quantile(0.98)
        q_h = df['hits'].quantile(0.98)
        df_not_empty, df_empty = self.separate_data(df_enc=df, col_name='hits')
        df_clean = df_not_empty[df_not_empty['session_duration'] < q_sd]
        df_clean = df_clean[df_clean['hits'] < q_h]
        df_clean = pd.concat([df_clean, df_empty], axis=0)
        return df_clean

    def clean_df(self, df_unclean):
        df = df_unclean.copy()
        # fixing the typo in the column name
        df = df.rename(columns={'session_durantion': 'session_duration'})
        df = df.replace('\\N', np.nan)
        # Filling in the missing values in session_duration
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(df[['session_duration']])
        df['session_duration'] = imputer.transform(df[['session_duration']])
        # changing to a numeric type after filling in the missing values
        df['session_duration'] = df['session_duration'].astype(float)
        df['hits'] = df['hits'].astype(float)
        df_clean = self.remove_extreme_outliers(df)
        return df_clean

    @classmethod
    def feature_encode(cls, df_cl):
        # since hits have a significant correlation with the length of path id set (see data_check.ipynb),
        # introducing a new feature for path length
        def get_path_length(x):
            if x is not np.nan:
                x = str(x)
                y = len(x.split(';'))
            else:
                y = 0
            return y
        df_cl['path_length'] = df_cl['path_id_set'].apply(lambda x: get_path_length(x))
        df_encoded = df_cl.copy()
        # First, converting all the nominal categorical variables to one hot encoding
        for i in ('locale', 'agent_id', 'traffic_type'):
            dummies = pd.get_dummies(df_cl[i], drop_first=True, prefix=i.split('_')[0])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[i])

        # Second, feature hasher for variables with too many category values
        hasher = FeatureHasher(n_features=10, input_type='string')
        hashed_features = hasher.fit_transform(df_encoded['entry_page'].astype(str))
        hashed_features = hashed_features.toarray()
        df_encoded = pd.concat([df_encoded.drop(columns=['entry_page']), pd.DataFrame(hashed_features,
                                                                                      columns=['ep_0', 'ep_1', 'ep_2',
                                                                                               'ep_3', 'ep_4', 'ep_5',
                                                                                               'ep_6', 'ep_7', 'ep_8',
                                                                                               'ep_9'])], axis=1)
        hasher = FeatureHasher(n_features=10, input_type='string')
        hashed_features = hasher.fit_transform(df_encoded['path_id_set'].astype(str))
        hashed_features = hashed_features.toarray()

        df_encoded = pd.concat([df_encoded.drop(columns=['path_id_set']), pd.DataFrame(hashed_features,
                                                                                       columns=['pid_0', 'pid_1',
                                                                                                'pid_2', 'pid_3',
                                                                                                'pid_4', 'pid_5',
                                                                                                'pid_6', 'pid_7',
                                                                                                'pid_8',
                                                                                                'pid_9'])], axis=1)



        # Third, the continuous features
        scaler = MinMaxScaler()
        df_encoded[['path_length', 'session_duration']] = scaler.fit_transform(
            df_cl[['path_length', 'session_duration']])

        # Finally, the cyclic features
        day_mapper = dict(zip(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'],
                              [0, 1, 2, 3, 4, 5, 6]))
        df_encoded['day_of_week_enc'] = df_encoded['day_of_week'].map(day_mapper)
        df_encoded['day_sin'] = np.sin(2 * np.pi * df_encoded.day_of_week_enc / 7)
        df_encoded['day_cos'] = np.cos(2 * np.pi * df_encoded.day_of_week_enc / 7)
        df_encoded['hour_sin'] = np.sin(2 * np.pi * df_encoded.hour_of_day / 24)
        df_encoded['hour_cos'] = np.cos(2 * np.pi * df_encoded.hour_of_day / 24)
        df_encoded = df_encoded.drop(columns=['day_of_week', 'day_of_week_enc', 'hour_of_day'])

        return df_encoded

    @classmethod
    def separate_data(cls, df_enc, col_name='hits', write=False):
        df_with_values = df_enc[df_enc[col_name].notna()]
        df_without_values = df_enc[df_enc[col_name].isna()]
        if write:
            df_with_values.to_csv('feature_engineered_data_with_hits.csv')
            df_without_values.to_csv('feature_engineered_data_without_hits.csv')

        return df_with_values, df_without_values


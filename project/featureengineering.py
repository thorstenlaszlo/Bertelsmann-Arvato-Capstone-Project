from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from common_functions import is_numeric, load_model, load_dataframe, save_model, save_dataframe


class Documentation:
    """
    The Documentation class wraps the content of the 'DIAS Attributes - Values 2017.xlsx' file in a dictionary.
    """
    _meta_dict: dict

    def __init__(self, df):
        self._meta_dict = Documentation._rename_columns(
            Documentation._rename_rz_columns(
                Documentation._generate_metadata_dict(df)
            )
        )
        self._meta_dict = self._add_numeric_keys()
        self._meta_dict = self._recode_cameo_intl()
        self._meta_dict = self._recode_familie_grob()
        self._meta_dict = self._recode_lp_status_grob()
        self._meta_dict = self._recode_o_w()
        self._meta_dict = self._recode_kosnumtyp()

    @property
    def meta_dict(self):
        """
        The Documentation as dictionary
        :return: The Documentation as dictionary
        """
        return self._meta_dict

    def find_unknown_values(self, unknown_values_texts=None):
        """
        Find the values (=item-keys) of entries labeled with 'unknown' and other texts
        :param unknown_values_texts:
        :return: Dictionary of the unknown values
        """
        if unknown_values_texts is None:
            unknown_values_texts = ['unknown / no main age detectable', 'unknown']
        unknown_values = {}
        for key, value in self.meta_dict.items():
            for i_key, i_value in value['items'].items():
                if i_value in unknown_values_texts:
                    if not key in unknown_values:
                        unknown_values[key] = set([])
                    unknown_values[key].add(int(i_key))
                    unknown_values[key].add(str(int(i_key)))
                    unknown_values[key].add(float(i_key))
                    unknown_values[key].add(str(float(i_key)))
        return unknown_values

    def _find_attributes_with_non_numeric_keys(self):
        attributes_with_non_numeric_keys = {}
        for key, value in self.meta_dict.items():
            for it_key, it_value in value['items'].items():
                if not is_numeric(it_key):
                    if not key in attributes_with_non_numeric_keys:
                        attributes_with_non_numeric_keys[key] = []
                    attributes_with_non_numeric_keys[key].append((it_key, it_value))
        return attributes_with_non_numeric_keys

    def _recode_cameo_intl(self):
        """
        Convert the feature 'CAMEO_INTL_2015' into two features 'CAMEO_INTL_2015_WEALTH','CAMEO_INTL_2015_FAMILY_AGE'
        and update the Documentation-dictionary.
        :return: The Documentation-dictionary
        """
        self._meta_dict['CAMEO_INTL_2015_WEALTH'] = {'desc': '', 'item_keytype': 'int',
                                                     'items': Mappings.map_cameo_intl_wealth_inverse}
        self._meta_dict['CAMEO_INTL_2015_FAMILY_AGE'] = {'desc': '', 'item_keytype': 'int',
                                                         'items': Mappings.map_cameo_intl_family_age_inverse}
        return self._meta_dict

    def _recode_familie_grob(self):
        """
        Recode the feature 'LP_FAMILIE_GROB' and update the Documentation-dictionary.
        :return: The Documentation-dictionary
        """
        self._meta_dict['LP_FAMILIE_GROB']['items'] = Mappings().map_family_inverse
        return self._meta_dict

    def _recode_lp_status_grob(self):
        """
        Recode the feature 'LP_STATUS_GROB' and update the Documentation-dictionary.
        :return: The Documentation-dictionary
        """
        self._meta_dict['LP_STATUS_GROB']['items'] = Mappings().map_status_inverse
        return self._meta_dict

    def _recode_o_w(self):
        """
        Convert 'OST_WEST_KZ' into a numeric feature and update the Documentation-dictionary.
        :return: The Documentation-dictionary
        """
        self._meta_dict['OST_WEST_KZ']['items'] = Mappings().map_o_w_inverse
        return self._meta_dict

    def _recode_kosnumtyp(self):
        """
        adjust spacing
        we have
        1	Universal
        2	Versatile
        3	Gourmet
        4	Family
        5	Informed
        6	Modern
        9	Inactive
        This method replaces the mapping [9]->'Incative' with [7]->Inactive
        :return: the internal meta_dict
        """
        self._meta_dict['D19_KONSUMTYP']['items'][7] = self._meta_dict['D19_KONSUMTYP']['items'].pop(9)
        return self._meta_dict

    def _add_numeric_keys(self):
        """
        Convert the type of item keys to int where possible
        :return: the internal meta_dict
        """
        non_numeric_keys = self._find_attributes_with_non_numeric_keys()
        for key in self._meta_dict.keys():
            if key in non_numeric_keys:
                self._meta_dict[key]['item_keytype'] = 'str'
            else:
                self._meta_dict[key]['item_keytype'] = 'int'
                self._meta_dict[key]['items'] = {int(k): v for k, v in self._meta_dict[key]['items'].items()}
        return self._meta_dict

    @staticmethod
    def _generate_metadata_dict(df: pd.DataFrame):
        """
        Convert the DataFrame of the EXCEL 'DIAS Attributes - Values 2017.xlsx' to the internal meta_dict
        :param df: df the EXCEL as DataFrame
        :return: the internal meta_dict
        """
        key = None
        last_valid_val = None
        meta_dict = {}
        for _, row in df.iterrows():
            if not pd.isna(row[1]):
                key = row[1]
                meta_dict[key] = {
                    'desc': row[2],
                    'items': {},
                }
            if not pd.isna(row[4]):
                last_valid_val = str(row[4]).strip()
            if key:
                if isinstance(row[3], str):
                    vals = list(map(lambda it: str(it).strip(), row[3].split(',')))
                    for val in vals:
                        meta_dict[key]['items'][val] = str(row[4]).strip()
                else:
                    meta_dict[key]['items'][str(row[3]).strip()] = last_valid_val
        return meta_dict

    @staticmethod
    def _rename_columns(meta_dict: dict):
        """
        Rename a few columns in order to cope with the column names in the DataFrames
        :param meta_dict: our meta_dict
        :return: the modifies meta_dict
        """
        meta_dict['CAMEO_INTL_2015'] = meta_dict.pop('CAMEO_DEUINTL_2015')
        meta_dict['D19_BUCH_CD'] = meta_dict.pop('D19_BUCH')
        meta_dict['KBA13_CCM_1401_2500'] = meta_dict.pop('KBA13_CCM_1400_2500')
        meta_dict['SOHO_KZ'] = meta_dict.pop('SOHO_FLAG')
        return meta_dict

    @staticmethod
    def _rename_rz_columns(meta_dict: dict):
        """
        Remove the '_RZ' suffix from the columns in order to cope with the column names in the DataFrames
        :param meta_dict: our meta_dict
        :return: the modifies meta_dict
        """
        keys_to_rename = list(filter(lambda it: '_RZ' in str(it), meta_dict.keys()))
        for key_to_rename in keys_to_rename:
            meta_dict[str(key_to_rename).replace('_RZ', '')] = meta_dict.pop(key_to_rename)
        return meta_dict


class FeatureEngineer:
    _additional_customer_columns = ['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP']
    _numeric_columns = [
        'ALTER_KIND1',
        'ALTER_KIND2',
        'ALTER_KIND3',
        'ALTER_KIND4',
        'ANZ_HAUSHALTE_AKTIV',
        'ANZ_HH_TITEL',
        'ANZ_KINDER',
        'ANZ_PERSONEN',
        'ANZ_STATISTISCHE_HAUSHALTE',
        'ANZ_TITEL',
        'ARBEIT',
        # 'EINGEZOGENAM_HH_JAHR', removed after revision
        'EXTSEL992',
        'GEBURTSJAHR',
        'KBA13_ANZAHL_PKW',
        'VERDICHTUNGSRAUM',
    ]
    _documentation = None

    def __init__(self, documentation=None):
        self._documentation = documentation

    def drop_unneeded_customer_columns(self, customers: pd.DataFrame):
        """
        Drop the unneeded columns ['CUSTOMER_GROUP', 'ONLINE_PURCHASE', 'PRODUCT_GROUP']  from the
        customers DataFrame
        :param customers:
        :return: customers with unneeded columns removed
        """
        customers = customers.drop(columns=self._additional_customer_columns, axis=1)
        return customers

    def clean(self, df: pd.DataFrame):
        """
        Perform cleaning of the DataFrame:
        - drop unneeded columns,
        - replace invalid values by np.nan,
        - ensure consistent feature spacing,
        - remove unknown values (due to documentation)
        :param df: the DataFrame to clean
        :return: The cleaned DataFrame
        """
        df = FeatureEngineer._drop_unneeded_columns(df)
        df = FeatureEngineer._invalid_values_to_nan(df)
        df = FeatureEngineer._ensure_consistent_spacing(df)
        df = self._remove_unknown_values(df)
        return df

    def drop_missing_and_recode_special_features(self, df: pd.DataFrame,
                                                 threshold=.32, columns_to_keep=None,
                                                 convert_to_float=True):
        if columns_to_keep is None:
            columns_to_keep = ['GEBURTSJAHR']

        df = self._drop_missing_columns(df, threshold, columns_to_keep)
        df = FeatureEngineer._recode_o_w(df)
        df = self._recode_cameo_intl(df)
        df = FeatureEngineer._recode_konsumtyp(df)

        if convert_to_float:
            return df.astype('float32')
        return df

    @staticmethod
    def columns_with_stddev_below_threshold(df, threshold=0.25):
        result = []
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in df.select_dtypes(include=numeric_dtypes).columns:
            if df[col].std() < threshold:
                result.append(col)
        return result

    def outlier_columns(self, df, cols=[], multiplier=2.2, tolerance=0.03):
        outliers = []
        cols_ = cols if cols else self._numeric_columns
        for col in cols_:
            upper_outlier = FeatureEngineer._upper_outliers_column_values(df, col, multiplier)
            lower_outlier = FeatureEngineer._lower_outliers_column_values(df, col, multiplier)
            if lower_outlier['outlier_pct_count'] > tolerance \
                    or upper_outlier['outlier_pct_count'] > tolerance:
                outliers.append(col)
        return outliers

    @staticmethod
    def handle_outliers(df, columns, multiplier=2.2):
        df_ = df.copy(deep=True)
        for col in columns:
            upper_outlier = FeatureEngineer._upper_outliers_column_values(df_, col, multiplier)
            lower_outlier = FeatureEngineer._lower_outliers_column_values(df_, col, multiplier)
            upper_limit = upper_outlier.loc['limit'] if upper_outlier.loc['is_outlier'] else 1000000.0
            lower_limit = lower_outlier.loc['limit'] if lower_outlier.loc['is_outlier'] else -1000000.0
            df_[col] = df_[col].apply(
                lambda it: FeatureEngineer._lambda_replace_outliers_by_thresholds(it, lower_limit, upper_limit))
        return df_

    @staticmethod
    def default_imputer(df, filename):
        try:
            imputer = load_model(filename)
            print('Loaded persisted imputer...')
        except:
            print('Training imputer...')
            imputer = FeatureEngineer.fit_imputer(
                df,
                n_nearest_features=4,
                tol=0.1,
                imputation_order='ascending',
                initial_strategy='mean'
            )
            save_model(imputer, filename)
        print('Done')
        return imputer

    @staticmethod
    def fit_imputer(df: pd.DataFrame,
                    n_nearest_features=3, random_state=None,
                    tol=0.08, imputation_order='ascending',
                    initial_strategy='median', sample_frac=1.0):
        df_metrics = df.describe()
        imputer = IterativeImputer(max_iter=30,
                                   random_state=random_state,
                                   sample_posterior=False,
                                   n_nearest_features=n_nearest_features,
                                   skip_complete=True,
                                   tol=tol,  # tol=0.065,#
                                   imputation_order=imputation_order,
                                   min_value=df_metrics.loc['min', :],
                                   max_value=df_metrics.loc['max', :],
                                   initial_strategy=initial_strategy,
                                   verbose=2)
        if sample_frac < 1.0:
            imputer.fit(df.sample(frac=sample_frac, random_state=42))
        else:
            imputer.fit(df)
        return imputer

    @staticmethod
    def impute_geburtsjahr(df, filename):
        relevant_columns = [
            'ALTERSKATEGORIE_FEIN',
            'GEBURTSJAHR',
            'KOMBIALTER',
            'CJT_TYP_1',
            'CJT_TYP_2',
        ]
        test_df = df[relevant_columns].copy(deep=True)
        try:
            imputer = load_model(filename)
        except:
            df_metrics = test_df.describe()
            imputer = IterativeImputer(max_iter=10,
                                       random_state=0,
                                       imputation_order='ascending',
                                       sample_posterior=True,
                                       skip_complete=True,
                                       min_value=df_metrics.loc['min', :],
                                       max_value=df_metrics.loc['max', :],
                                       initial_strategy='mean',
                                       tol=0.001,
                                       verbose=2)
            imputer.fit(test_df)
            save_model(imputer, filename)
        result = pd.DataFrame(data=imputer.transform(test_df),
                              index=test_df.index,
                              columns=test_df.columns)
        return result.round(0)

    @property
    def numeric_columns(self):
        return self._numeric_columns

    @staticmethod
    def _drop_unneeded_columns(df: pd.DataFrame):
        drop_cols = [
            'EINGEFUEGT_AM',
            'MIN_GEBAEUDEJAHR',
            'LP_LEBENSPHASE_FEIN',
            'LP_STATUS_FEIN',
            'LP_FAMILIE_FEIN',
            'PRAEGENDE_JUGENDJAHRE',
            'CAMEO_DEU_2015',
            'GEMEINDETYP',  # odd spacing undocumented
            'EINGEZOGENAM_HH_JAHR',  # low value range from 1994 to 2017
            'KBA13_BJ_1999',
            'KBA13_BJ_2000',
            'KBA13_BJ_2004',
            'KBA13_BJ_2006',
            'KBA13_BJ_2008',
            'KBA13_BJ_2009',
            'KBA13_HALTER_20',
            'KBA13_HALTER_25',
            'KBA13_HALTER_30',
            'KBA13_HALTER_35',
            'KBA13_HALTER_40',
            'KBA13_HALTER_45',
            'KBA13_HALTER_50',
            'KBA13_HALTER_55',
            'KBA13_HALTER_60',
            'KBA13_HALTER_65',
            'KBA13_HALTER_66',
            'KBA13_CCM_1000',
            'KBA13_CCM_1200',
            'KBA13_CCM_1400',
            'KBA13_CCM_1500',
            'KBA13_CCM_1600',
            'KBA13_CCM_1800',
            'KBA13_CCM_2000',
            'KBA13_CCM_2500',
            'KBA13_CCM_3000',
            'KBA13_CCM_3001',
            'KBA13_KMH_110',
            'KBA13_KMH_140',
            'KBA13_KMH_180',
            'KBA13_KMH_210',
            'KBA13_KMH_250',
            'KBA13_KMH_251',
            'KBA13_KW_30',
            'KBA13_KW_40',
            'KBA13_KW_50',
            'KBA13_KW_60',
            'KBA13_KW_70',
            'KBA13_KW_80',
            'KBA13_KW_90',
            'KBA13_KW_110',
            'KBA13_KW_120',

            'KBA13_BMW',
            'KBA13_MERCEDES',
            'KBA13_AUDI',
            'KBA13_VW',
            'KBA13_FORD',
            'KBA13_OPEL',
            'KBA13_FIAT',
            'KBA13_PEUGEOT',
            'KBA13_RENAULT',
            'KBA13_MAZDA',
            'KBA13_NISSAN',
            'KBA13_TOYOTA',
            'KBA13_ANTG1',  # undocumented we have KBA05_ANTG1
            'KBA13_ANTG2',  # undocumented we have KBA05_ANTG1
            'KBA13_ANTG3',  # undocumented we have KBA05_ANTG1
            'KBA13_ANTG4',  # undocumented we have KBA05_ANTG1
            'KBA13_BAUMAX',  # undocumented we have KBA05_BAUMAX
            'KBA13_GBZ',  # undocumented we have KBA05_GBZ
            'KBA05_ZUL1',
            'KBA05_ZUL2',
            'KBA05_ZUL3',
            'KBA05_ZUL4',
        ]
        return df.drop(
            columns=drop_cols, axis=1
        )

    @staticmethod
    def drop_unneeded_after_birthyear_reconstruction(df):
        df_ = df.copy(deep=True)
        drop_cols = [
            'ALTERSKATEGORIE_GROB',
            'ALTERSKATEGORIE_FEIN'
        ]
        return df_.drop(columns=drop_cols, axis=1)

    def _drop_missing_columns(self, df: pd.DataFrame, threshold, columns_to_keep):
        cols = self._missing_percentage_columns(df, threshold, columns_to_keep)
        return df.drop(columns=cols, axis=1)

    @staticmethod
    def _upper_outliers_column_values(df, column_name, multiplier=2.2):
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        upper_lim = q3 + (iqr * multiplier)
        num_all = df[column_name].apply(lambda it: it if pd.isnull(it) else 1).sum()
        num_upper_out = df[column_name][df[column_name] > upper_lim].apply(lambda it: it if pd.isnull(it) else 1).sum()

        df_non_null = df[column_name].dropna()
        val_cnts = df_non_null.value_counts()  # .sort_index()
        return pd.Series({
            'is_outlier': num_upper_out > 0,
            'limit': upper_lim,
            'outlier_abs_count': num_upper_out,
            'outlier_pct_count': num_upper_out / num_all,
            'column_value_cnt': df_non_null.count(),
            'column_distinct_count': len(val_cnts),
            'mean': round(df[column_name].mean(), 4),
            'std': round(df[column_name].std(), 4),
            'min': round(df[column_name].min(), 4),
            'q1': round(q1, 4),
            'median': round(df[column_name].quantile(0.5), 4),
            'q3': round(q3, 4),
            'max': round(df[column_name].max(), 4),
        })

    @staticmethod
    def _lower_outliers_column_values(df, column_name, multiplier=2.2):
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_lim = q1 - (iqr * multiplier)
        num_all = df[column_name].apply(lambda it: it if pd.isnull(it) else 1).sum()
        num_lower_out = df[column_name][df[column_name] < lower_lim].apply(lambda it: it if pd.isnull(it) else 1).sum()

        df_non_null = df[column_name].dropna()
        val_cnts = df_non_null.value_counts()
        return pd.Series({
            'is_outlier': num_lower_out > 0,
            'limit': lower_lim,
            'outlier_abs_count': num_lower_out,
            'outlier_pct_count': num_lower_out / num_all,
            'column_value_cnt': df_non_null.count(),
            'column_distinct_count': len(val_cnts),
            'mean': round(df[column_name].mean(), 4),
            'std': round(df[column_name].std(), 4),
            'min': round(df[column_name].min(), 4),
            'q1': round(q1, 4),
            'median': round(df[column_name].quantile(0.5), 4),
            'q3': round(q3, 4),
            'max': round(df[column_name].max(), 4),
        })

    # https://datatest.readthedocs.io/en/stable/how-to/outliers.html
    @staticmethod
    def _lambda_replace_outliers_by_thresholds(val, lower_lim, upper_lim):
        if pd.isnull(val):
            return val
        if val < lower_lim:
            return lower_lim
        if val > upper_lim:
            return upper_lim
        return val

    @staticmethod
    def _invalid_values_to_nan(df: pd.DataFrame):
        # we could also use replace but replace checks for type equality in addition
        df['CAMEO_DEUG_2015'] = df['CAMEO_DEUG_2015'].apply(lambda it: np.nan if it == 'X' else it)
        df['CAMEO_INTL_2015'] = df['CAMEO_INTL_2015'].apply(lambda it: np.nan if it == 'XX' else it)
        df['LP_FAMILIE_GROB'] = df['LP_FAMILIE_GROB'].apply(lambda it: np.nan if it == 0.0 else it)
        df['LP_LEBENSPHASE_GROB'] = df['LP_LEBENSPHASE_GROB'].apply(lambda it: np.nan if it == 0.0 else it)
        df['ORTSGR_KLS9'] = df['ORTSGR_KLS9'].apply(lambda it: np.nan if it == 0.0 else it)
        df['GEBURTSJAHR'] = df['GEBURTSJAHR'].apply(lambda it: np.nan if it == 0.0 else it)
        df['ALTERSKATEGORIE_FEIN'] = df['ALTERSKATEGORIE_FEIN'].apply(lambda it: np.nan if it == 0.0 else it)
        df['KOMBIALTER'] = df['KOMBIALTER'].apply(lambda it: np.nan if it == 9.0 else it)
        df['VERDICHTUNGSRAUM'] = df['VERDICHTUNGSRAUM'].apply(lambda it: np.nan if it == 0.0 else it)
        df['D19_LETZTER_KAUF_BRANCHE'] = df['D19_LETZTER_KAUF_BRANCHE'].apply(
            lambda it: np.nan if it == 'D19_UNBEKANNT' else it)
        return df

    @staticmethod
    def _ensure_consistent_spacing(df: pd.DataFrame):
        df['KBA05_MODTEMP'] = df['KBA05_MODTEMP'].apply(lambda it: 5.0 if it == 6.0 else it)
        return df

    def _remove_unknown_values(self, df: pd.DataFrame):
        unknown_values = self._documentation.find_unknown_values()
        for column in df.columns:
            df[column] = df[column].apply(lambda it: FeatureEngineer._unknown_value(it, column, unknown_values))
        return df

    def _recode_cameo_intl(self, df):
        df['CAMEO_INTL_2015_WEALTH'] = df['CAMEO_INTL_2015'].apply(
            lambda it: self._handle_mapping_with_subkey(it, 'wealth', Mappings().map_cameo_intl))
        df['CAMEO_INTL_2015_FAMILY_AGE'] = df['CAMEO_INTL_2015'].apply(
            lambda it: self._handle_mapping_with_subkey(it, 'familiy', Mappings().map_cameo_intl))
        df = df.drop(columns=['CAMEO_INTL_2015'], axis=1)
        return df

    @staticmethod
    def _recode_o_w(df):
        df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map(Mappings().map_o_w, na_action='ignore')
        return df

    @staticmethod
    def _recode_konsumtyp(df):
        df['D19_KONSUMTYP'] = df['D19_KONSUMTYP'].replace(9.0, 7).replace(9, 7)
        return df

    @staticmethod
    def _unknown_value(value, column: str, unknown_values: dict):
        if pd.isnull(value):
            return np.nan
        if column in unknown_values and value in unknown_values[column]:
            return np.nan
        return value

    def _missing_percentage_columns(self, df: pd.DataFrame, threshold, columns_to_keep):
        cols = []
        for col in df.columns:
            number_of_missing = df[col].isnull().mean()
            if number_of_missing > threshold and col not in columns_to_keep:
                cols.append(col)
        return cols

    def _handle_mapping_with_subkey(self, key, subkey, the_map):
        if pd.isnull(key):
            return np.nan
        return the_map[int(key)][subkey]

    def _handle_mapping(self, key, the_map):
        if pd.isnull(key):
            return np.nan
        return the_map[int(key)]


class Mappings:
    """
    The Mappings class defines some dictionaries to perform the recoding of some features
    """
    map_o_w = {'O': 1, 'W': 2, 'O+W': 3}

    map_o_w_inverse = {0: 'unknown', 1: 'O', 2: 'W', 3: 'O+W'}

    map_cameo_intl_wealth = {
        'poorer': 1,
        'less_affluent': 2,
        'comfortable': 3,
        'prosperous': 4,
        'wealthy': 5,
    }

    map_cameo_intl_wealth_inverse = {
        # 0: 'unknown',
        1: 'poorer',
        2: 'less_affluent',
        3: 'comfortable',
        4: 'prosperous',
        5: 'wealthy',
    }

    map_cameo_intl_family_age = {
        'pre_family': 1,
        'young_family': 2,
        'family': 3,
        'old_family': 4,
        'retired': 5,
    }

    map_cameo_intl_family_age_inverse = {
        # 0: 'unknown',
        1: 'pre_family',
        2: 'young_family',
        3: 'family',
        4: 'old_family',
        5: 'retired',
    }

    map_cameo_intl = {
        # 0: {0, 0},
        11: {'wealth': map_cameo_intl_wealth['wealthy'], 'familiy': map_cameo_intl_family_age['pre_family']},
        12: {'wealth': map_cameo_intl_wealth['wealthy'], 'familiy': map_cameo_intl_family_age['young_family']},
        13: {'wealth': map_cameo_intl_wealth['wealthy'], 'familiy': map_cameo_intl_family_age['family']},
        14: {'wealth': map_cameo_intl_wealth['wealthy'], 'familiy': map_cameo_intl_family_age['old_family']},
        15: {'wealth': map_cameo_intl_wealth['wealthy'], 'familiy': map_cameo_intl_family_age['retired']},
        21: {'wealth': map_cameo_intl_wealth['prosperous'], 'familiy': map_cameo_intl_family_age['pre_family']},
        22: {'wealth': map_cameo_intl_wealth['prosperous'], 'familiy': map_cameo_intl_family_age['young_family']},
        23: {'wealth': map_cameo_intl_wealth['prosperous'], 'familiy': map_cameo_intl_family_age['family']},
        24: {'wealth': map_cameo_intl_wealth['prosperous'], 'familiy': map_cameo_intl_family_age['old_family']},
        25: {'wealth': map_cameo_intl_wealth['prosperous'], 'familiy': map_cameo_intl_family_age['retired']},
        31: {'wealth': map_cameo_intl_wealth['comfortable'], 'familiy': map_cameo_intl_family_age['pre_family']},
        32: {'wealth': map_cameo_intl_wealth['comfortable'], 'familiy': map_cameo_intl_family_age['young_family']},
        33: {'wealth': map_cameo_intl_wealth['comfortable'], 'familiy': map_cameo_intl_family_age['family']},
        34: {'wealth': map_cameo_intl_wealth['comfortable'], 'familiy': map_cameo_intl_family_age['old_family']},
        35: {'wealth': map_cameo_intl_wealth['comfortable'], 'familiy': map_cameo_intl_family_age['retired']},
        41: {'wealth': map_cameo_intl_wealth['less_affluent'], 'familiy': map_cameo_intl_family_age['pre_family']},
        42: {'wealth': map_cameo_intl_wealth['less_affluent'], 'familiy': map_cameo_intl_family_age['young_family']},
        43: {'wealth': map_cameo_intl_wealth['less_affluent'], 'familiy': map_cameo_intl_family_age['family']},
        44: {'wealth': map_cameo_intl_wealth['less_affluent'], 'familiy': map_cameo_intl_family_age['old_family']},
        45: {'wealth': map_cameo_intl_wealth['less_affluent'], 'familiy': map_cameo_intl_family_age['retired']},
        51: {'wealth': map_cameo_intl_wealth['poorer'], 'familiy': map_cameo_intl_family_age['pre_family']},
        52: {'wealth': map_cameo_intl_wealth['poorer'], 'familiy': map_cameo_intl_family_age['young_family']},
        53: {'wealth': map_cameo_intl_wealth['poorer'], 'familiy': map_cameo_intl_family_age['family']},
        54: {'wealth': map_cameo_intl_wealth['poorer'], 'familiy': map_cameo_intl_family_age['old_family']},
        55: {'wealth': map_cameo_intl_wealth['poorer'], 'familiy': map_cameo_intl_family_age['retired']},
    }

    map_family_inverse = {
        1: 'single',
        2: 'couple',
        3: 'single parent',
        4: 'family',
        5: 'multiperson household'
    }

    map_status_inverse = {
        1: 'low-income earners',
        2: 'average earners',
        3: 'independants',
        4: 'houseowners',
        5: 'top earners',
    }


class PreProcessor:
    _datafolder: str
    _feature_engineer: Optional[FeatureEngineer]
    _azdias: pd.DataFrame
    _customers: pd.DataFrame
    _mailout_train: pd.DataFrame
    _mailout_train_target: pd.Series
    _mailout_test: pd.DataFrame
    _missing_values_threshold: float

    _DATAFRAME_FILES = {
        'azdias': {
            'cleaned': '01_df_azdias_cleaned.h5',
            'outliers_handled': '02_df_azdias_outliers.h5',
            'missing_handled': '03_df_azdias_missing.h5',
            'imputed': '04_df_azdias_imputed.h5',
            'scaled': '05_df_azdias_scaled.h5',
        },
        'customers': {
            'cleaned': '01_df_customers_cleaned.h5',
            'outliers_handled': '02_df_customers_outliers.h5',
            'missing_handled': '03_df_customers_missing.h5',
            'imputed': '04_df_customers_imputed.h5',
            'scaled': '05_df_customers_scaled.h5',
        },
        'mailout_train': {
            'cleaned': '01_df_mailout_train_cleaned.h5',
            'outliers_handled': '02_df_mailout_train_outliers.h5',
            'missing_handled': '03_df_mailout_train_missing.h5',
            'imputed': '04_df_mailout_train_imputed.h5',
            'scaled': '05_df_mailout_train_scaled.h5',
        },
        'mailout_test': {
            'cleaned': '01_df_mailout_test_cleaned.h5',
            'outliers_handled': '02_df_mailout_test_outliers.h5',
            'missing_handled': '03_df_mailout_test_missing.h5',
            'imputed': '04_df_mailout_test_imputed.h5',
            'scaled': '05_df_mailout_test_scaled.h5',
        }
    }

    _MODELS = {
        'scalers': {
            'azdias': '05_scaler_azdias.joblib',
            'customers': '05_scaler_customers.joblib',
            'mailout_train': '05_scaler_mailout_train.joblib',
            'mailout_test': '05_scaler_mailout_test.joblib',
        },
        'imputers': {
            'default': '03_iterative_imputer.joblib',
            'geburtsjahr': 'geb_dat_imp_bay_mean_asc_03.joblib',
        }
    }

    _TARGET_FILE = 'target.h5'

    def __init__(self, feature_engineer=None,
                 azdias=None, customers=None,
                 mailout_train=None, mailout_test=None,
                 root_path='.', out_dir='tmp_dat_prod',
                 missing_values_threshold=0.32):
        self._reset()
        self._datafolder = f"{root_path}/{out_dir}/pre_processor"
        self._missing_values_threshold = missing_values_threshold
        self._feature_engineer = feature_engineer
        Path(self._datafolder).mkdir(parents=True, exist_ok=True)
        if azdias is not None:
            self._azdias = PreProcessor._prepare_df(azdias)
        if customers is not None:
            self._customers = PreProcessor._prepare_df(customers)
        if mailout_train is not None:
            self._mailout_train, self._mailout_train_target = self._prepare_mailout_train(
                PreProcessor._prepare_df(mailout_train)
            )
        if mailout_test is not None:
            self._mailout_test = PreProcessor._prepare_df(mailout_test)

    def process(self):
        """
        Perform all preprocessing steps on the given datasets. These steps are:
        - 1 cleaning
        - 2 handle outliers and features with low std_dev
        - 3 handle missing values and recode some features
        - 4 impute
        - 5 scale
        :return: None
        """
        self.step1_clean()
        self.step2_handle_outliers_and_low_std_dev()
        self.step3_handle_missing_and_recode_composite_features()
        imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train = self.step4_impute()
        self.step5_scale(imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train)

    def step1_clean(self):
        """
        Perform step 1 of preprocessing - clean the data on all DataFrames
        (azdias, customers, mailout_train, mailout_test):
        - drop unneeded columns,
        - replace invalid values by np.nan,
        - ensure consistent feature spacing,
        - remove unknown values (due to documentation)
        :return: None
        """
        print('Step 1: Clean')
        print('azdias, customers')
        self._check_preconditions()
        self._azdias = self._feature_engineer.clean(self._azdias)
        self._customers = self._feature_engineer.drop_unneeded_customer_columns(self._customers)
        self._customers = self._feature_engineer.clean(self._customers)
        if self._is_mailout_defined():
            print('mailout_train, mailout_test')
            self._mailout_train = self._feature_engineer.clean(self._mailout_train)
            self._mailout_test = self._feature_engineer.clean(self._mailout_test)
        self._validate_column_consistency()
        self._save_in_phase('cleaned', self._azdias, self._customers, self._mailout_test, self._mailout_train)

    def step2_handle_outliers_and_low_std_dev(self):
        """
        Perform step 2 of preprocessing - handle outliers and columns with low standard deviation on all DataFrames
        (azdias, customers, mailout_train, mailout_test).
        :return: None
        """
        print('Step 2: Handle Outliers and cols with low stddev')
        print('azdias, customers')
        self._check_preconditions()
        cols_to_check = [col for col in self._feature_engineer.numeric_columns if col != 'GEBURTSJAHR']
        outlier_cols = self._feature_engineer.outlier_columns(self._azdias, cols_to_check)
        self._azdias = FeatureEngineer.handle_outliers(self._azdias, outlier_cols).round(0)
        self._customers = FeatureEngineer.handle_outliers(self._customers, outlier_cols).round(0)
        if self._is_mailout_defined():
            print('mailout_train, mailout_test')
            self._mailout_train = FeatureEngineer.handle_outliers(self._mailout_train, outlier_cols).round(0)
            self._mailout_test = FeatureEngineer.handle_outliers(self._mailout_test, outlier_cols).round(0)
        # low stddev
        print('azdias, customers')
        low_stddev = FeatureEngineer.columns_with_stddev_below_threshold(self._azdias)
        self._azdias = self._azdias.drop(columns=low_stddev, axis=1)
        self._customers = self._customers.drop(columns=low_stddev, axis=1)
        if self._is_mailout_defined():
            print('mailout_train, mailout_test')
            self._mailout_train = self._mailout_train.drop(columns=low_stddev, axis=1)
            self._mailout_test = self._mailout_test.drop(columns=low_stddev, axis=1)
        self._validate_column_consistency()
        self._save_in_phase('outliers_handled', self._azdias, self._customers, self._mailout_test, self._mailout_train)

    def step3_handle_missing_and_recode_composite_features(self):
        """
        Perform step 3 of preprocessing - remove missing values and recode composite features on all DataFrames
        (azdias, customers, mailout_train, mailout_test).
        :return: None
        """
        print('Step 3: Remove missing values and recode composite features')
        cols_to_keep = ['GEBURTSJAHR', 'ALTERSKATEGORIE_FEIN']
        print('azdias, customers')
        self._check_preconditions()
        self._azdias = \
            self._feature_engineer.drop_missing_and_recode_special_features(self._azdias,
                                                                            threshold=self._missing_values_threshold,
                                                                            columns_to_keep=cols_to_keep)
        self._customers = \
            self._feature_engineer.drop_missing_and_recode_special_features(self._customers,
                                                                            threshold=self._missing_values_threshold,
                                                                            columns_to_keep=cols_to_keep)
        if self._is_mailout_defined():
            print('mailout_train, mailout_test')
            self._mailout_train = \
                self._feature_engineer.drop_missing_and_recode_special_features(self._mailout_train,
                                                                                threshold=self._missing_values_threshold,
                                                                                columns_to_keep=cols_to_keep)
            self._mailout_test = \
                self._feature_engineer.drop_missing_and_recode_special_features(self._mailout_test,
                                                                                threshold=self._missing_values_threshold,
                                                                                columns_to_keep=cols_to_keep)
        # drop left over columns in other dataframes explicitly
        self._customers = self._customers.drop(
            PreProcessor._cols_not_in_b(self._customers, self._azdias),
            axis=1, errors='ignore')
        if self._is_mailout_defined():
            self._mailout_train = self._mailout_train.drop(
                PreProcessor._cols_not_in_b(self._mailout_train, self._azdias),
                axis=1, errors='ignore')
            self._mailout_test = self._mailout_test.drop(
                PreProcessor._cols_not_in_b(self._mailout_test, self._azdias),
                axis=1, errors='ignore')
        self._validate_column_consistency()
        self._save_in_phase('missing_handled', self._azdias, self._customers, self._mailout_test, self._mailout_train)

    def step4_impute(self):
        """
        Perform step 4 of preprocessing - imputation of all DataFrames
        (azdias, customers, mailout_train, mailout_test).
        :return: None
        """
        print('Step 4: Impute')
        # imputation
        self._impute_geburtsjahr()

        self._handle_unneeded_columns_after_geburtsjahr_imputation()
        self._validate_column_consistency()
        print('impute: azdias, customers')
        imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train = self._impute()
        self._validate_column_consistency()
        self._save_in_phase('imputed', imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train)
        return imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train

    def step5_scale(self, imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train):
        """
        Perform step 5 of preprocessing - scaling of all DataFrames
        (azdias, customers, mailout_train, mailout_test).
        :return: None
        """
        print('Step 5: Scale')
        print('azdias, customers')
        self._check_preconditions()
        azdias_scaler = self._scaler(
            imputed_azdias,
            f"{self._datafolder}/{PreProcessor._MODELS['scalers']['azdias']}"
        )
        scaled_azdias = pd.DataFrame(azdias_scaler.transform(imputed_azdias.values),
                                     index=imputed_azdias.index,
                                     columns=imputed_azdias.columns)
        customers_scaler = self._scaler(
            imputed_customers,
            f"{self._datafolder}/{PreProcessor._MODELS['scalers']['customers']}"
        )
        scaled_customers = pd.DataFrame(customers_scaler.transform(imputed_customers.values),
                                        index=imputed_customers.index,
                                        columns=imputed_customers.columns)
        scaled_mailout_train = None
        scaled_mailout_test = None
        if self._is_mailout_defined():
            print('mailout_train, mailout_test')
            mailout_train_scaler = self._scaler(
                imputed_mailout_train,
                f"{self._datafolder}/{PreProcessor._MODELS['scalers']['mailout_train']}"
            )
            scaled_mailout_train = pd.DataFrame(mailout_train_scaler.transform(imputed_mailout_train.values),
                                                index=imputed_mailout_train.index,
                                                columns=imputed_mailout_train.columns)

            mailout_test_scaler = self._scaler(
                imputed_mailout_test,
                f"{self._datafolder}/{PreProcessor._MODELS['scalers']['mailout_test']}"
            )
            scaled_mailout_test = pd.DataFrame(mailout_test_scaler.transform(imputed_mailout_test.values),
                                               index=imputed_mailout_test.index,
                                               columns=imputed_mailout_test.columns)
        self._validate_column_consistency()
        self._save_in_phase('scaled', scaled_azdias, scaled_customers, scaled_mailout_test, scaled_mailout_train)

    def load_step_1_cleaned_dfs(self):
        """
        Load all Processed dataframes in Phase 'clean' and return them as tuple
        :return: tuple of DataFrames. It has the form (azdias, customers, mailout_train, mailout_test)
        """
        return self._load_in_phase('cleaned')

    def load_step_2_outliers_handled_dfs(self):
        """
        Load all Processed dataframes in Phase 'outliers_handled' and return them as tuple
        :return: tuple of DataFrames. It has the form (azdias, customers, mailout_train, mailout_test)
        """
        return self._load_in_phase('outliers_handled')

    def load_step_3_missing_handled_dfs(self):
        """
        Load all Processed dataframes in Phase 'missing_handled' and return them as tuple
        :return: tuple of DataFrames. It has the form (azdias, customers, mailout_train, mailout_test)
        """
        return self._load_in_phase('missing_handled')

    def load_step_4_imputed_dfs(self):
        """
        Load all Processed dataframes in Phase 'missing_handled' and return them as tuple
        :return: tuple of DataFrames. It has the form (azdias, customers, mailout_train, mailout_test)
        """
        return self._load_in_phase('imputed')

    def load_step_5_scaled_dfs(self):
        """
        Load all Processed dataframes in Phase 'scaled' and return them as tuple
        :return: tuple of DataFrames. It has the form (azdias, customers, mailout_train, mailout_test)
        """
        return self._load_in_phase('scaled')

    def load_scalers(self):
        """Load the stored scalers from the filesystem"""
        azdias = load_model(f"{self._datafolder}/{PreProcessor._MODELS['scalers']['azdias']}")
        customers = load_model(f"{self._datafolder}/{PreProcessor._MODELS['scalers']['customers']}")
        mailout_train = None
        mailout_test = None
        if Path(f"{self._datafolder}/{PreProcessor._MODELS['scalers']['mailout_train']}").is_file():
            mailout_train = load_model(f"{self._datafolder}/{PreProcessor._MODELS['scalers']['mailout_train']}")
        if Path(f"{self._datafolder}/{PreProcessor._MODELS['scalers']['mailout_test']}").is_file():
            mailout_test = load_model(f"{self._datafolder}/{PreProcessor._MODELS['scalers']['mailout_test']}")
        return azdias, customers, mailout_train, mailout_test

    def load_target_variable(self):
        """Load the target variable for training from the filesystem"""
        filename = f"{self._datafolder}/{self._TARGET_FILE}"
        return pd.Series(pd.read_hdf(filename, key='df'))

    def _load_in_phase(self, phase):
        azdias = load_dataframe(self._df_filename('azdias', phase))
        customers = load_dataframe(self._df_filename('customers', phase))
        mailout_train = None
        mailout_test = None
        if Path(self._df_filename('mailout_train', phase)).is_file():
            mailout_train = load_dataframe(self._df_filename('mailout_train', phase))
        if Path(self._df_filename('mailout_test', phase)).is_file():
            mailout_test = load_dataframe(self._df_filename('mailout_test', phase))
        return azdias, customers, mailout_train, mailout_test

    def _save_in_phase(self, phase, azdias, customers, mailout_test=None, mailout_train=None):
        save_dataframe(azdias, self._df_filename('azdias', phase))
        save_dataframe(customers, self._df_filename('customers', phase))
        if self._is_mailout_defined():
            save_dataframe(mailout_train, self._df_filename('mailout_train', phase))
            save_dataframe(mailout_test, self._df_filename('mailout_test', phase))

    def _impute_geburtsjahr(self):
        print('_impute_geburtsjahr: azdias, customers')
        self._check_preconditions()
        filename = f"{self._datafolder}/{PreProcessor._MODELS['imputers']['geburtsjahr']}"
        self._azdias['GEBURTSJAHR'] = FeatureEngineer.impute_geburtsjahr(self._azdias, filename)['GEBURTSJAHR']
        self._customers['GEBURTSJAHR'] = FeatureEngineer.impute_geburtsjahr(self._customers, filename)['GEBURTSJAHR']
        if self._is_mailout_defined():
            print('_impute_geburtsjahr: mailout_train, mailout_test')
            self._mailout_train['GEBURTSJAHR'] = FeatureEngineer.impute_geburtsjahr(self._mailout_train, filename)[
                'GEBURTSJAHR']
            self._mailout_test['GEBURTSJAHR'] = FeatureEngineer.impute_geburtsjahr(self._mailout_test, filename)[
                'GEBURTSJAHR']

    def _handle_unneeded_columns_after_geburtsjahr_imputation(self):
        print('drop_unneeded_after_birthyear_reconstruction: azdias, customers')
        self._azdias = FeatureEngineer.drop_unneeded_after_birthyear_reconstruction(self._azdias)
        self._customers = FeatureEngineer.drop_unneeded_after_birthyear_reconstruction(self._customers)
        if self._is_mailout_defined():
            print('drop_unneeded_after_birthyear_reconstruction: mailout_train, mailout_test')
            self._mailout_train = FeatureEngineer \
                .drop_unneeded_after_birthyear_reconstruction(self._mailout_train)
            self._mailout_test = FeatureEngineer \
                .drop_unneeded_after_birthyear_reconstruction(self._mailout_test)
        # drop left over columns in other dataframes explicitly
        self._customers = self._customers.drop(
            PreProcessor._cols_not_in_b(self._customers, self._azdias),
            axis=1, errors='ignore')
        if self._is_mailout_defined():
            self._mailout_train = self._mailout_train.drop(
                PreProcessor._cols_not_in_b(self._mailout_train, self._azdias),
                axis=1, errors='ignore')
            self._mailout_test = self._mailout_test.drop(
                PreProcessor._cols_not_in_b(self._mailout_test, self._azdias),
                axis=1, errors='ignore')

    def _impute(self):
        filename = f"{self._datafolder}/{PreProcessor._MODELS['imputers']['default']}"
        imputer = FeatureEngineer.default_imputer(self._azdias, filename)
        # step4_impute
        print('impute: azdias, customers')
        imputed_azdias = pd.DataFrame(
            imputer.transform(self._azdias),
            index=self._azdias.index,
            columns=self._azdias.columns
        ).round(0)
        imputed_customers = pd.DataFrame(
            imputer.transform(self._customers),
            index=self._customers.index,
            columns=self._customers.columns
        ).round(0)
        if self._is_mailout_defined():
            print('impute: mailout_train, mailout_test')
            imputed_mailout_train = pd.DataFrame(
                imputer.transform(self._mailout_train),
                index=self._mailout_train.index,
                columns=self._mailout_train.columns
            ).round(0)
            imputed_mailout_test = pd.DataFrame(
                imputer.transform(self._mailout_test),
                index=self._mailout_test.index,
                columns=self._mailout_test.columns
            ).round(0)
            return imputed_azdias, imputed_customers, imputed_mailout_test, imputed_mailout_train
        return imputed_azdias, imputed_customers, None, None

    def _df_filename(self, df_name, phase):
        if df_name not in PreProcessor._DATAFRAME_FILES:
            raise AttributeError(f"Unknown key DATAFRAME_FILES: '{df_name}'")
        if phase not in PreProcessor._DATAFRAME_FILES[df_name]:
            raise AttributeError(f"Unknown key DATAFRAME_FILES[{df_name}]: '{phase}'")
        return '{0}/{1}'.format(
            self._datafolder,
            PreProcessor._DATAFRAME_FILES[df_name][phase]
        )

    def _is_mailout_defined(self):
        return self._mailout_train is not None and self._mailout_test is not None

    def _scaler(self, df, filename, recreate=False):
        try:
            if recreate:
                Path(filename).unlink(missing_ok=True)
            scaler = load_model(filename)
        except:
            scaler = StandardScaler().fit(df.values)
            save_model(scaler, filename)
        return scaler

    @staticmethod
    def _cols_not_in_b(a, b):
        return list(set(a.columns) - set(b.columns))

    def _check_preconditions(self):
        if self._azdias is None:
            raise ValueError('Violation: azdias must not be None')

    def _validate_column_consistency(self):
        a = set(self._azdias.columns)
        b = set(self._customers.columns)
        c = set(self._mailout_train.columns)
        d = set(self._mailout_test.columns)
        if self._is_mailout_defined():
            if a != b or a != c or a != d:
                msg = "Number of columns must be equal " \
                      "(azdias, customers, mailout_train, mailout_test): " \
                      "({0},{1},{2},{3}). " \
                      "Diffenence of sets (customers-azdias, mailout_train-azdias, mailout_test-azdias) " \
                      "({4},{5},{6})".format(
                    len(a),
                    len(b),
                    len(c),
                    len(d),
                    b - a,
                    c - a,
                    d - a
                )
                raise ValueError(msg)
        else:
            if a != b:
                msg = "Number of columns must be equal " \
                      "(azdias, customers): ({0},{1})" \
                      "Diffenence of sets (customers-azdias) ({2})".format(
                    len(a),
                    len(b),
                    b - a
                )
                raise ValueError(msg)

    def _reset(self):
        self._root_path = ''
        self._out_dir = ''
        self._feature_engineer = None
        self._azdias = pd.DataFrame()
        self._customers = pd.DataFrame()
        self._mailout_train = pd.DataFrame()
        self._mailout_train_target = pd.Series()
        self._mailout_test = pd.DataFrame()
        self._missing_values_threshold = 0

    def _prepare_mailout_train(self, mailout_train):
        mailout_train_ = mailout_train.drop(columns=['RESPONSE'], axis=1, errors='ignore')
        if 'RESPONSE' in mailout_train.columns:
            filename = f"{self._datafolder}/{self._TARGET_FILE}"
            mailout_train['RESPONSE'].to_hdf(filename, key='df', mode='w', index=True)
            return mailout_train_, mailout_train['RESPONSE']
        return mailout_train_, pd.Series()

    @staticmethod
    def _prepare_df(df):
        if 'LNR' in df.columns:
            df = df.set_index('LNR')
        return df

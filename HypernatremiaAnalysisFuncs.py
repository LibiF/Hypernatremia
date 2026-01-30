## All analysis functions

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as stats
import lifelines
import math
import statsmodels.api as sm
from lifelines.statistics import *
from scipy.stats import fisher_exact
import seaborn as sns

COMORB_COL_IDXS = np.arange(83,99)
DIURETICS_COL_IDXS = np.arange(103,107)
LAB_RESULT_COL_IDXS = np.arange(52,75,2)


def clear_outliers(main_df, corr_rate_cols, sodium_cols,
                   corr_hour_cols, max_sodium_col, outlier_thresh):
    
    '''This function clears the outliers related to sodium correction rates
     based on the supplied outlier threshold (shold be in "0.05" format)'''
    
    # Define outliers based on outlier_threshold
    all_corr_values = np.unique(np.reshape(corr_rate_cols.values, -1))
    all_corr_values = all_corr_values[~np.isnan(all_corr_values)]
    q_low, q_high = np.quantile(all_corr_values,[outlier_thresh, 1-outlier_thresh])
    
    
    # Get indices of patients with outliers
    extreme_low = corr_rate_cols.replace(np.nan,0)>=q_low
    extreme_low = extreme_low.sum(axis=1)
    extreme_low_idx = extreme_low.index[extreme_low < len(corr_rate_cols.columns)].tolist()
    
    extreme_high = corr_rate_cols.replace(np.nan,0)<=q_high
    extreme_high = extreme_high.sum(axis=1)
    extreme_high_idx = extreme_high.index[extreme_high < len(corr_rate_cols.columns)].tolist()
    
    outlier_indices = set(extreme_low_idx + extreme_high_idx)
    print('\n The number of rate outliers is:', len(outlier_indices))
    
    # Remove outliers based on index from all supplied Dfs
    main_df.drop(index = outlier_indices, inplace=True)
    sodium_cols.drop(index = outlier_indices, inplace=True)
    corr_rate_cols.drop(index = outlier_indices, inplace=True)
    corr_hour_cols.drop(index = outlier_indices, inplace=True)
    max_sodium_col.drop(index = outlier_indices, inplace=True)
    
    return main_df, corr_rate_cols, sodium_cols, corr_hour_cols, max_sodium_col

def calculate_daily_sod_corr(sod_cols, corr_hours):
    
    '''This function calculated the sodium correction of users within
    the first 24 hours in 2 scenarios:
    1. Measurements from the first 24 hours following hypernatremia detection
        exist - in this case we take the latest measurement taken in this time range
        and define the correction rate within the first 24 hours as 
        (Latest sodium) - (Initial sodium)
    2. No measurements were taken in the first 24 hours following hypernatremia detection - 
        in this case we take the earliest measurment following hypernatremia detection
        and define the correction rate within the first 24 hours as 
        (2nd sodium measurement)-(Initial sodium)/(2nd measurement time)*24
        
    The output of the function is a dataframe with 3 columns: Sodium_in_24_hours (the actual sodium
    level), Measurement_time, Corr_rate_in_24_hours (correction rate calculated using the
    methodology described above)'''
    
    THRESH = 24
    
    sod_in_24 = []
    corr_rate_in_24 = []
    measurement_time_array =[]
    user_ind_array = []
    
    # for i in np.arange(20):
    for i in np.arange(len(sod_cols)):

        user_corr_hours = corr_hours.iloc[i,:]
        user_ind = corr_hours.index.values[i]
        user_sod_vals = sod_cols.loc[user_ind]
        sod_base = user_sod_vals.iloc[0]
        
        if sum(user_corr_hours<= THRESH) > 1:
            # print('yes')
            user_corr_time_ind = np.where(user_corr_hours<=THRESH)[0][-1]
            measurement_time = user_corr_hours.iloc[user_corr_time_ind]
            user_sod_in_24 = user_sod_vals.iloc[user_corr_time_ind]
            user_corr_rate_in_24 = user_sod_in_24 - sod_base
            
            
        elif sum(user_corr_hours<= THRESH) == 1:
            # print('no')
            user_corr_time_ind = 1
            measurement_time =  user_corr_hours.iloc[user_corr_time_ind]
            user_sod_in_24 = user_sod_vals.iloc[1]
            user_corr_rate_in_24 = ((user_sod_in_24 - sod_base)/measurement_time)*24
            
        # print(sod_base, measurement_time, user_sod_in_24, user_corr_rate_in_24)    
            
        user_ind_array.append(user_ind)
        sod_in_24.append(user_sod_in_24)
        corr_rate_in_24.append(user_corr_rate_in_24)
        measurement_time_array.append(measurement_time)
        
    
    sod_24_hours_df = pd.DataFrame({'Sodium_in_24_hours': sod_in_24,
                                    'Measurement_time': measurement_time_array,
                                    'Corr_rate_in_24_hours':corr_rate_in_24},
                                     index=user_ind_array)
    
    return sod_24_hours_df

def define_correction_columns(main_df, sodium_columns_df, corr_rate_columns_df, corr_hour_columns):
    
    '''This function calculates all metrics regarding sodium correction.
    Fixed parameters:
    a. SOD_THRESH - Threshold for resolution of hypernatremia is defined as <=145
    b. CORR_RATE_THRESH - 0.5 mmol/L/h. Note that since correction rates represent 
        a decrease in sodium levels the true value is "-0.5"
    
    The function calculates the following stats:
    1. First_under145 - The first sodium value which shows resolution of hypernatremia.
        If hypernatremia was not resolved we take the last documented sodium value!!!
    2. First_time_under145 - The measurement time corresponding to the value shown in (1)
    3. First_under145_last_correction_rate - correction rate from First_under145 and its 
        preceding measurement
    4. First_under145_overall_corr_rate - correction rate calculated across the entire time it
        took to reach "First_under145"
    5. '''
    
    # Define fixed parameters
    CORR_RATE_THRESH = -0.5
    SOD_THRESH = 145
    FIRST_SOD_COL = 'sodium 1-Result numeric'
    
    # Find indices of first corrected sodium values and last sodium measurement documented
    first_corrected_ind = sodium_columns_df.apply(lambda x: np.where(x<=SOD_THRESH)[0],axis=1)
    last_non_nan_ind = sodium_columns_df.apply(lambda x: np.where(~x.isna())[0],axis=1)
    # choose first corrected ind if exists or last Sodium value possible
    sod_val_ind = [first_corrected_ind.iloc[row][0] if len(first_corrected_ind.iloc[row])>0 else\
                   last_non_nan_ind.iloc[row][-1]for row in range(len(first_corrected_ind))]
    
    # Set arrays or first corrected, correction rate,
    first_corrected = sodium_columns_df.values[np.arange(len(sod_val_ind)),list(sod_val_ind)]
    first_corrected_rate = corr_rate_columns_df.values[np.arange(len(sod_val_ind)),
                                                 np.array(sod_val_ind).T-1]
    first_corrected_time = corr_hour_columns.values[np.arange(len(sod_val_ind)),list(sod_val_ind)]
    
    # Append calculated columns
    main_df['First_under145'] = first_corrected
    main_df['First_time_under145'] = first_corrected_time
    main_df['First_under145_last_correction_rate'] = first_corrected_rate
    main_df['First_under145_overall_corr_rate'] = (main_df['First_under145'] - main_df[FIRST_SOD_COL])/ main_df['First_time_under145']
    main_df['Fastest_correction_rate'] = corr_rate_columns_df.apply(lambda x: min(x),axis=1)
    main_df['Sod_diff_from_onset'] = main_df['First_under145'] - main_df[FIRST_SOD_COL]
    
    main_df['Reached_normal'] = main_df['First_under145']<= SOD_THRESH 
    main_df['Is_slow_corr_overall'] = main_df['First_under145_overall_corr_rate']>= CORR_RATE_THRESH
    main_df['Is_slow_maxcorr'] = main_df['Fastest_correction_rate']>= CORR_RATE_THRESH

    return main_df


def create_BUN_creatinine_columns(main_df):
    BUN_col = 'BUN-Result numeric'
    BUN_time_col = 'BUN-Collection Date-Hours from Reference'
    creat_col = 'createnine-Result numeric'
    creat_time_col = 'createnine-Collection Date-Hours from Reference'
    
    main_df['BUN_to_creatinine'] = main_df[BUN_col]/main_df[creat_col]
    main_df['BUN_creatinine_TD'] = main_df[BUN_time_col] - main_df[creat_time_col]
    
    return main_df

def check_glucose_outliers(main_df, GLUC_THRESH=500):
    

    first_gluc_col = 'glucose-result numeric_1'
    main_df['Initial_gluc>thresh'] = main_df[first_gluc_col]>GLUC_THRESH
    print('Num exceeding gluc threshold:', main_df['Initial_gluc>thresh'].sum())

    return main_df

def check_if_from_ICU(main_df):
    
    dept_col = 'reference event-ordering department'
    main_df['Is_ICU'] = main_df.apply(lambda row: 'ט.נ' in str(row[dept_col]) or 'נמרץ' in str(row[dept_col]) \
                                      and 'פנימ' not in str(row[dept_col]), axis=1)
    return main_df


def daily_correction_above_threshs(df, thresholds):
    
    # hyp_on_admission_col = 'Hypernatremia admission 1 or hospitalization 2'
    daily_corr_rate_col = 'Corr_rate_in_24_hours'
    correct_above_daily_thresh = 'Corr_rate_perday_above_thresh'
    # HOURS_IN_DAY = 24
  
    for i in range(len(thresholds)):
        thresh = thresholds[i]
        correct_above_daily_thresh_num = correct_above_daily_thresh + '=' + str(abs(thresh))
        df[correct_above_daily_thresh_num] =  df[daily_corr_rate_col]<thresh
        
    return df

def plot_mortality_vs_daily_corr_bar(df, plot_color, max_range):
    
    
    daily_corr_rate_col = 'Corr_rate_in_24_hours'
    mortality_30days_col = '30 day mortality 1 - Yes 0 - No'
    age_col = 'Reference Event-Age at lab test'
    gender_col = 'Gender 1 = M 2 =F'
    charlson_score_col = 'charlson-Charlson Score'
    creatinine_col = 'createnine-Result numeric'
    
    
    daily_corr_rate_range = np.arange(0,-max_range,-1)
 
    in_range_num = []
    in_range_mort_ratio = []   
    in_range_deaths = []
 
    for i in range(len(daily_corr_rate_range)-1):
        
        bin_min = daily_corr_rate_range[i]
        bin_max = daily_corr_rate_range[i+1]
        
        patients_in_range = df[(df[daily_corr_rate_col] <= bin_min) & \
                               (df[daily_corr_rate_col] > bin_max)]
        in_range_mort_ratio.append(patients_in_range[mortality_30days_col].sum()/len(patients_in_range)*100)
        in_range_num.append(len(patients_in_range))
        in_range_deaths.append(patients_in_range[mortality_30days_col].sum())        
    
    # Append last correction rate bin (>14)
    bin_min = bin_max
    patients_in_range = df[(df[daily_corr_rate_col] <= bin_min)]
    in_range_mort_ratio.append(patients_in_range[mortality_30days_col].sum()/len(patients_in_range)*100)
    in_range_num.append(len(patients_in_range))
    in_range_deaths.append(patients_in_range[mortality_30days_col].sum())      
    
    x_label = [str(x) for x in abs(daily_corr_rate_range[:-1])]    
    x_label.append('>'+str(abs(bin_min)))
    print(x_label)
    mort_rate = pd.DataFrame({'Mort': in_range_mort_ratio}, index=x_label)    
    # mort_sum = pd.DataFrame({'Mort': in_range_deaths}, index=x_label)    
    
    fig, ax = plt.subplots()
    # Plot the first x and y axes:
        
        
    # mort_rate.plot(use_index=True, kind='line',y='Mort', ax=ax,
    #                color = plot_color, legend=False)
    # # plt.bar()
    # # df.plot(kind='hist', x=daily_corr_rate_col, y=mortality_30days_col, ax=ax,
    # #         color=plot_color)
    # xtickslocs = ax.get_xticks()
    # # ax.set_xticks(ticks=xtickslocs-0.5, labels = x_label)
    # # ax.set_xticks( labels = x_label)

    
    # # ax.set_xticklabels(abs(daily_corr_rate_range))
    # ax.set_xlabel('Daily Sodium Correction [mmol/L/d]')
    # ax.set_ylabel('30 day mortality rate') 
    # plt.show()
    # fig.savefig('Mortality_vs_DailyCorr_080523.svg',format='svg' ,dpi=300)  
    
    mort_df = pd.DataFrame({'Num':in_range_num, 'Mortality_Count': in_range_deaths,
                            'Ratio': in_range_mort_ratio},index=x_label)
    
    # short_mort_df = mort_df.iloc[:14,:].copy()    
    
    # # ind_var = range(len(mort_df))
    # # ind_var = ["{:01d}".format(x) for x in ind_var]
    # # print(ind_var)
    mort_df['ind'] = range(max_range)
    
    # # over14_df = mort_df.iloc[14:,:].sum()
    # # over14_df['ind'] = '>14'
    # # over14_df['Ratio'] = round((over14_df['Mortality_Count']/over14_df['Num'])*100,2)
    # # short_mort_df = short_mort_df.append( over14_df, ignore_index=True)

    mort_correctrate_corr = mort_df['Ratio'].corr(mort_df['ind'])
    # print(mort_correctrate_corr)
    res = stats.pearsonr(mort_df['Ratio'].values, mort_df['ind'].values)
    # print(r, p)
    print(res)
    # res.confidence_interval(confidence_level=0.95)
    # mort_correctrate_corr = np.corrcoef(daily_corr_rate_range[:-1], in_range_mort_ratio[:-1])
    print( "Correlation between correct rate and mort_ratio is: ", np.round( mort_correctrate_corr, 4))
    
    ax = sns.regplot(x=mort_df['ind'], y=mort_df['Ratio'],marker='o', color = plot_color)
    ax.set_ylim(20,60)
    ax.set_xlabel('First 24 hour Sodium Correction [mmol/L/d]')
    ax.set_ylabel('30 day mortality rate [%]')
    # ax.set_indexylim([20,60])
    # fig.savefig('Mortality_vs_DailyCorr_170523.svg',format='svg' ,dpi=300)  

    print(mort_df['Ratio'], mort_df['ind'])
    
    return mort_df


def plot_mortality_vs_cont_daily_corr(df, color_above, color_below):
    
    daily_corr_rate_col = 'Corr_rate_in_24_hours'
    mortality_30days_col = '30 day mortality 1 - Yes 0 - No'
    age_col = 'Reference Event-Age at lab test'
    gender_col = 'Gender 1 = M 2 =F'
    charlson_score_col = 'charlson-Charlson Score'
    creatinine_col = 'createnine-Result numeric'
    
    daily_corr_rate_range = np.arange(-1,-21,-1)
    
    above_mort_rate = []
    below_mort_rate = []
    
    above_num = []
    below_num = []

    for i in range(len(daily_corr_rate_range)):

        thresh = daily_corr_rate_range[i]
        daily_above = df[df[daily_corr_rate_col] < thresh]
        daily_below = df[df[daily_corr_rate_col] >= thresh]
        # daily_above = df[cont_corr_rate_df.iloc[:,i]==True]
        # daily_below = df[cont_corr_rate_df.iloc[:,i]==False]
        
        above_mort_rate.append(daily_above[mortality_30days_col].sum()/len(daily_above)*100)
        above_num.append(len(daily_above))
        below_mort_rate.append(daily_below[mortality_30days_col].sum()/len(daily_below)*100)
        below_num.append(len(daily_below))
        
        group_a_mor = daily_above[mortality_30days_col].sum()
        group_b_mor = daily_below[mortality_30days_col].sum()
        
        group_a_alive = len(daily_above)-group_a_mor
        group_b_alive = len(daily_below)-group_b_mor
        
        print(daily_corr_rate_range[i])
        print('Group Above size: ', len(daily_above))
        print('Group Below size: ', len(daily_below))
        
        print('Mortality Group Above: ', group_a_mor, round(group_a_mor/(group_a_mor+group_a_alive)*100,2))
        print('Mortality Group Below: ', group_b_mor, round(group_b_mor/(group_b_mor+group_b_alive)*100,2))
        
        cont_table = np.array([[group_b_mor, group_b_alive],[group_a_mor, group_a_alive]])
        oddsr, p = sp.stats.fisher_exact(cont_table, alternative='two-sided')
        print(oddsr, p)
        
        CI_up = math.exp(np.log(oddsr)+1.96*math.sqrt(1/group_a_mor+1/group_b_mor+
                                                       1/group_a_alive+1/group_b_alive)) 
        CI_down = math.exp(np.log(oddsr)-1.96*math.sqrt(1/group_a_mor+1/group_b_mor+
                                                       1/group_a_alive+1/group_b_alive)) 
        
        print( 'Odds ratio: ', round(oddsr,2), '[', round(CI_down,2), '-', round(CI_up,2), ']')
        
        mortality_vec_a = daily_above[mortality_30days_col].values
        mortality_vec_b = daily_below[mortality_30days_col].values
        
        age_vec_a = daily_above[age_col].values
        age_vec_b = daily_below[age_col].values
        male_vec_a = daily_above[gender_col].values
        male_vec_a = np.where(male_vec_a != 1, 0, male_vec_a)
        male_vec_b = daily_below[gender_col].values
        male_vec_b = np.where(male_vec_b != 1, 0, male_vec_b)
        charls_vec_a = daily_above[charlson_score_col].values
        charls_vec_b = daily_below[charlson_score_col].values
        
        #extended correction - new
        creatinine_vec_a = daily_above[creatinine_col].values
        creatinine_vec_b = daily_below[creatinine_col].values
        
        
        mortality_vec_total = np.concatenate([mortality_vec_a, mortality_vec_b])
        diff_rate_vec = np.concatenate([np.zeros(len(daily_above)), np.ones(len(daily_below))])
        total_age_vec = np.concatenate([age_vec_a,age_vec_b])
        total_gender_vec = np.concatenate([male_vec_a, male_vec_b])
        total_charls_vec = np.concatenate([charls_vec_a, charls_vec_b])
        total_creatinine_vec = np.concatenate([creatinine_vec_a, creatinine_vec_b])
        
        
        all_vars_extended_array = np.c_[diff_rate_vec, total_age_vec, total_gender_vec,
                                        total_charls_vec, total_creatinine_vec]
        mortality_vec_extended_adj = mortality_vec_total[~np.isnan( \
                                            all_vars_extended_array).any(axis=1)]
        all_vars_extended_array = all_vars_extended_array[~np.isnan(all_vars_extended_array).any(axis=1), :]    
        all_vars_extended_array = sm.add_constant(all_vars_extended_array)
        
        extended_adj_model = sm.Logit(mortality_vec_extended_adj, all_vars_extended_array)
        extended_adj_result = extended_adj_model.fit(method='newton')
        
        odd_r = np.round(np.exp(extended_adj_result.params[1]),2)
        adj_CI = np.round(np.exp(extended_adj_result.conf_int()[1]),2)
        print('Adjusted reg:OR=', odd_r, adj_CI)

        # print(extended_adj_result.summary())
        # input("Press Enter to continue...")
        
        
    x_label = [str(x) for x in abs(np.arange(-1,-21,-1))]    
    mort_rate = pd.DataFrame({'Over': above_mort_rate,
    'Under': below_mort_rate,'Above_Num': above_num,
    'Below_Num': below_num}, index=x_label)    
    
    # Create the figure and axes object
    fig, ax = plt.subplots()
    # Plot the first x and y axes:
    mort_rate.plot(use_index=True, kind='line',y=['Over','Under'], ax=ax,
                   color = [color_above, color_below])
    ax.set_xlabel('First 24 hour Sodium Correction [mmol/L/d]')
    ax.set_ylabel('30 day mortality rate')
    ax.set_xticks(abs(daily_corr_rate_range)-1, x_label, rotation=45)
    
    plt.show()
    fig.savefig('Mortality_vs_DailyCorr_MayUpd.svg',format='svg' ,dpi=300)  
    
    return mort_rate

def compare_mortality_across_daily_corrections(group_df, thresholds, group_name):

        correct_above_daily_thresh = 'Corr_rate_perday_above_thresh'
        mortality_30days_col = '30 day mortality 1 - Yes 0 - No'
        color_below = '#0071BB'
        color_above = '#E31A1C'
        sub_fig_title_format = '{0} mmol/L/day'
        fig1, axs1 = plt.subplots(1, 3)
        
        for i in range(len(thresholds)):
            thresh = thresholds[i]
            sub_fig_title = sub_fig_title_format.format(str(thresh))
            correct_above_daily_thresh_num = correct_above_daily_thresh + '=' + str(abs(thresh))
            
            
            df_daily_over = group_df[group_df[correct_above_daily_thresh_num]==True]
            df_daily_under = group_df[group_df[correct_above_daily_thresh_num]==False]
            data = {'Above':(1-df_daily_over[mortality_30days_col].sum()/len(df_daily_over))*100,
                  'Below':(1-df_daily_under[mortality_30days_col].sum()/len(df_daily_under))*100}
        
            tags = list(data.keys())
            values = list(data.values())
            print(tags) 
            print(values)
            axs1[i].bar(tags, values, color =[color_above, color_below])
            axs1[i].set_ylim((0,90))
            axs1[0].set_ylabel("30 day mortality [%]")
            axs1[i].set_xlabel(sub_fig_title)
            fig1.suptitle(group_name)
            print('#N Over: ',  len(df_daily_over), 'Mortality: ', df_daily_over[mortality_30days_col].sum(),
                  'Survivall: ', (df_daily_over[mortality_30days_col]==0).sum())
            print('#N Under: ',  len(df_daily_under), 'Mortality: ', df_daily_under[mortality_30days_col].sum(),
                  'Survivall: ', (df_daily_under[mortality_30days_col]==0).sum())

            oddsratio, pvalue = fisher_exact([[df_daily_over[mortality_30days_col].sum(), (len(df_daily_over)-df_daily_over[mortality_30days_col].sum())],
                                              [df_daily_under[mortality_30days_col].sum(), (len(df_daily_under)-df_daily_under[mortality_30days_col].sum())]])
            print('pval: ', pvalue)
            
        
        plt.show()
        fig_name = 'Mortality_vs_dailycorrect'+'_' + group_name + '.svg'
        fig1.savefig(fig_name,format='svg' ,dpi=300)    
        return fig1

def make_categorical_cont_table(col_a, col_b):
    a = [col_a.sum(), len(col_a)-col_a.sum()]
    b = [col_b.sum(), len(col_b)-col_b.sum()]
    table = [a, b]
    return table
    
def analyze_group_stats(group_df, master_group_df, lab_result_max_time_diff,
                        comorb_col_idxs = COMORB_COL_IDXS,
                        diuretics_col_idxs = DIURETICS_COL_IDXS,
                        lab_result_col_idxs = LAB_RESULT_COL_IDXS):
    
    hours_in_day = 24
    
    Age_col = 'Reference Event-Age at lab test'
    Sex_col = 'Gender 1 = M 2 =F'
    charlson_col = 'charlson-Charlson Score'
    mortality_7days_col = '7 day mortality 1 - Yes 0 - No'
    mortality_30days_col = '30 day mortality 1 - Yes 0 - No'
    mortality_1year_col = '1 year mortality 1 - Yes 0 - No'

    diabetes_col = 'charlson-Diabetes 1= End-Organ Damage 2 = Uncomplicated 3 = no or diet controled'
    liver_dis_col = 'charlson-Liver Disease 1= mild 2= moderate 0 = no'
    tumor_dis_col = 'charlson-Tumor 1= local 2 = metastatic 0 = no'
    length_of_stay_col = 'diagnosis at discharge-Diagnosis date-Days from Reference'
    adm_dep_col = 'charlson-First Admmiting Department'
    
    # comorb_col_idxs = np.arange(83,99)
    # diuretics_col_idxs = np.arange(103,107)
    # lab_result_col_idxs = np.arange(52,75,2)
    
    
    neurological_outcomes_col = 'diagnosis at discharge-Selected ICD9 Code'
    neurological_outcomes_codes = ['348.5','780','780.39','345.9']
    
    BUN_to_creat_col = 'BUN_to_creatinine'
    BUN_to_creat_TD_col = 'BUN_creatinine_TD' 
    
    glucose_over_thresh_col = 'Initial_gluc>thresh'
    Is_ICU_col = 'Is_ICU'
    
    daily_corr_rate_col_num = 'Corr_rate_in_24_hours'
    Daily_corr_rate_cols_thresh = 'Corr_rate_perday_above_thresh'
    Daily_corr_rate_cols_thresh = [col for col in group_df if col.startswith(Daily_corr_rate_cols_thresh)]
    
    group_n = len(group_df)
    master_group_n = len(master_group_df)
    print( 'Group size ', group_n, '(', round(group_n/master_group_n*100), ')')
    
    print( 'Group Age ',  round(group_df[Age_col].quantile(0.5)), '(', 
          round(group_df[Age_col].quantile(0.25)),
         '-', round(group_df[Age_col].quantile(0.75) ),')')
    
    
    print('Gender - Males ', sum(group_df[Sex_col]==1), '(',
          round(sum(group_df[Sex_col]==1)/group_n*100),')')
    print('Gender - Females ', sum(group_df[Sex_col]==2), '(',
          round(sum(group_df[Sex_col]==2)/group_n*100),')')
    
    charlson0 = sum(group_df[charlson_col] == 0)
    charlson1 = sum(group_df[charlson_col] == 1)
    charlson2 = sum(group_df[charlson_col] == 2)
    charlson3andup = sum(group_df[charlson_col] >= 3)
    total_charls = sum(~group_df[charlson_col].isna())
    
    print('Charlson 0 ', charlson0 , '(', round(charlson0/total_charls*100) ,')')
    print('Charlson 1 ', charlson1 , '(', round(charlson1/total_charls*100) ,')')
    print('Charlson 2 ', charlson2 , '(', round(charlson2/total_charls*100) ,')')
    print('Charlson 3<= ', charlson3andup , '(', round(charlson3andup/total_charls*100) ,')')
    print('Total_charls_scores ', total_charls)
    print('Mean charlson score', group_df[charlson_col].mean())
    print('Median charlson score', group_df[charlson_col].median(),
          '(', round(group_df[charlson_col].quantile(0.25),2), '-',
          round(group_df[charlson_col].quantile(0.75),2), ')')

    
    print('7 day mortality ', group_df[mortality_7days_col].sum() ,
          '(', round(group_df[mortality_7days_col].sum()/group_n*100),')')
    
    print('30 day mortality ', group_df[mortality_30days_col].sum() ,
          '(', round(group_df[mortality_30days_col].sum()/group_n*100),')')
    
    print('1 year mortality ', group_df[mortality_1year_col].sum() ,
          '(', round(group_df[mortality_1year_col].sum()/group_n*100),')')
    
    print('Length of stay (days) ', round(group_df[length_of_stay_col].quantile(0.5),2),
          '(', round(group_df[length_of_stay_col].quantile(0.25),2), '-',
          round(group_df[length_of_stay_col].quantile(0.75),2), ')')
    
    for com_col_ind in comorb_col_idxs:
        col_name = group_df.columns[com_col_ind]
      #   print(col_name)
        n_pats_with_comorb = group_df[col_name].sum()
        precent_with_comorb = round(n_pats_with_comorb/group_n*100,2)
        print(col_name , n_pats_with_comorb ,'(', precent_with_comorb,')')
        
    print('Diabetes "End organ damage" ', sum(group_df[diabetes_col]==1), 
          '(', round(sum(group_df[diabetes_col]==1)/group_n*100,2),')')
    print('Diabetes "Uncomplicated" ', sum(group_df[diabetes_col]==2), 
          '(', round(sum(group_df[diabetes_col]==2)/group_n*100,2),')')
    print('Diabetes "No/diet controled" ', sum(group_df[diabetes_col]==3), 
          '(', round(sum(group_df[diabetes_col]==3)/group_n*100,2),')')
    
    print('Liver disease "Mild" ', sum(group_df[liver_dis_col]==1), 
          '(', round(sum(group_df[liver_dis_col]==1)/group_n*100,2),')')
    print('Liver disease "Moderate" ', sum(group_df[liver_dis_col]==2), 
          '(', round(sum(group_df[liver_dis_col]==2)/group_n*100,2),')')
    print('Liver disease "None" ', sum(group_df[liver_dis_col]==0), 
          '(', round(sum(group_df[liver_dis_col]==0)/group_n*100,2),')')
    
    print('Tumor "Local" ', sum(group_df[tumor_dis_col]==1), 
          '(', round(sum(group_df[tumor_dis_col]==1)/group_n*100,2),')')
    print('Tumor "Metastatic" ', sum(group_df[tumor_dis_col]==2), 
          '(', round(sum(group_df[tumor_dis_col]==2)/group_n*100,2),')')
    print('Tumor disease "None" ', sum(group_df[tumor_dis_col]==0), 
          '(', round(sum(group_df[tumor_dis_col]==0)/group_n*100,2),')')
    
    for diu_col_ind in diuretics_col_idxs:
        col_name = group_df.columns[diu_col_ind]
        n_pats_with_diu = group_df[col_name].sum()
        precent_with_diu = round(n_pats_with_diu/group_n*100,2)
        print(col_name , n_pats_with_diu ,'(', precent_with_diu,')')
        
    diu_cols = group_df.iloc[:,diuretics_col_idxs]
    diu_cols = diu_cols.sum(axis=1)
    print('Total patients on diuretics: ', (diu_cols>0).sum())
        
    for lab_res_col in lab_result_col_idxs:
        col_name = group_df.columns[lab_res_col]
        col_collect_time = group_df.columns[lab_res_col -1]
        lab_res_in_td = group_df[col_name][group_df[col_collect_time]<=lab_result_max_time_diff]
        n_res = len(lab_res_in_td)
        col_med = round(lab_res_in_td.quantile(0.5),2)
        col_Q1 = round(lab_res_in_td.quantile(0.25),2)
        col_Q3 = round(lab_res_in_td.quantile(0.75),2)
        print(col_name, col_med, '(',col_Q1,'-',col_Q3,'),', 'n=', n_res)
    
    neurological_outcome_sum = group_df[neurological_outcomes_col].astype(str).isin(neurological_outcomes_codes).sum()
    print('Neurological outcomes ', neurological_outcome_sum ,
          '(', round(neurological_outcome_sum/group_n*100,2),')')    
    
    valid_BUN_to_creat = group_df[abs(group_df[BUN_to_creat_TD_col])<hours_in_day][BUN_to_creat_col] 
    n_BUN_to_creat = len(valid_BUN_to_creat)
    BUN_to_creat_median = round(valid_BUN_to_creat.quantile(0.5),2)
    BUN_to_creat_Q1 = round(valid_BUN_to_creat.quantile(0.25),2)
    BUN_to_creat_Q3 = round(valid_BUN_to_creat.quantile(0.75),2)
    print('BUN/Creatinine', BUN_to_creat_median, '(',BUN_to_creat_Q1,'-',BUN_to_creat_Q3,'),', 'n=', n_BUN_to_creat)
    
    Is_ICU = sum(group_df[Is_ICU_col]==True)
    print('Number of patients in ICU', Is_ICU, 
      '(', round( Is_ICU/group_n*100,1),')')
    
    glucose_over_thresh = sum(group_df[glucose_over_thresh_col]==True)
    print('Initial Glucose exceeding thresh', glucose_over_thresh, 
          '(', round( glucose_over_thresh/group_n*100,1),')')
    
    daily_correct_median = round(group_df[daily_corr_rate_col_num].quantile(0.5),2)
    daily_correct_Q1 = round(group_df[daily_corr_rate_col_num].quantile(0.25),2)
    daily_correct_Q3 = round(group_df[daily_corr_rate_col_num].quantile(0.75),2)
    print('Dailly sodium correction rat', daily_correct_median, '(',daily_correct_Q1,'-',daily_correct_Q3,')')


    print('First 24-hour Correction Rate columns', '\n')
    for corr_rate_col in Daily_corr_rate_cols_thresh:
        print(corr_rate_col, 
        sum(group_df[corr_rate_col]==1), '(', round(sum(group_df[corr_rate_col]==1)/group_n*100,2),')')



def compare_general_stats(df_a, df_b, comorb_col_idxs = COMORB_COL_IDXS,
                          diuretics_col_idxs = DIURETICS_COL_IDXS,
                          lab_result_col_idxs = LAB_RESULT_COL_IDXS):
    
    hours_in_day = 24

    Age_col = 'Reference Event-Age at lab test'
    Sex_col = 'Gender 1 = M 2 =F'
    glucose_over_thresh_col = 'Initial_gluc>thresh'
    Is_ICU_col = 'Is_ICU'
    charlson_col = 'charlson-Charlson Score'
    mortality_7days_col = '7 day mortality 1 - Yes 0 - No'
    mortality_30days_col = '30 day mortality 1 - Yes 0 - No'
    mortality_1year_col = '1 year mortality 1 - Yes 0 - No'

    diabetes_col = 'charlson-Diabetes 1= End-Organ Damage 2 = Uncomplicated 3 = no or diet controled'
    liver_dis_col = 'charlson-Liver Disease 1= mild 2= moderate 0 = no'
    tumor_dis_col = 'charlson-Tumor 1= local 2 = metastatic 0 = no'
    length_of_stay_col = 'diagnosis at discharge-Diagnosis date-Days from Reference'
    adm_dep_col = 'charlson-First Admmiting Department'
    columns_to_compare = [Age_col, length_of_stay_col]
    mortality_cols = [mortality_7days_col, mortality_30days_col,
                          mortality_1year_col]
    
    # comorb_col_idxs = np.arange(83,99)
    # diuretics_col_idxs = np.arange(103,107)
    # lab_result_col_idxs = np.arange(52,75,2)
    comorb_col_idxs = np.delete(comorb_col_idxs , [5,10,15])

    neurological_outcomes_col = 'diagnosis at discharge-Selected ICD9 Code'
    neurological_outcomes_codes = ['348.5','780','780.39','345.9']
    
    BUN_to_creat_col = 'BUN_to_creatinine'
    BUN_to_creat_TD_col = 'BUN_creatinine_TD' 
    
    daily_corr_rate_col_num = 'Corr_rate_in_24_hours'
    Daily_corr_rate_cols = 'Corr_rate_perday_above_thresh'
    Daily_corr_rate_cols = [col for col in df_a if col.startswith(Daily_corr_rate_cols)]
    
    print('Gender - Males ')
    col_a = df_a[Sex_col]==1
    col_b = df_b[Sex_col]==1
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    # U1, p = sp.stats.mannwhitneyu(col_a, col_b)
    # print(U1, p)
    print( oddsratio, 'pval' ,pvalue)
    
    print('Gender - Females ')
    col_a = df_a[Sex_col]==2
    col_b = df_b [Sex_col]==2
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)

    print('Patients in ICU ')
    col_a = df_a[Is_ICU_col]==1
    col_b = df_b[Is_ICU_col]==1
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)

    print('Extreme hyperglycemic patients ')
    col_a = df_a[glucose_over_thresh_col]==1
    col_b = df_b[glucose_over_thresh_col]==1
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)

    print('\n', 'General', '\n')
    for col in columns_to_compare:
        col_a = df_a[col]
        col_b = df_b[col]
        U1, p = stats.mannwhitneyu(col_a.dropna(), col_b.dropna())
        print(col)
        print(U1, p)

    print('\n', 'Mortality columns', '\n')
    for col in mortality_cols:
        col_a = df_a[col]
        col_b = df_b[col]
        cont_tab = make_categorical_cont_table(col_a, col_b)
        oddsratio, pvalue = fisher_exact(cont_tab)
        print(col)
        print( oddsratio, 'pval' ,pvalue)        
    
    print('\n', 'Comorbidities', '\n')
    for com_col_ind in comorb_col_idxs:
        col_name = df_a.columns[com_col_ind]
        col_a = df_a[col_name]
        col_b = df_b[col_name]
        # U1, p = sp.stats.mannwhitneyu(col_a, col_b)
        cont_tab = make_categorical_cont_table(col_a, col_b)
        oddsratio, pvalue = fisher_exact(cont_tab)
        print( col_name, oddsratio, 'pval' ,pvalue)
        
    print('\n', 'General Charlson', '\n')
    col_a = df_a[charlson_col]
    col_b = df_b[charlson_col]
    U1, p = sp.stats.mannwhitneyu(col_a.dropna(), col_b.dropna())
    print('Charlson score' , U1, 'pval: ', p)
    print('\n', 'Charlson', '\n')
    col_a = df_a[charlson_col] == 0
    col_b = df_b[charlson_col] == 0
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print('Charlson=0' , oddsratio, 'pval: ', pvalue)
    col_a = df_a[charlson_col] == 1
    col_b = df_b[charlson_col] == 1
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print('Charlson=1' , oddsratio, 'pval: ', pvalue)
    col_a = df_a[charlson_col] == 2
    col_b = df_b[charlson_col] == 2
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print('Charlson=2' , oddsratio, 'pval: ', pvalue)
    col_a = df_a[charlson_col] >= 3
    col_b = df_b[charlson_col] >= 3
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print('Charlson>=3' , oddsratio, 'pval: ', pvalue)
    
        
    print('\n', 'Diuretics', '\n')    
    for diu_col_ind in diuretics_col_idxs:
        col_name = df_a.columns[diu_col_ind]
        col_a = df_a[col_name]
        col_b = df_b[col_name]
        cont_tab = make_categorical_cont_table(col_a, col_b)
        oddsratio, pvalue = fisher_exact(cont_tab)
        print(col_name, oddsratio, 'pval: ', pvalue)

    
    print('\n', 'Diabetes', '\n')
    print('Diabetes "End organ damage" ')
    col_a = df_a[diabetes_col]==1
    col_b = df_b[diabetes_col]==1
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('Diabetes "Uncomplicated" ')
    col_a = df_a[diabetes_col]==2
    col_b = df_b[diabetes_col]==2
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('Diabetes "No/diet controled" ')
    col_a = df_a[diabetes_col]==3
    col_b = df_b[diabetes_col]==3
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('\n', 'Liver disease' , '\n')
    print('Liver disease "Mild" ')
    col_a = df_a[liver_dis_col]==1 
    col_b = df_b[liver_dis_col]==1 
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('Liver disease "Moderate" ')
    col_a = df_a[liver_dis_col]==2
    col_b = df_b[liver_dis_col]==2 
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('\n','Tumor','\n')
    
    print('Tumor "Local" ')
    col_a = df_a[tumor_dis_col]==1 
    col_b = df_b[tumor_dis_col]==1 
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    print('Tumor "Metastatic" ')
    col_a = df_a[tumor_dis_col]==2 
    col_b = df_b[tumor_dis_col]==2 
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('\n', 'Lab results', '\n')
    for lab_res_col in lab_result_col_idxs:
        col_name = df_a.columns[lab_res_col]
        col_a = df_a[col_name]
        col_b = df_b[col_name]
        U1, p = sp.stats.mannwhitneyu(col_a.dropna(), col_b.dropna())
        # cont_tab = make_categorical_cont_table(col_a, col_b)
        # oddsratio, pvalue = fisher_exact(cont_tab)
        print( col_name, U1, 'pval' ,p)
        
    print('\n','Nurological outcomes', '\n')
    col_a = df_a[neurological_outcomes_col].astype(str).isin(neurological_outcomes_codes)
    col_b = df_b[neurological_outcomes_col].astype(str).isin(neurological_outcomes_codes)
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)
    
    print('\n','BUN to Creatinine ratio', '\n')
    valid_BUN_to_creat_col_a = df_a[abs(df_a[BUN_to_creat_TD_col])<hours_in_day][BUN_to_creat_col] 
    n_BUN_to_creat_col_a = len(valid_BUN_to_creat_col_a)
    valid_BUN_to_creat_col_b = df_b[abs(df_b[BUN_to_creat_TD_col])<hours_in_day][BUN_to_creat_col] 
    n_BUN_to_creat_col_b = len(valid_BUN_to_creat_col_b)        
    U1, p = sp.stats.mannwhitneyu(valid_BUN_to_creat_col_a.dropna(), valid_BUN_to_creat_col_b.dropna())
    print( BUN_to_creat_col, U1, 'pval' ,p)
 
    print('Daily Correction Rate columns', '\n')
    for corr_rate_col in Daily_corr_rate_cols:
        col_a = df_a[corr_rate_col]==1
        col_b = df_b[corr_rate_col]==1
        cont_tab = make_categorical_cont_table(col_a, col_b)
        oddsratio, pvalue = fisher_exact(cont_tab)
        print(corr_rate_col, oddsratio, 'pval: ', pvalue)
        
    print('\n' ,'Daily sodium correction rate numeric value', '\n')
    sod_corr_daily_col_a = df_a[daily_corr_rate_col_num]
    sod_corr_daily_col_b = df_b[daily_corr_rate_col_num]
    U1, p = sp.stats.mannwhitneyu(sod_corr_daily_col_a.dropna(), sod_corr_daily_col_b.dropna())
    print( daily_corr_rate_col_num, U1, 'pval' ,p)


def analyze_group_sodium(group_df , group_name, group_color):
    
    first_sodium_col = 'sodium 1-Result numeric'
    first_corrected_col = 'First_under145'
    reached_norm_col = 'Reached_normal'
    sod_diff_col = 'Sod_diff_from_onset'
    sod_timediff_col = 'First_time_under145'
    sod_overall_corr_rate_col = 'First_under145_overall_corr_rate'
    max_corr_rate_col = 'Fastest_correction_rate'
    length_of_stay_col = 'diagnosis at discharge-Diagnosis date-Days from Reference'
    daily_corr_rate_col = 'Corr_rate_in_24_hours'
    
    
    print(group_name)
    binwidth = 2.5
    peak_sod = group_df[first_sodium_col]
    first_sod_med = group_df[first_sodium_col].quantile(0.5)
    first_sod_Q1 = group_df[first_sodium_col].quantile(0.25)
    first_sod_Q3 = group_df[first_sodium_col].quantile(0.75)
    print('First sod: ', first_sod_med, '(', first_sod_Q1,'-',first_sod_Q3, ')')
    print(peak_sod.info())
    
    # histogram plot
    graph1  = plt.hist(group_df[first_sodium_col],color=group_color,
                       bins=np.arange(min(peak_sod), max(peak_sod) + binwidth, binwidth))
    plt.xlabel('First Sodium concentration [mmol/L]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    
    first_corr_sod_med = group_df[first_corrected_col].quantile(0.5)
    first_corr_sod_Q1 = group_df[first_corrected_col].quantile(0.25)
    first_corr_sod_Q3 = group_df[first_corrected_col].quantile(0.75)
    print('First corrected sod: ',
          first_corr_sod_med, '(', first_corr_sod_Q1,'-',first_corr_sod_Q3, ')')
    
    # histogram plot
    graph2  = plt.hist(group_df[first_corrected_col],color=group_color,
                       bins=np.arange(min(group_df[first_corrected_col]),
                                      max(group_df[first_corrected_col]) + binwidth,
                                      binwidth))
    plt.xlabel('First corrected Sodium concentration [mmol/L]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    
    graph3 = plt.bar(['Yes','No'], [group_df[reached_norm_col].sum(),
                                    sum(group_df[reached_norm_col]==0)], color=group_color)
    plt.xlabel('Reached normal Sodium levels')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    print('Reached corrected levels : ', group_df[reached_norm_col].sum() ,
          '(', round(group_df[reached_norm_col].sum()/len(group_df)*100,2), ')')
    
    sod_diff_from_start = group_df[sod_diff_col]
    graph4  = plt.hist(sod_diff_from_start,color=group_color,
                   bins=np.arange(min(sod_diff_from_start),
                                  max(sod_diff_from_start) + binwidth,
                                  binwidth))
    plt.xlabel('Sodium difference from onset [mmol/L]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    print('Sodium difference from onset: ',
          sod_diff_from_start.quantile(0.5), '(', sod_diff_from_start.quantile(0.25),
          '-',sod_diff_from_start.quantile(0.75), ')')
    
    corrected_sod_timediff = group_df[sod_timediff_col]
    graph5  = plt.hist(corrected_sod_timediff,color=group_color,
                       bins=np.arange(min(corrected_sod_timediff),
                                      max(corrected_sod_timediff) + 4*binwidth,
                                      4*binwidth))
    plt.xlabel('Sodium correction time-diff [hours]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    print('Time to Sodium correction: ',
          round(corrected_sod_timediff.quantile(0.5),2), '(',
          round(corrected_sod_timediff.quantile(0.25),2),
          '-',round(corrected_sod_timediff.quantile(0.75),2), ')')
    
    correct_rate = group_df[sod_overall_corr_rate_col]
    graph6  = plt.hist(correct_rate,color=group_color,
                       bins=np.arange(min(correct_rate),
                                      max(correct_rate) + 0.01*binwidth,
                                      0.01*binwidth))
    plt.xlabel('Overall Sodium correction rate [mmol/L/h]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    print('Overall Sodium correction rate: ',
          round(correct_rate.quantile(0.5),2), '(',
          round(correct_rate.quantile(0.25),2),
          '-',round(correct_rate.quantile(0.75),2), ')')
    
    max_corr_rate = group_df[max_corr_rate_col]
    graph7 = plt.hist(max_corr_rate, color=group_color,
                      bins=np.arange(min(max_corr_rate),
                                     max(max_corr_rate) +binwidth,
                                     binwidth))
    plt.xlabel('Maximal Sodium correction rate in period [mmol/L/h]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    print('Maximal Sodium correction rate: ',
          round(max_corr_rate.quantile(0.5),2), '(',
          round(max_corr_rate.quantile(0.25),2),
          '-',round(max_corr_rate.quantile(0.75),2), ')')
    
    length_of_stay = group_df[length_of_stay_col]
    graph8 = plt.hist(length_of_stay, color=group_color,
                      bins=np.arange(np.nanmin(length_of_stay),
                                      np.nanmax(length_of_stay) +binwidth,
                                      binwidth))
    plt.xlabel('Length of stay [d]')
    plt.ylabel('Counts')
    plt.title(group_name)
    plt.show()
    print('Length of stay in the hospital: ',
          round(length_of_stay.quantile(0.5),2), '(',
          round(length_of_stay.quantile(0.25),2),
          '-',round(length_of_stay.quantile(0.75),2), ')')


    first_24hour_correction = group_df[daily_corr_rate_col]
    print('Correction during first 24-hours: ',
          round(first_24hour_correction.quantile(0.5),2), '(',
          round(first_24hour_correction.quantile(0.25),2),
          '-',round(first_24hour_correction.quantile(0.75),2), ')')
    


def compare_group_sodium_stats(df_a, df_b):

    first_sodium_col = 'sodium 1-Result numeric'
    first_corrected_col = 'First_under145'
    reached_norm_col = 'Reached_normal'
    sod_diff_col = 'Sod_diff_from_onset'
    sod_timediff_col = 'First_time_under145'
    sod_overall_corr_rate_col = 'First_under145_overall_corr_rate'
    max_corr_rate_col = 'Fastest_correction_rate'
    length_of_stay_col = 'diagnosis at discharge-Diagnosis date-Days from Reference'
    columns_to_compare = [first_sodium_col, first_corrected_col,
                          sod_diff_col, sod_timediff_col,
                          sod_overall_corr_rate_col, max_corr_rate_col,
                          length_of_stay_col]

    for col in columns_to_compare:
        col_a = df_a[col]
        col_b = df_b[col]
        U1, p = sp.stats.mannwhitneyu(col_a.dropna(), col_b.dropna())

        print(col)
        print(U1, p)
        
    print('Proportion reached correction')
    col_a = df_a[reached_norm_col]
    col_b = df_b[reached_norm_col]
    cont_tab = make_categorical_cont_table(col_a, col_b)
    oddsratio, pvalue = fisher_exact(cont_tab)
    print( oddsratio, 'pval' ,pvalue)  
    


def plot_KaplanMayer_curve(df_a, df_b, group_a_name, group_b_name,fig_title,
                           group_a_color, group_b_color, weightings_str, csv_name):
    decease_time_col = 'exitus-Deceased date-Days from Reference' 
    UPPER_THRESH_FOR_FIG_DAYS = 1500
    fig, fig_ax = plt.subplots()

    
    death_time_a = df_a[decease_time_col]
    if_death_a = death_time_a.apply(lambda x: x>0)
    death_time_a = death_time_a.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    kmf_a = lifelines.KaplanMeierFitter()
    kmf_a.fit(death_time_a, if_death_a, label=group_a_name, timeline = np.arange(0,31,1))
    fig_ax = kmf_a.plot(ax=fig_ax, color=group_a_color)
    
    death_time_b = df_b[decease_time_col]
    if_death_b = death_time_b.apply(lambda x: x>0)
    death_time_b = death_time_b.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    kmf_b = lifelines.KaplanMeierFitter()
    kmf_b.fit(death_time_b, if_death_b, label=group_b_name, timeline=np.arange(0,31,1))
    fig_ax = kmf_b.plot(ax=fig_ax, color=group_b_color)
    
    
    df_a_tocsv = pd.DataFrame({'Time': death_time_a, 'Event': if_death_a, 'Group': 'Slow'})
    df_b_tocsv = pd.DataFrame({'Time': death_time_b, 'Event': if_death_b, 'Group': 'Fast'})
    all_tocsv = pd.concat([df_a_tocsv, df_b_tocsv])
    all_tocsv.to_csv(csv_name+'.csv', index=False)
    
    # These calls below are equivalent
    # lifelines.plotting.add_at_risk_counts(kmf_a, kmf_b)
    lifelines.plotting.add_at_risk_counts(kmf_a, kmf_b ,rows_to_show=['At risk'],
                                          xticks=[0,5,10,15,20,25,30], ax=fig_ax, fig=fig)
    plt.tight_layout()
    
    # fig_ax.set_xlim([0,30])
    fig_ax.set_ylim([0,1])
    fig_ax.set_xlabel('Time [days]')
    fig_ax.set_ylabel('Survival probability')
    fig_ax.set_title(fig_title)
    
    fig.savefig(fig_title+'_nsurvs.svg',format='svg' ,dpi=300)
    
    print('Compare at 30 days')
    surv_stats_30days = survival_difference_at_fixed_point_in_time_test(30, kmf_a, kmf_b)

    surv_stats_30days.print_summary()
    print(surv_stats_30days.p_value)
    
    #% For statistical test regarding 30 day mortality 
    if_death_a_in_30 = death_time_a.apply(lambda x: x<=30)
    death_time_a_compare30 = death_time_a.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    if_death_b_in_30 = death_time_b.apply(lambda x: x<=30)
    death_time_b_compare30 = death_time_b.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    
    
    print(fig_title)
    print('Log-rank test')
    results = logrank_test(death_time_a, death_time_b, event_observed_A=if_death_a,event_observed_B=if_death_b)
    # results = logrank_test(death_time_a, death_time_b, if_death_a, if_death_b)
    results.print_summary()
    print(results.p_value)        
    print(results.test_statistic)
    
    
    fhr_a = lifelines.BreslowFlemingHarringtonFitter()
    fhr_a.fit(death_time_a, if_death_a, label=group_a_name, timeline = np.arange(0,31,1))

    fhr_b = lifelines.BreslowFlemingHarringtonFitter()
    fhr_b.fit(death_time_b, if_death_b, label=group_b_name, timeline = np.arange(0,31,1))   
    print('\n','Flemington-harrington')
    results = logrank_test(death_time_a, death_time_b, event_observed_A=if_death_a,event_observed_B=if_death_b,
                           weightings=weightings_str, p=1,q=0)
    # results = logrank_test(death_time_a, death_time_b, if_death_a, if_death_b)
    results.print_summary()
    print(results.p_value)        
    print(results.test_statistic)
    
    # results = logrank_test(death_time_a_compare30, death_time_b_compare30, if_death_a_in_30, if_death_b_in_30,
    #                        weightings=weightings_str)
    
    
    #% For statistical test regarding 60 day mortality 
    if_death_a_in_7 = death_time_a.apply(lambda x: x<=7)
    death_time_a_compare7 = death_time_a.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    if_death_b_in_7 = death_time_b.apply(lambda x: x<=7)
    death_time_b_compare7 = death_time_b.fillna(UPPER_THRESH_FOR_FIG_DAYS)
   
    
    #% For statistical test regarding 365 day mortality 
    if_death_a_in_year = death_time_a.apply(lambda x: x<=365)
    death_time_a_compare_year = death_time_a.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    if_death_b_in_year = death_time_b.apply(lambda x: x<=365)
    death_time_b_compare_year = death_time_b.fillna(UPPER_THRESH_FOR_FIG_DAYS)
    
    return fhr_a

def analyze_mortality_odds_ratio(df_a, df_b, compare_name):
    
    print(compare_name)
    mortality_30days_col = '30 day mortality 1 - Yes 0 - No'
    # mortality_30days_col = '7 day mortality 1 - Yes 0 - No'


    age_col = 'Reference Event-Age at lab test'
    gender_col = 'Gender 1 = M 2 =F'
    charlson_score_col = 'charlson-Charlson Score'
    
    creatinine_col = 'createnine-Result numeric'
    
    glucose_over_thresh_col = 'Initial_gluc>thresh'
    Is_ICU_col = 'Is_ICU'
    first_sodium_col = 'sodium 1-Result numeric'
    potassium_col = 'potasium-Result numeric'
    
    COPD_col = 'charlson-COPD 1= yes 0 = no'
    CHF_col = 'charlson-Congestive Heart Failure 1= yes 0 = no'
    Dementia_col = 'charlson-Dementia 1= yes 0 = no'
    Renal_dis_col = 'charlson-Renal Disease 1= yes 0 = no'
    
    group_a_mor = df_a[mortality_30days_col].sum()
    group_b_mor = df_b[mortality_30days_col].sum()
    
    group_a_alive = len(df_a)-group_a_mor
    group_b_alive = len(df_b)-group_b_mor
    
    print('Group A size: ', len(df_a))
    print('Group B size: ', len(df_b))
    
    print('Mortality Group A: ', group_a_mor, round(group_a_mor/(group_a_mor+group_a_alive)*100,2))
    print('Mortality Group B: ', group_b_mor, round(group_b_mor/(group_b_mor+group_b_alive)*100,2))
    
    cont_table = np.array([[group_a_mor, group_a_alive],[group_b_mor, group_b_alive]])
    oddsr, p = sp.stats.fisher_exact(cont_table, alternative='two-sided')
    print(oddsr, p)
    
    CI_up = math.exp(np.log(oddsr)+1.96*math.sqrt(1/group_a_mor+1/group_b_mor+
                                                   1/group_a_alive+1/group_b_alive)) 
    CI_down = math.exp(np.log(oddsr)-1.96*math.sqrt(1/group_a_mor+1/group_b_mor+
                                                   1/group_a_alive+1/group_b_alive)) 
    
    print( 'Odds ratio: ', round(oddsr,2), '(', round(CI_down,2), '-', round(CI_up,2), ')')
    
    mortality_vec_a = df_a[mortality_30days_col].values
    mortality_vec_b = df_b[mortality_30days_col].values
    
    age_vec_a = df_a[age_col].values
    age_vec_b = df_b[age_col].values
    male_vec_a = df_a[gender_col].values
    male_vec_a = np.where(male_vec_a != 1, 0, male_vec_a)
    male_vec_b = df_b[gender_col].values
    male_vec_b = np.where(male_vec_b != 1, 0, male_vec_b)
    charls_vec_a = df_a[charlson_score_col].values
    charls_vec_b = df_b[charlson_score_col].values
    
    #extended correction - new
    creatinine_vec_a = df_a[creatinine_col].values
    creatinine_vec_b = df_b[creatinine_col].values
    first_sod_vec_a = df_a[first_sodium_col].values
    first_sod_vec_b = df_b[first_sodium_col].values
    potassium_vec_a = df_a[potassium_col].values
    potassium_vec_b = df_b[potassium_col].values

    is_icu_vec_a = df_a[Is_ICU_col].values
    is_icu_vec_b = df_b[Is_ICU_col].values
    is_hyperglycemic_vec_a = df_a[glucose_over_thresh_col].values
    is_hyperglycemic_vec_b = df_b[glucose_over_thresh_col].values    
    
    
    mortality_vec_total = np.concatenate([mortality_vec_a, mortality_vec_b])
    diff_rate_vec = np.concatenate([np.ones(len(df_a)), np.zeros(len(df_b))])
    total_age_vec = np.concatenate([age_vec_a,age_vec_b])
    total_gender_vec = np.concatenate([male_vec_a, male_vec_b])
    total_charls_vec = np.concatenate([charls_vec_a, charls_vec_b])
    
    total_creatinine_vec = np.concatenate([creatinine_vec_a, creatinine_vec_b])
    total_first_sod_vec = np.concatenate([first_sod_vec_a, first_sod_vec_b])
    total_potassium_vec = np.concatenate([potassium_vec_a, potassium_vec_b])
    total_is_ICU_vec = np.concatenate([is_icu_vec_a, is_icu_vec_b])
    total_is_hyperglycemic = np.concatenate([is_hyperglycemic_vec_a, is_hyperglycemic_vec_b])
    
    
    all_vars_extended_array = np.c_[diff_rate_vec, total_age_vec, total_gender_vec,
                                    total_charls_vec, total_creatinine_vec,
                                    total_first_sod_vec, total_potassium_vec,
                                    total_is_ICU_vec, total_is_hyperglycemic]
    mortality_vec_extended_adj = mortality_vec_total[~np.isnan( \
                                        all_vars_extended_array).any(axis=1)]
    all_vars_extended_array = all_vars_extended_array[~np.isnan(all_vars_extended_array).any(axis=1), :]
    print('\n', 'Num of patients in final analysis:')
    print(all_vars_extended_array.shape, '\n')
    
    diff_rate_vec_na = sm.add_constant(diff_rate_vec)
    non_adj_model = sm.Logit(mortality_vec_total, diff_rate_vec_na)
    na_result = non_adj_model.fit(method='newton')
    print(na_result.summary())
    
    
    all_vars_extended_array = sm.add_constant(all_vars_extended_array)
    extended_adj_model = sm.Logit(mortality_vec_extended_adj, all_vars_extended_array)
    extended_adj_result = extended_adj_model.fit(method='newton')
    print(extended_adj_result.summary())
    
    print('Var order: offset, correction rate, age, gender, charlson,' '\n',
          'creatinine, first sodium, potassium, ICU, hyperglycemic' )

    return na_result, extended_adj_result

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as st
from statsmodels.stats.proportion import proportions_ztest

def clean_columns(df):
    '''
    This function cleans the columns to have them in a standard manner.
    '''
    df.columns = df.columns.str.lower()
    df.rename(columns={'bal': 'balance'}, inplace=True)
    df['client_id'] = df['client_id'].astype(str)
    
    return df

def update_dtypes(df):
    '''
    This function changes the dtypes of the selected columns to it's appropiate dtype.
    '''
    df["client_id"] = df["client_id"].astype(str)
    df["date_time"] = pd.to_datetime(df["date_time"])
    
    return df

def general_clean(df):
    '''
    Drops null values and changes values inside gendr column.
    '''
    
    df.dropna(inplace = True)
    df["gendr"] = df["gendr"].replace("X", "U") 
    
    return df

def filtered_age_chart(fd):
    '''
    Filters age and creates a chart with oultining the 2 peaks.
    '''
        # Filter for ages between 25 and 35
    filtered_age_25_35 = fd[(fd['clnt_age'] >= 25) & (fd['clnt_age'] <= 35)]
    count_age_25_35 = filtered_age_25_35['clnt_age'].count()
    
    # Filter for ages between 53 and 63
    filtered_age_53_63 = fd[(fd['clnt_age'] >= 53) & (fd['clnt_age'] <= 63)]
    count_age_53_63 = filtered_age_53_63['clnt_age'].count()
    
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    sns.histplot(fd['clnt_age'], kde=True, color="salmon", bins=15)
    
    # Add vertical lines for the age ranges
    plt.axvline(25, color='blue', linestyle='--', label='Age 25')
    plt.axvline(35, color='blue', linestyle='--', label='Age 35')
    plt.axvline(53, color='green', linestyle='--', label='Age 53')
    plt.axvline(63, color='green', linestyle='--', label='Age 63')
    
    # Add text annotations for counts
    plt.text(30, 5, f'Count (25-35): {count_age_25_35}', fontsize=12, color='black', ha='center')
    plt.text(58, 5, f'Count (53-63): {count_age_53_63}', fontsize=12, color='black', ha='center')
    
    # Customize plot
    plt.title('Client Age Distribution with Counts for Age Ranges')
    plt.xlabel('Client Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Print the counts of people aged in both ranges
    print(f'Number of people aged between 25 and 35: {count_age_25_35}')
    print(f'Number of people aged between 53 and 63: {count_age_53_63}')
    
def age_categories(df):
    '''
    Here we group ages in order to make categories.
     0-18 yrs are teens, 18-25 are young adults, 25-55 are adults and 55+ are seniors.
    '''
    bins = [0, 18, 25, 55,df['clnt_age'].max()]
    labels = ['Teen', 'Young Adult', 'Adult', "Senior"]
    df['clnt_age_category'] = pd.cut(df['clnt_age'], bins=bins, labels=labels, include_lowest=True)

def basic_stats(df):
    median = df["balance"].median()
    skew = df["balance"].skew()
    kurtosis = df["balance"].kurt() 
    q_05 = df["balance"].quantile(0.5)
    q_075 = df["balance"].quantile(0.75)
    q_095 = df["balance"].quantile(0.95)
    
    print(f"The median is: {median}")
    print(f"The skew is: {skew}")
    print(f"The kurtosis is: {kurtosis}")
    print(f"Quantile 0.5 is: {q_05}")
    print(f"Quantile 0.75 is: {q_075}")
    print(f"Quantile 0.95 is: {q_095}")

def clean_balance_chart(df):
    '''
    This function gets excludes the outliers and only uses data up to percentile 95 in order to have a better view of the graph.
    '''
    
    # Calculate the 95th percentile of the 'balance' column
    balance_95th_percentile = df['balance'].quantile(0.95)
    
    # Filter the data to include only balances up to the 95th percentile
    filtered_data = df[df['balance'] <= balance_95th_percentile]
    
    # Plot the histogram of the filtered data
    plt.figure(figsize=(10, 6))
    sns.histplot(filtered_data['balance'], kde=True, color="salmon")
    
    # Customize the plot
    plt.title('Histogram of Account Balances (First 95% of Data)', fontsize=14)
    plt.xlabel('Balance')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    
def completion_rate_kpi(variation_group):
    '''
    This function creates a completion rate KPI taking as an argument the desired group.
    It basically compares the total amout of clients that completed the proccess in the 
    given order against the total amout of clients in the same group.
    '''
    
# Define the required steps
    required_steps = {"start", "step_1", "step_2", "step_3"}
    
    
    # Function to check if a user went through all required steps before confirming
    def has_all_required_steps(steps, required_steps):
        # Convert the steps to a set of unique steps
        steps_set = set(steps)
        # Check if all required steps are in the steps_set
        return required_steps.issubset(steps_set)
    
        # Group the dataframe by client_id and check if they passed through all required steps
    valid_clients = variation_group.groupby("client_id")["process_step"].apply(lambda steps: has_all_required_steps(steps, required_steps))
    
    # Filter to get only valid client_ids that followed the correct steps
    valid_client_ids = valid_clients[valid_clients].index
    
    # Now, filter the original dataframe to get only those with the correct step order
    total_confirmed = variation_group[(variation_group["process_step"] == "confirm") & (variation_group["client_id"].isin(valid_client_ids))]
    
    
    
    # Get the number of unique client_ids that reached the 'confirm' step after following the correct steps
    total_confirmed_count = total_confirmed["client_id"].nunique()
    
    total_participation = variation_group["client_id"].nunique()
    
    #Calculation of the completion rate
    completion_rate = total_confirmed_count / total_participation *100
    return f"The completion rate KPI is: {round(completion_rate, 2)} %"


def time_per_step_kpi(variation_group):
    '''
    This function calculates the time per step KPI.
    '''
    df = variation_group.sort_values(by=['client_id', 'visit_id', 'date_time'])
    df['time_diff'] = df.groupby(['client_id', 'visit_id'])['date_time'].diff().shift(-1).dt.total_seconds()
    average_time_per_step = df.groupby('process_step')['time_diff'].mean().reset_index()
    
    return average_time_per_step

def error_rate_kpi(variation_group):
    '''
    This function calculates the error rate KPI.
    '''
    
    #TEST GROUP
    ## Error Rates: If there’s a step where users go back to a previous step, it may indicate confusion or an error. 
    #You should consider moving from a later step to an earlier one as an error.
    
    # Step 1: Define the correct order of process steps
    step_order = ['start', 'step_1', 'step_2', 'step_3', 'confirm']
    
    # Step 2: Convert process_step to categorical with the defined order
    variation_group['process_step'] = pd.Categorical(variation_group['process_step'], categories=step_order, ordered=True)
    
    # Step 3: Sort the DataFrame by client_id, visit_id, and date_time
    variation_group = variation_group.sort_values(by=['client_id', 'visit_id', 'date_time'])
    
    # Step 4: Shift the process_step column to compare with the previous step within each client/visit group
    variation_group['prev_process_step'] = variation_group.groupby(['client_id', 'visit_id'])['process_step'].shift(1)
    
    # Step 5: Identify errors where the process_step moves backward
    variation_group['error'] = variation_group['process_step'] < variation_group['prev_process_step']
    
    # Step 6: Count the number of errors per client_id (or globally)
    total_errors = variation_group['error'].sum()
    
    # Display the DataFrame with errors and total error count
    #print(test_df[['client_id', 'visit_id', 'process_step', 'prev_process_step', 'error']])
    print(f"\nTotal number of errors: {total_errors}")
    print(f"Error Rate: {round(total_errors/variation_group['process_step'].count()*100, 2)}%")
    
    
def completion_hypothesis(test_df, control_df):
    '''
    This function runs a hypothesis test related to the completion rate.
    '''
    
    # Test and Control Group Data
    completion_rate_test = 0.68
    completion_rate_control = 0.6462
    
    # Number of users in the Test and Control groups
    n_test = test_df["client_id"].nunique() 
    n_control = control_df["client_id"].nunique()
    
    # Convert completion rates to actual counts of completions
    completions_test = int(completion_rate_test * n_test)
    completions_control = int(completion_rate_control * n_control)
    
    # Store the completion counts and number of users
    counts = np.array([completions_test, completions_control])
    nobs = np.array([n_test, n_control])
    
    # Run a Z-test for proportions
    stat, p_value = proportions_ztest(counts, nobs)
    
    # Set alpha for 5% significance level
    alpha = 0.05
    
    
    #Stat tells you which way it is. It is positive, which means the first one is higher than the second [Completion test > completion control]
    print(f"Stat: {stat}")  
 
    
    # Check the p-value and print the result
    if p_value < alpha:
        print(f"We reject the null hypothesis (p-value = {p_value:.20f}). The completion rates are significantly different between the Test and Control groups.")
    else:
        print(f"We fail to reject the null hypothesis (p-value = {p_value:.20f}). No significant difference in completion rates between the Test and Control groups.")

def error_rate_hypothesis(test_df, control_df):
        '''
    This function runs a hypothesis test related to the error rate.
    '''
    
    
    # Test and Control Group Data
    error_rate_test = 0.0914
    error_rate_control = 0.067
    
    # Number of users in the Test and Control groups
    n_test = 26961 
    n_control = 23526
    
    # Convert error rates to actual counts of errors
    error_test = int(error_rate_test * n_test)
    error_control = int(error_rate_control * n_control)
    
    # Store the error counts and number of users
    counts = np.array([error_test, error_control])
    nobs = np.array([n_test, n_control])
    
    # Run a Z-test for proportions
    stat, p_value = proportions_ztest(counts, nobs)
    
    # Set alpha for 5% significance level
    alpha = 0.05
    
        #Stat tells you which way it is. It is positive, which means the first one is higher than the second [Completion test > completion control]
    print(f"Stat: {stat}")  
    
    # Check the p-value and print the result
    if p_value < alpha:
        print(f"We reject the null hypothesis (p-value = {p_value:.30f}). The error rates are significantly different between the Test and Control groups.")
    else:
        print(f"We fail to reject the null hypothesis (p-value = {p_value:.30f}). No significant difference in error rates between the Test and Control groups.")
        

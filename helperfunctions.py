## helper functions for Triage model

import numpy as np
import pandas as pd
import os
import pickle
import random
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import math
import scipy.stats as st
import statsmodels.api as sm


############################### DATA ####################################

def pickle_dump(obj_to_pkl, filename, data_path_stem, lclyn='y', lcl_pkl="C://Users//TYWARD//Documents"):
    filepath = data_path_stem + filename
    if os.path.exists(filepath):
        print('File already exists. Overwrite (y/n)?')
        x = input()
    else: 
        x = 'y'
    if x == 'y':
        outfile = open(filepath, 'wb')
        pickle.dump(obj_to_pkl, outfile)
        outfile.close()
        print('Wrote object to file.')
    else:
        print('Did not write to file.')
    lcl_pkl_path = lcl_pkl + filename
    if lclyn == 'y':
        if os.path.exists(lcl_pkl_path):
            print('File already exists. Overwrite (y/n)?')
            x = input()
        else: 
            x = 'y'
        if x == 'y':
            outfile = open(lcl_pkl_path, 'wb')
            pickle.dump(obj_to_pkl, outfile)
            outfile.close()
            print('Wrote object to file (local).')
        else:
            print('Did not write to file (local).')

def pickle_read(filepath):
    if os.path.exists(filepath):
        infile = open(filepath, 'rb')
        data_to_rtn = pickle.load(infile)
        infile.close()
        print('File exists - wrote to object.')
        return data_to_rtn
    else: 
        print('File does not exist.')

################################## METRICS ###############################

def standard_confusion_matrix(y_true, y_pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def profit_curve(cost_benefit_mat, y_pred_proba, y_true):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(y_true))
    # Make sure that 1 is going to be one of our thresholds
#     maybe_one = [] if 1 in y_pred_proba else [1]
    thresholds = sorted(list(np.linspace(0,1,101)), reverse=True)
    profits = []
    for threshold in thresholds:
#         print(threshold)
        y_predict = y_pred_proba >= threshold
        confusion_matrix = standard_confusion_matrix(y_true, y_predict)
#         print(confusion_matrix)
        threshold_profit = np.sum(confusion_matrix * cost_benefit_mat)/n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)

def profit_curve_cost(y_pred_proba, y_cost, cost_red=.05, intervent_cost=300):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    y_cost_reduct = y_cost * cost_red

    n_obs = float(len(y_cost))
    # Make sure that 1 is going to be one of our thresholds
#     maybe_one = [] if 1 in y_pred_proba else [1]
    thresholds = sorted(list(np.linspace(0,1,101)), reverse=True)
    profits = []
    for threshold in thresholds:
#         print(threshold)
        y_pred = y_pred_proba >= threshold
        # confusion_matrix = standard_confusion_matrix(y_true, y_predict)
#         print(confusion_matrix)
        threshold_profit = ((y_pred * y_cost_reduct).sum() - (y_pred.sum()*intervent_cost)) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)

############################### VISUALIZATIONS ###########################

def plot_ci_model_covars(covars, bs_ests, title, image_path_stem, image_name):
    fig = plt.figure(figsize=(5,2))
    ax = fig.add_subplot(111)
    for i in range(len(covars)):
        var = covars[i]
        ax.plot([(np.percentile(bs_ests[var], 5)), (np.percentile(bs_ests[var], 95))],
                 [var,var], color='black')
        ax.plot((np.percentile(bs_ests[var], 50)), i, 'o', markersize=3, color='red')
    # ax.plot
    ax.plot([0,0], [-0.5,(len(covars)-.5)], '--', color='black')
    ax.set_title(title)
    ax.set_ylim([-0.5,len(covars)-0.5])
    ax.set_yticks(covars)
    fig.savefig(img_pth_stm + '_' + image_name + '.jpg', dpi=300, bbox_inches='tight', pad_inches=.1);

def plot_precision_recall_curve(y_true, y_pred_proba, plot_mins_tf=False,
	min_precision=.4, min_recall=.4,
	 title='Precision, Recall by Model Threshold',
	 imagepath='../images/precision_recall_curve.jpg'):
    proffig = plt.figure(figsize=(8,6))
    ax1 = proffig.add_subplot(111)
    if plot_mins_tf:
    	ax1.plot([0,1],[min_precision,min_precision], color='black') ## from x1,y1 to x2,y2 would be [x1, x2], [y1, y2]
    	ax1.plot([min_recall,min_recall],[0,1], color='black')
    	rect_patch = plt.Rectangle((min_recall, min_precision), (1-min_recall), (1-min_precision), alpha=.2, color='green')
    	ax1.add_patch(rect_patch)
    	ax1.text(.6, .9, 'Desired Precision, Recall', color='darkgreen')
    	ax1.text(min_recall+.01, 0, 'Min Recall = {}'.format(min_recall))
    	ax1.text(0, min_precision-.05, 'Min Precision = {}'.format(min_precision))
    # ax1.plot([1,0],[0,0], c='black')
    # profy = max(profs)
    # profx = float(profthresh[(np.where(max(profs) == profs)[0])])
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba, pos_label=None, sample_weight=None)
    auprc = average_precision_score(y_true, y_pred_proba)
    ax1.set_title(title)
    ax1.set_ylabel('Precision')
    ax1.set_xlabel('Recall')
    # ax1.set_xlim([xmin,xmax])
    # ax1.set_ylim([ymin,ymax])
    ax1.plot(recall, precision, color='blue', label="AUPRC = {0:0.3f}".format(auprc))
    # ax1.plot(profx, profy, 'ro', label='Max of ${0:.2f} at threshold of {1}'.format(profy, profx))
    ax1.legend()
    proffig.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1);

def plot_profit_curve(profthresh, profs, 
	xmax=1, xmin=0, ymax=1000, ymin=-1000,
	 title='Program Net Benefit, by Model Threshold',
	 y_label='Net Benefit ($s), PMPM', imagepath='../images/profit_curve.jpg'):
    proffig = plt.figure(figsize=(8,6))
    ax1 = proffig.add_subplot(111)
    ax1.plot([1,0],[0,0], c='black')
    profy = max(profs)
    profx = float(profthresh[(np.where(max(profs) == profs)[0])])
    ax1.set_title(title)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel('Thresholds')
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    ax1.plot(profthresh, profs, color='black')
    ax1.plot(profx, profy, 'ro', label='Max of ${0:.2f} at threshold of {1}'.format(profy, profx))
    ax1.legend()
    proffig.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1);


def plot_profit_curves_cost(y_pred_proba, y_cost, cost_red_array=[.03,.05,.03,.05], 
    intervent_cost_array=[200,200,400,400], labels=['LL','HL', 'LH', 'HH'],
    xmax=1, xmin=0, ymax=1000, ymin=-1000,
     title='Program Net Benefit, by Model Threshold',
     y_label='Net Benefit ($s), PMPM', imagepath='../images/profit_curve_cost.jpg'):
    proffig = plt.figure(figsize=(8,6))
    ax1 = proffig.add_subplot(111)
    ax1.plot([1,0],[0,0], c='black')
    for cost_red, intervent, lab in zip(cost_red_array, intervent_cost_array, labels):
        profs, threshes = profit_curve_cost(y_pred_proba, y_cost, cost_red, intervent)    
        profy = max(profs)
        profx = float(threshes[(np.where(max(profs) == profs)[0])])
        ax1.plot(threshes, profs, color='black')
        ax1.plot(profx, profy, 'o', label='{0}, {1}: ${2:.2f}'.format(cost_red, intervent, profy))
    ax1.set_title(title)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel('Thresholds')
    ax1.set_xlim([xmin,xmax])
    ax1.set_ylim([ymin,ymax])
    ax1.legend()
    proffig.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1);

def plot_bar_chart(df, quantile_var, y_true, title_prefix='High Utilizer Rate', imagepath='../images/myimage.jpg'):
    fig_name = 'fig'+y_true
    ax_name = 'ax'+y_true
    if len(df[quantile_var].unique()) == 10:
    	x_var_label = 'Decile'
    elif len(df[quantile_var].unique()) == 20:
    	x_var_label = 'Ventile'
    else:
    	x_var_label = 'Quantile'
    if df[quantile_var].min() == 1:
    	add1 = 0
    else:
    	add1 = 1
    fig_name = plt.figure(figsize=(12,6))
    ax_name = fig_name.add_subplot(111)
    ax_name.set_title(title_prefix+' by '+x_var_label)
    ax_name.bar(df.groupby(quantile_var)[y_true].mean().index + add1, df.groupby(quantile_var)[y_true].mean()*100)
    mean_rate = df[y_true].mean() *100
    num_quant = (df[quantile_var].unique().shape[0])
    ax_name.plot([.5, num_quant+.5], [mean_rate, mean_rate], '-', lw=3, color='darkorange')  ## from x1,y1 to x2,y2 would be [x1, x2], [y1, y2]
    ax_name.text(.5, (mean_rate+0.3), 'population rate')
    ax_name.set_xticks((df.groupby(quantile_var)[y_true].mean().index + add1))
    ax_name.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_name.set_ylabel(title_prefix)
    ax_name.set_xlabel(x_var_label)
    if os.path.exists(imagepath):
        print('Image already exists. Overwrite (y/n)?')
        x = input()
    else: 
        x = 'y'
    if x == 'y':
        fig_name.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1)

def plot_bar_chart_continous(df, quantile_var, cont_var, title_prefix='AVG. 60-Day ND INP COST', imagepath='../images/my_cont_var_chart.jpg'):
    ## plot bar chart is for binary vars, this is for continous vars
    ## returns mean of continuos variable for each quantile
    fig_name = 'fig'+cont_var
    ax_name = 'ax'+cont_var
    if len(df[quantile_var].unique()) == 10:
        x_var_label = 'Decile'
    elif len(df[quantile_var].unique()) == 20:
        x_var_label = 'Ventile'
    else:
        x_var_label = 'Quantile'
    if df[quantile_var].min() == 1:
        add1 = 0
    else:
        add1 = 1
    fig_name = plt.figure(figsize=(12,6))
    ax_name = fig_name.add_subplot(111)
    ax_name.set_title(title_prefix+' by '+x_var_label)
    ax_name.bar(df.groupby(quantile_var)[cont_var].sum().index + add1, df.groupby(quantile_var)[cont_var].mean())
    # mean_rate = df[y_true].mean() *100
    # num_quant = (df[quantile_var].unique().shape[0])
    # ax_name.plot([.5, num_quant+.5], [mean_rate, mean_rate], '-', lw=3, color='darkorange')  ## from x1,y1 to x2,y2 would be [x1, x2], [y1, y2]
    # ax_name.text(.5, (mean_rate+0.3), 'population rate')
    ax_name.set_xticks((df.groupby(quantile_var)[cont_var].sum().index + add1))
    # ax_name.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_name.set_ylabel(title_prefix)
    ax_name.set_xlabel(x_var_label)
    if os.path.exists(imagepath):
        print('Image already exists. Overwrite (y/n)?')
        x = input()
    else: 
        x = 'y'
    if x == 'y':
        fig_name.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1)

def plot_dist_by_target(df, target, variable, num_bins=10, imagepath='../images/myimage.jpg'):
    fig_name = 'fig'+variable
    ax_name = 'ax'+variable
    fig_name = plt.figure(figsize=(12,6))
    ax_name = fig_name.add_subplot(111)
    ax_name.set_title('Histogram, '+variable)
    ax_name.hist(df[df[target]==1][variable], bins = num_bins, alpha = 0.5, density=1, label=target)
    ax_name.hist(df[df[target]==0][variable], bins = num_bins, alpha = 0.5, color='g', density=1, label='not '+target)
    ax_name.set_ylabel('Frequency')
    ax_name.legend()
    if os.path.exists(imagepath):
        print('Image already exists. Overwrite (y/n)?')
        x = input()
    else: 
        x = 'y'
    if x == 'y':
        fig_name.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=0);
 
def plotroc(TPR, FPR):
    roc_auc = auc(TPR, FPR)
    plt.figure(figsize=(8,8))
    lw = 2
    plt.plot(TPR, FPR, color='darkorange',
             lw=lw, label="ROC curve area = {0:0.4f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plotroc_by_horizon(df, varlist, y_pred_proba='score', imagepath='../images/roc_by_horizon.jpg'):
    fig3 = plt.figure(figsize=(10, 8))
    ax31 = fig3.add_subplot(111)
    lw = 2
    for var in varlist:
        TPR, FPR, thresholds = roc_curve(df[var], df[y_pred_proba], pos_label=None, sample_weight=None, drop_intermediate=True)
        roc_auc = auc(TPR, FPR)
        ax31.plot(TPR, FPR,
                 lw=lw, label="AUC for {0} = {1:0.3f}".format(var[4:], roc_auc))
    ax31.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    ax31.set_xlim([0.0, 1.0])
    ax31.set_ylim([0.0, 1.05])
    ax31.set_xlabel('False Positive Rate')
    ax31.set_ylabel('True Positive Rate')
    ax31.set_title('Receiver operating characteristic')
    ax31.legend(loc="lower right")
    fig3.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=0)

def plotroc_bymonth(df, dates_list, date_field='pred_dt', y_true_label='died90', y_pred_proba='score', imagepath='../images/myimage.jpg'):
    fig3 = plt.figure(figsize=(10, 8))
    ax31 = fig3.add_subplot(111)
    lw = 2
    for var in dates_list:
        newdf = df[df[date_field]==pd.to_datetime(var)]
        TPR, FPR, thresholds = roc_curve(newdf[y_true_label], newdf[y_pred_proba], pos_label=None, sample_weight=None, drop_intermediate=True)
        roc_auc = auc(TPR, FPR)
        ax31.plot(TPR, FPR,
                 lw=lw, label="AUC for {0} = {1:0.3f}".format(var, roc_auc))
    ax31.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    ax31.set_xlim([0.0, 1.0])
    ax31.set_ylim([0.0, 1.05])
    ax31.set_xlabel('False Positive Rate')
    ax31.set_ylabel('True Positive Rate')
    ax31.set_title('Receiver operating characteristic')
    ax31.legend(loc="lower right")
    fig3.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1)

def plot_bs_coefs(bs_coefs, col_names, nbins=10, imagepath='../images/myimage.jpg', saveyn = 'n'):
    ax_rows = math.ceil(bs_coefs.shape[1]/3)
    plt_h = ax_rows * 4
    fig, axes = plt.subplots(ax_rows,3, figsize=(14,plt_h))
    for m, ax in zip(col_names, axes.flatten()):
        ax.hist(bs_coefs[m], bins=nbins)
        zero_bar_h = ax.get_ylim()[1]
        ax.plot([0, 0], [0, zero_bar_h], color='red', linestyle='-', linewidth=2)
        ax.set_title(m)
    if saveyn == 'n':
        pass
    else:
        if os.path.exists(imagepath):
            print('Image already exists. Overwrite (y/n)?')
            x = input()
        else: 
            x = 'y'
        if x == 'y':
            fig.savefig(imagepath, dpi=300, bbox_inches='tight', pad_inches=.1)

############################### OTHER ###################################

def create_metric_chart(df, thresh_dict, y_true, y_pred_proba, quantile_var='Decile'):

    if len(df[quantile_var].unique()) == 10:
        x_var_label = 'Decile'
    elif len(df[quantile_var].unique()) == 20:
        x_var_label = 'Ventile'
    else:
        x_var_label = 'Quantile'

    if df[quantile_var].min() == 1:
        add1 = 0
    else:
        add1 = 1

    myarray = np.zeros((1,7))
    max_quantile = df[quantile_var].max()
    pop_size = df.shape[0]

    for key in thresh_dict:
        tn, fp, fn, tp = confusion_matrix(df[y_true], df[y_pred_proba]>thresh_dict[key]).ravel()
        thresh = "{0:.1f}%".format((thresh_dict[key]) * 100)
        count_cum = (df[df[y_pred_proba]>thresh_dict[key]][y_pred_proba]).count()
        if key == max_quantile:
            count = count_cum
        else:
            count = (df[df[y_pred_proba]>thresh_dict[key]][y_pred_proba]).count() - (df[df[y_pred_proba]>thresh_dict[key+1]][y_pred_proba]).count()
        perc_cum = "{0:.1f}%".format(count_cum/pop_size*100)
        # spec = "{0:.1f}%".format((tn/(tn+fp)) * 100)
        sense = "{0:.1f}%".format((tp/(fn+tp)) * 100)
        PPV = "{0:.1f}%".format((tp/(tp+fp)) * 100)
        # NPV = "{0:.1f}%".format((tn/(tn+fn)) * 100)
        # ppv_pct = tp/(tp+fp)
        # sense_pct = tp/(fn+tp)
        new = np.array([(key+add1), thresh, count, count_cum, perc_cum, PPV, sense])
        myarray = np.vstack((myarray, new))
        
    myarray = myarray[1:,:]
    mychart = pd.DataFrame(myarray, columns = [x_var_label, 'Threshold', 'Count', 'Count, Cum.', '%, Cum.', 'Precision', 'Recall'])
    mychart[x_var_label] = pd.to_numeric(mychart[x_var_label])
    # mychart['ppvpct'] = pd.to_numeric(mychart['ppvpct'])
    # mychart['sensepct'] = pd.to_numeric(mychart['sensepct'])
    mychart.sort_values(x_var_label, ascending=False, inplace=True)
    mychart.reset_index(drop=True, inplace=True)
        
    return mychart

## Function that uses "bootstrapping" to find the distribution of likely values for model coefficients.
def bootstrap_ci_coefficients(X_train, y_train, num_bootstraps, log_reg_inst, column_names):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        log_reg_inst.fit(X_samples, y_samples)
        bootstrap_estimates.append(log_reg_inst.coef_[0])
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    bootstrap_estimates = pd.DataFrame(bootstrap_estimates, columns=column_names)
    return bootstrap_estimates


## Returns bootstrapped coefficients for all variables in a poisson model
## Offset data: log of time variable


def bootstrap_ci_coeffs_pois(X_train, y_train, num_bootstraps, column_names, offset_data):
    bootstrap_estimates = []
    column_names.insert(0, 'Constant')
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train.loc[sample_index]
        y_samples = y_train.loc[sample_index]
        offset_samples = offset_data.loc[sample_index]
        model = sm.GLM(y_samples, X_samples, family=sm.families.Poisson(), offset=offset_samples)
        results = model.fit()
        bootstrap_estimates.append(results.params)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    bootstrap_estimates = pd.DataFrame(bootstrap_estimates, columns=column_names)
    return bootstrap_estimates

def convert_yrmo_to_date(df, yrmo_field, to_field):
    yrmos = df[yrmo_field].astype(str)
    dts = []
    for ym in yrmos:
        dts.append(ym[:4]+'-'+ym[4:]+'-01')
    df[to_field] = pd.to_datetime(dts)
    return df

def convert_date_to_yrmo(df, date_field, to_field):
    yrmos = []
    for index, row in df.iterrows():
        yrmos.append(int(str(row[date_field])[:4] +str(row[date_field])[5:7]))
    df[to_field] = yrmos
    return df

################################# Bootstrapping for different CIs #####################################


## find bootstrapped confidence intervals for change in counts
## from raw data after propensity matching, provides % differences in counts. 
## for admissions counts, counts limited to 3
def bootstrap_ci_diff_counts(df, num_bootstraps, col_name, group_col, num_matches=3):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(df)), len(df))
        df_samples = df.loc[sample_index]
        contingency_table = pd.crosstab(df_samples['MSR'], df_samples['f_adm'])
        c_sum = np.sum(contingency_table.loc[0] * np.array([0, 1, 2, 3]))/num_matches
        t_sum = np.sum(contingency_table.loc[1] * np.array([0, 1, 2, 3]))
        bootstrap_estimates.append((t_sum/c_sum)-1)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates

## find bootstrapped confidence intervals for change in counts per 100 patients
def bootstrap_ci_cnts_per_100(df, num_bootstraps, col_name, group_col):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(df)), len(df))
        df_samples = df.loc[sample_index]
        c_rate = df_samples[df_samples[group_col]==0][col_name].mean()*100
        t_rate = df_samples[df_samples[group_col]==1][col_name].mean()*100
        bootstrap_estimates.append(t_rate-c_rate)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates

## % changes in number of admitting patients between test and control
def bootstrap_ci_binary(df, num_bootstraps, col_name, group_col, num_matches=3):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(df)), len(df))
        df_samples = df.loc[sample_index]
        c_sum = df_samples[df_samples[group_col]==0][col_name].sum()/num_matches
        t_sum = df_samples[df_samples[group_col]==1][col_name].sum()
        bootstrap_estimates.append((t_sum/c_sum)-1)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates

def bootstrap_ci_binary_100(df, num_bootstraps, col_name, group_col):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(df)), len(df))
        df_samples = df.loc[sample_index]
        c_sum = df_samples[df_samples[group_col]==0][col_name].mean() * 100
        t_sum = df_samples[df_samples[group_col]==1][col_name].mean() * 100
        bootstrap_estimates.append(t_sum - c_sum)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates

## Recycled Predictions inputs
## inputs are X matrix (ensure constant added as first columns), coefs (np.array of coefficients from model), column num of treatment, values (usually use 80% conf + mean)
def recycled_predictions_pois(model, X, coeffs, exp_col, values = [-0.05, 0, 0.05]):
    n = X.shape[0]
    ## Find values if no patients have exposure
    X = np.array(X)
    X[:,exp_col] = 0
    preds0 = np.sum(model.predict(coeffs, X))
    per_100_values = []
    X[:,exp_col] = 1
    for val in values:
        coeffs[exp_col] = val
        preds = np.sum(model.predict(coeffs, X))
        per_100_values.append(-(preds - preds0)/(n/100))
    return per_100_values


## Recycled Predictions inputs.
## inputs are X matrix (ensure constant added as first columns), Xmarg stands for 'X marginal', ie used for marginal effects.
## coefs (np.array of coefficients from model, add intercept to front), 
## column num of treatment, values (usually use 80% conf + mean)
def recycled_predictions(Xmarg, coefs, column_num, values=[-.1, 0, .1]):
    ## Find values if no patients have exposure
    Xmarg[:,column_num] = 0
    odds_no_msr = np.exp(np.dot(Xmarg, coefs))
    probas_no_msr = odds_no_msr/(1+odds_no_msr)
    exp_no_msr = probas_no_msr.sum()
    ## Find values if all patients have exposure
    per_100_values = []
    Xmarg[:,column_num] = 1
    for value in values:
        coefs[column_num] = value
        odds_msr = np.exp(np.dot(Xmarg, coefs))
        probas_msr = odds_msr/(1+odds_msr)
        exp_msr = probas_msr.sum()
        per_100_values.append((exp_msr - exp_no_msr)/(len(Xmarg)/100))
    print('For every 100 patients who are exposed, it is estimated that {1:0.2f} additional outcomes will occur. (CI {0:0.2f} to {2:0.2f})'.format(per_100_values[0], per_100_values[1], per_100_values[2]))


## Input: dataframe, an exposure variable, and an outcome.
## Returns: verbiage reporting the unadjusted odds, probability, and log odds of outcome
def raw_odds_and_probability(df, exposure, outcome):
    odds_exp = df[df[exposure]==1][outcome].mean() / (1 - df[df[exposure]==1][outcome].mean())
    odds_no_exp = df[df[exposure]==0][outcome].mean() / (1 - df[df[exposure]==0][outcome].mean())
    prob_exp = df[df[exposure]==1][outcome].mean()
    prob_no_exp = df[df[exposure]==0][outcome].mean()
    odds_inc = (odds_exp - odds_no_exp) / odds_no_exp
    prob_inc = (prob_exp - prob_no_exp) / prob_no_exp
    print('The odds of the outcome for the exposed and non-exposed groups are {0:.2f} and {1:.2f} respectively.'.format(odds_exp, odds_no_exp))
    print('The probability of the outcome for the exposed and non-exposed groups are {0:.2f}% and {1:.2f}% respectively.'.format(prob_exp*100, prob_no_exp*100))
    print('Exposure increases the odds of the outcome by {0:.2f}% and the probability of the outcome by {1:.2f}%.'.format(odds_inc*100, prob_inc*100))
    print('The unadjusted odds ratio is {0:.2f}.'.format(odds_exp / odds_no_exp))
    print('The unadjusted log odds is {0:.2f}.'.format(np.log(odds_exp / odds_no_exp)))

## Used to provide an example of how an example patient's odds
## of the outcome change when exposure changes, keeping all other
## variables constant.
## Inputs: logistic regression bootstrapped coefficients, 
## Input: avgp -- np.array of values that describe average patient
## Input: intercept value from model
## returns: verbiage describing how odds and probability change
def modeled_odds_proba_logistic(model_coeffs, avgp, x_cols, intercept, exposure_col):
    ## Find odds and probability of non_exposed patient
    lo_ne = intercept
    index_col = 0
    avgp[exposure_col] = 0
    for col in x_cols:
        lo_ne += (model_coeffs[index_col] * avgp[col])
        index_col +=1
    o_ne = np.exp(lo_ne)
    p_ne = o_ne / (1 + o_ne)
    avgp[exposure_col] = 1
    lo_e = intercept
    index_col = 0
    for col in x_cols:
        lo_e += (model_coeffs[index_col] * avgp[col])
        index_col +=1
    o_e = np.exp(lo_e)
    p_e = o_e / (1 + o_e)
    print('Non Exposed: log odds {0:0.2f}, odds {1:0.2f}, probability {2:0.2f}'.format(lo_ne, o_ne, p_ne))
    print('Exposed: log odds {0:0.2f}, odds {1:0.2f}, probability {2:0.2f}'.format(lo_e, o_e, p_e))

## Used to provide CIs for all variables in the model
## not just the exposure
## Inputs: coefficients (returned from bootstrapped ci)
## Inputs: confidence level

def model_ci_dataframe(mod_coefs, conf_lev):
    val_dict = {'lower': ((1-conf_lev)/2*100),
                'mean': 50,
                'upper': (((1-conf_lev)/2)+conf_lev)*100}
    cols = list(mod_coefs.columns)
    rslt_df = pd.DataFrame(index = list(aa_coefs.columns), columns = ['lower', 'mean', 'upper'])
    for col in cols:
        for val in ['lower', 'mean', 'upper']:
            rslt_df.loc[col, val] = round(np.percentile(mod_coefs[col], val_dict[val]), 2)
    return rslt_df


def case_control_static(df, event='event', stop='stop', exposure='HEART_FAILURE', risk_score='risk_score', hzn=180):
    results_df = pd.DataFrame({'days': np.arange(0, hzn+1, 1), 'control_exp': 0, 'control_no_exp': 0, 'case_exp': 0, 'case_no_exp': 0}, index=np.arange(0, hzn+1, 1))
    cases_count = 0  ## person time
    control_count = 0  ## person time
    case_exp = 0  ## counts cases that were exposed to toc
    case_no_ex = 0  ## counts cases that were not exposed to toc
    control_exp = 0  ## counts controls that were exposed to toc
    control_no_ex = 0  ## counts controls that were not exposed to toc

    concord_dict = {'concord_exp':0, 'concord_no_exp':0, 'discord_case_exp':0, 'discord_case_no_exp':0}

    for i in range(hzn+1):
        # print(i)

        cases = df[(df[stop] == i) & (df[event] == True)]
        controls = df[df[stop] > i]
        for index, row in cases.iterrows():
            ## find a control that has the closest risk score to this case
            cases_count += i
            if row[exposure] == 1:
                case_exp += 1
            else:
                case_no_ex += 1
            control = controls.iloc[(controls[risk_score] - row[risk_score]).abs().argsort()[:1]]
            # print(control)
            ## randomly choose 1 control from the top 50
            # control = pot_controls.sample(1)
            # print(int(control['TOC_2_DAY']))
            control_count += i
            ## find the TOC_2_D for that control
            myval = int(control[exposure])
            # print(myval)
            if myval == 1:
                control_exp += 1
            else:
                control_no_ex += 1
            # print(case_exp, myval)
            if row[exposure] == myval == 1:
                concord_dict['concord_exp'] += 1
            elif row[exposure] == myval == 0:
                concord_dict['concord_no_exp'] += 1
            elif (row[exposure] == 1) & (myval == 0):
                concord_dict['discord_case_exp'] += 1
            elif (row[exposure] == 0)& (myval == 1):
                concord_dict['discord_case_no_exp'] += 1
        ## Record in results_df values collected in this loop
        results_df.loc[i, 'control_exp'] = control_exp
        results_df.loc[i, 'control_no_exp'] = control_no_ex
        results_df.loc[i, 'case_exp'] = case_exp
        results_df.loc[i, 'case_no_exp'] = case_no_ex
        results_df['odds_ratio'] = (results_df['case_exp']/results_df['case_no_exp'])/(results_df['control_exp']/results_df['control_no_exp'])
    return results_df, concord_dict

## Peform a proportions z_test on a binary exposure 

def z_test(df, col, exposure):
    avg_cont = pd.crosstab(df[col], df[exposure])
    counts = np.array(avg_cont.loc[1,:])
    nobs = np.array(avg_cont.sum(axis=0))
    stat, pval = proportions_ztest(counts, nobs, alternative='smaller')
    return pval


##### Color, color blue and orange for Davita decks

## blue = ((0/255), (105/255), (177/255))
## orange = ((238/255), (128/255), 0)


def one_hot_selector(df, cat_col, target, thresh):
    ### creates new binary variables from a categorical variable for all levels that have relationship
    ### with target based on chi-square test
    values = list(df[cat_col].unique())
    one_hot_values = []
    for val in values:
        pval = st.chi2_contingency(st.contingency.crosstab(df[cat_col]==val, df[target]).count).pvalue
        if pval < thresh:
            one_hot_values.append(val)
            new_col_name = cat_col[:4] + '_' + val[0:10] 
            df[new_col_name] = np.where(df[cat_col]==1, 1, 0)
    other_val = cat_col[:4] + '_OTHER'
    df[other_val] = np.where(~df[cat_col].isin(one_hot_values), 1,0)
    df.drop(columns = cat_col, inplace=True)
    return df

def impute_eRAF(df, impute_cols):
    if df['eRAF'].isnull().sum() > 0:
        if is_sublist(impute_cols, df.columns):
            imputer = KNNImputer(n_neighbors=5)
            df[impute_cols] = imputer.fit_transform(df[impute_cols])
        else:
            print('Not all eRAF imputation vars are available.')
    else:
        print('No nulls in eRAF.')
    return df

def log_transform(df, cols_to_log):
    for col in cols_to_log:
        df[col] = np.log1p(df[col])
    return df

def scale_columns(df, cols_to_scale):
    scaler = StandardScaler()
    scaler.fit(df[cols_to_scale])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df

def plot_scaled_columns(df, cols_to_scale):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for m, ax in zip(cols_to_scale, axes.flatten()):
        ax.hist(df[m], bins=30)
        zero_bar_h = ax.get_ylim()[1]
        ax.plot([0, 0], [0, zero_bar_h], color='red', linestyle='-', linewidth=2)
        ax.set_title(m)
    plt.show()

def one_hot_select(df, cols_to_onehot, trtmnt_var, thresh):
    for col in cols_to_onehot:
        df = one_hot_selector(df, col, trtmnt_var, thresh)
    return df

def standard_processor(df, process_dict, thresh=.05):
    df = impute_eRAF(df, process_dict['impute_cols'])
    df = log_transform(df, process_dict['cols_to_log'])
    df = scale_columns(df, process_dict['cols_to_scale'])
    plot_scaled_columns(df, process_dict['cols_to_scale'])
    df = one_hot_select(df, process_dict['cols_to_onehot']
                            , process_dict['treatment_var'], thresh)
    return df

def is_sublist(list1, list2):
    return all(item in list2 for item in list1)

def generate_matched_df(pred_df, n_neigh = 30, trtmnt = 'TOC_2DAY_FLAG', with_replacement=True):
    caliper = (np.std(pred_df['propensity_score']) * 0.25)
    # setup knn
    knn = NearestNeighbors(n_neighbors=n_neigh, radius=caliper)
    ps_knn = pred_df[['propensity_score']]  # double brackets as a dataframe
    knn.fit(ps_knn)
    distances, neighbor_indexes = knn.kneighbors(ps_knn)
    matched_control = []  # keep track of the matched observations in control

    for current_index, row in pred_df.iterrows():  # iterate over the dataframe
        if row[trtmnt] == 0:  # the current row is in the control group
            pred_df.loc[current_index, 'matched'] = np.nan  # set matched to nan
        else: 
            for idx in neighbor_indexes[current_index, :]: # for each row in treatment, find the k neighbors
                # make sure the current row is not the idx - don't match to itself
                # and the neighbor is in the control 
                # if (current_index != idx) and (pscores.loc[idx].IRA == 0) and (pscores['ID'].str[:-7].loc[current_index] != pscores['ID'].str[:-7].loc[idx]):
                if (current_index != idx) and (pred_df.loc[idx][trtmnt] == 0): # and (pscores['EMPI_ID'].loc[current_index] != pscores['EMPI_ID'].loc[idx]):
                    if idx not in matched_control:
                        pred_df.loc[current_index, 'matched'] = idx  # record the matching
                        matched_control.append(idx)
                        break

    control = pred_df.iloc[matched_control,:].copy().reset_index(drop=True)
    test = pred_df[pred_df[trtmnt]==1].copy().reset_index(drop=True)

    return test, control


    ### After generating propensity scores using psmatch, use this to match patients

def random_idxmin(series):
    # Find the minimum value
    min_value = series.min()
    
    # Get all indices where the value is equal to the minimum
    min_indices = series[series == min_value].index
    
    # Randomly select one of these indices
    return np.random.choice(min_indices)

def match_patients(df, intervent = 'TOC_2DAY_FLAG', days_int = '2D_days'
    , time_to_outcome='duration', match_type = 'random', risk_set=True, score_col='ps_scores', n_matches=1):
    # Ensure 'TOC_2DAY_FLAG' is the treatment indicator
    treated = df[df[intervent] == 1].copy()
    untreated = df[df[intervent] == 0].copy()
    
    matched_data = []

    for _, treated_patient in treated.iterrows():
        # Find untreated patients who had not yet had an event
        if risk_set:
            eligible_untreated = untreated[
            (untreated[time_to_outcome] > treated_patient[days_int]) | 
            (untreated[time_to_outcome].isna())
            ].copy().reset_index(drop=True)
        else:
            eligible_untreated = untreated.copy()

        if match_type == 'score':
        
            if len(eligible_untreated) > 0:
                for i in range(n_matches):
                    # Calculate propensity score differences
                    eligible_untreated['ps_diff'] = abs(eligible_untreated[score_col] - treated_patient[score_col])
                    
                    ## Find the index of a random eligible with a tie for the minimum distance
                    # rand_min_index = random_idxmin(eligible_untreated['ps_diff']

                    # Find the untreated patient with the minimum propensity score difference
                    best_match = eligible_untreated.loc[eligible_untreated['ps_diff'].idxmin()]

                           # Add matched untreated patient data
                    untreated_data = best_match.to_dict()
                    # untreated_data['match_id'] = treated_patient['id']
                    # untreated_data['match_type'] = 'untreated'
                    matched_data.append(untreated_data)


        elif match_type == 'random':
            for i in range(n_matches):
                best_match = eligible_untreated.loc[random.randint(0, len(eligible_untreated)-1)]
                # Add matched untreated patient data
                untreated_data = best_match.to_dict()
                # untreated_data['match_id'] = treated_patient['id']
                # untreated_data['match_type'] = 'untreated'
                matched_data.append(untreated_data)

        else:
            print('Need to provide risk_type argument: random or score.')
                
        # Add treated patient data
        treated_data = treated_patient.to_dict()
        # treated_data['match_id'] = best_match['id']
        # treated_data['match_type'] = 'treated'
        matched_data.append(treated_data)
    
    # Create dataframe from matched data
    result_df = pd.DataFrame(matched_data)
    
    # Ensure all original columns are present, add any missing ones
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    # Reorder columns to match original dataframe
    result_df = result_df[df.columns.tolist()] ##+ ['match_id', 'match_type']]
    
    return result_df


def update_toc_results_dataframe(df, time, lob, toc, model, horizon, N, effect, eff_low, eff_high, per_100, per_100_low,
                       per_100_high, today, ua_rp_toc, ua_ar_toc, ua_rp_nontoc,
                        ua_ar_nontoc, a_rp_toc=None, a_ar_toc=None, a_rp_nontoc=None, a_ar_nontoc=None):
    # Create a boolean mask to find the matching row
    mask = (df['time'] == time) & \
           (df['LOB'] == lob) & \
           (df['TOC'] == toc) & \
           (df['model'] == model) & \
           (df['horizon'] == horizon)
    
    # Check if we found exactly one matching row
    if mask.sum() != 1:
        print(f"Error: Found {mask.sum()} matching rows. Expected 1.")
        return

    # Update the values in the matching row
    df.loc[mask, 'N'] = N
    df.loc[mask, 'effect'] = effect
    df.loc[mask, 'eff_low'] = eff_low
    df.loc[mask, 'eff_high'] = eff_high
    df.loc[mask, 'per_100'] = per_100
    df.loc[mask, 'per_100_low'] = per_100_low
    df.loc[mask, 'per_100_high'] = per_100_high
    df.loc[mask, 'today'] = today
    df.loc[mask, 'ua_rp_toc'] = ua_rp_toc
    df.loc[mask, 'ua_ar_toc'] = ua_ar_toc
    df.loc[mask, 'ua_rp_nontoc'] = ua_rp_nontoc
    df.loc[mask, 'ua_ar_nontoc'] = ua_ar_nontoc
    df.loc[mask, 'a_rp_toc'] = a_rp_toc
    df.loc[mask, 'a_ar_toc'] = a_ar_toc
    df.loc[mask, 'a_rp_nontoc'] = a_rp_nontoc
    df.loc[mask, 'a_ar_nontoc'] = a_ar_nontoc
    
    print("DataFrame updated successfully.")





def risk_set_matching(df, id_col, treatment_day_col, ever_treated_col, readmission_day_col, n_matches=1):


    df_sorted = df.sort_values(treatment_day_col, reset_index=True)
    treated = df_sorted[df_sorted[ever_treated_col] == 1].copy()
    untreated = df_sorted[df_sorted[ever_treated_col] == 0].copy()
    
    matched_rows = []
    
    for _, treated_patient in treated.iterrows():
        treatment_day = treated_patient[treatment_day_col]

        eligible_untreated = untreated[
            (untreated[readmission_day_col] > treated_patient[treatment_day]) | 
            (untreated[readmission_day_col].isna())
            ].copy()
        
        
        if len(eligible_untreated) >= n_matches:
            # Randomly select n matches
            matches = eligible_untreated.sample(n=n_matches, random_state=42)
            
            # Add treated patient and matches to the result
            matched_rows.append(treated_patient.to_dict())
            matched_rows.extend(matches.to_dict('records'))
        else:
            print(f"Warning: Not enough matches for patient {treated_patient[id_col]} on day {treatment_day}")
    
    # If matched_rows is empty, return an empty DataFrame with the same columns as the input
    if not matched_rows:
        return pd.DataFrame(columns=df.columns)
    
    # Create DataFrame from matched_rows and ensure all columns from original DataFrame are present
    result_df = pd.DataFrame(matched_rows)
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    return result_df[df.columns]


def get_date_before_quarter(quarter_string, months_before):
    # Parse the year and quarter
    year = int(quarter_string[:4])
    quarter = int(quarter_string[5])
    
    # Calculate the first month of the quarter
    first_month = (quarter - 1) * 3 + 1
    
    # Create a date object for the first day of the quarter
    quarter_start = datetime(year, first_month, 1)
    
    # Subtract the specified number of months
    result_date = quarter_start - relativedelta(months=months_before)
    
    return result_date

def get_last_date_of_quarter(quarter_string):
    # Parse the year and quarter
    year = int(quarter_string[:4])
    quarter = int(quarter_string[5])
    
    # Calculate the first month of the quarter
    first_month = (quarter - 1) * 3 + 1
    
    # Create a date object for the first day of the quarter
    quarter_start = datetime(year, first_month, 1)
    
    # Subtract the specified number of months
    result_date = quarter_start + relativedelta(months=3)
    
    return result_date
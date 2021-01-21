## helper functions for Triage model

import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import math

############################### DATA ####################################

def pickle_dump(obj_to_pkl, filepath):
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

def plot_bs_coefs(bs_coefs, col_names, nbins=10, imagepath='../images/myimage.jpg'):
    ax_rows = math.ceil(bs_coefs.shape[1]/3)
    plt_h = ax_rows * 4
    fig, axes = plt.subplots(ax_rows,3, figsize=(14,plt_h))
    for m, ax in zip(col_names, axes.flatten()):
        ax.hist(bs_coefs[m], bins=nbins)
        zero_bar_h = ax.get_ylim()[1]
        ax.plot([0, 0], [0, zero_bar_h], color='red', linestyle='-', linewidth=2)
        ax.set_title(m)
    if os.path.exists(imagepath):
        print('Image already exists. Overwrite (y/n)?')
        x = input()
    else: 
        x = 'y'
    if x == 'y':
        fig.savefig(imagepath, dpi=300, optimize=True, bbox_inches='tight', pad_inches=.1)

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

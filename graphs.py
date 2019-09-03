import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from scipy import optimize

data_file = r"/Users/shahrooz/Google Drive/ucl/thesis/data/15min_bins_simulated.csv"
df = pd.read_csv(data_file) #, parse_dates=['BIN_STARTS'] )
df['BIN_STARTS'] = pd.to_datetime(df['BIN_STARTS'])

path = r"/Users/shahrooz/Google Drive/ucl/thesis/data/"

dates = np.array(df['BIN_STARTS'])
counts = np.array(df['COUNTS'])

def date_count_plot(dates,counts,formatter = '%d-%m',x_label="Bin Dates",title="Count Distribution for RFQ Data"):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    #ax.set_facecolor('xkcd:silver')
    plt.plot(dates, counts,color="blue")
    if formatter:
        ax.xaxis.set_major_formatter(mdates.DateFormatter(formatter))

    plt.xlabel(x_label)
    plt.ylabel("RFQ Counts")
    #plt.title(title)
    plt.savefig(path + title + ".pdf")
    plt.show()

date_count_plot(dates,counts)

# day 1 is the first 95 records
day_1_dates = dates[0:95]
day_1_counts=counts[0:95]
date_count_plot(day_1_dates,day_1_counts,'%H:%M',x_label="Bin times", title="Count Distribution for RFQ Data for One Day")

#histogram_plot(day_1_counts, title= "Histogram Plot of RFQ Count Frequency for One Day")

# Make histogram data directly
import pandas
count_series = pandas.Series(counts)
hist_counts = count_series.value_counts().sort_index()
hist_dict = hist_counts.to_dict()
counts_list = np.array(list(hist_dict.keys())[1:]) #1 : is to ignore the first term (zero)
frequencies_list = np.array(list(hist_dict.values())[1:]) #1 is to ignore the first term (zero)



# POWER LAW FUNCTION
powerlaw = lambda x, amp, index: amp * (x ** index)


def power_law(xdata = counts_list, ydata = frequencies_list):
    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    # Note that the `logyerr` term here is ignoring a constant prefactor.
    #
    #  y = a * x^b
    #  log(y) = log(a) + b*log(x)
    #

    # Define function for calculating a power law
    yerr = 0.1

    logx = np.log10(xdata)
    logy = np.log10(ydata)
    logyerr = yerr / ydata

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]

    index = pfinal[1]
    amp = 10.0 ** pfinal[0]

    #indexErr = np.sqrt(covar[1][1])
#)##ampErr = np.sqrt(covar[0][0]) * amp

    ##########
    # Plotting data
    ##########

    plt.clf()
    #plt.subplot(1, 1, 1)
    plt.plot(xdata, ydata, color = "blue", label="Data") #yerr=yerr, fmt='k.')  # Data
    plt.plot(xdata, powerlaw(xdata, amp, index), color = "red", label="Power Law Prediction")  # Fit
    #plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    #plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
#    plt.title('Best Fit Power Law')
    plt.xlabel('Non-zero RFQ Counts')
    plt.ylabel('Frequency')
    plt.legend()
    plt.xlim(1, 135)

    # POWER LAW PLOT AGAINST FREQUENCY COUNTS
    plt.savefig(path + "Power law.pdf")
    plt.show()

    return amp, index

amp, index = power_law()


print("POWER LAW PARAMETERS FROM BEST FIT:")
print("AMPLITUDE:", amp)
print("POWER: ", index)
#y = amp * x ^ index


xdata = counts_list

def histogram_plot(counts, title="Histogram Plot of RFQ Count Frequency"):
    plt.hist(counts, color="blue")
    #plt.plot(xdata, powerlaw(xdata, amp, index))
    plt.xlabel("RFQ Counts")
    plt.ylabel("Frequency")
    #plt.title(title)
    plt.savefig(path + title + ".pdf")
    plt.show()


histogram_plot(counts)



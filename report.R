# Title: Time series forecasting
# Date: 12/20/2020
# Output: pdf_document

library(forecast)
library(ggplot2)
library(openxlsx)

dataset <- read.xlsx('elec-train.xlsx')
head(dataset, 25)

# Power is measured once every 15 minutes, temperature is measured once per hour at most (sometimes less often).

tail(dataset)

# The last 96 power values are missing, these are the values we're going to forecast (with and without covariant temperatures).
# Let's plot the power data as a time series.

dataset_power <- ts(dataset$`Power.(kW)`)
autoplot(dataset_power)

# It's difficult to see a long term trend here although we may expect a month-to-month evolution.
# On the other hand we can definitely observe a cyclic pattern so let's visualize the data on a smaller time range.

dataset_power_head <- head(dataset_power,200)
autoplot(dataset_power_head)

# Cycles seem to fit 96 timeslots (i.e. 1 day = 24h * 4 * 15min) which makes sense, so let's focus on the seasonal pattern after extracting the power data as a time series of frequency 96.

power <- ts(dataset$`Power.(kW)`, frequency=96)
ggseasonplot(power)

# The daily pattern is confirmed with low power consumption at nighttime, high consumption at daytime and a surge in the evening. This is typical of household power consumption although we may expect a consumption surge early in the morning too.
# Now let's separate train and test datasets in order to build and assess our future models.

power_train <- window(power, start=c(1,1), end=c(46,91))
power_test <- window(power, start=c(46,92), end=c(47,91))
print(power_test)


# Forecasting power without temperature

# Our first model is based on the Holt-Winters function. After playing with the alpha (smoothing factor), beta (double exponential smoothing) and gamma (seasonal component) values, we stop at alpha=0.00002, beta=1 and gamma=0.2.

hw_fit <- HoltWinters(power_train, alpha=0.00002, beta=1, gamma=0.2)
hw_prev <- forecast(hw_fit, h=96)
autoplot(power_test, series='data') + autolayer(hw_prev$mean, series='hw')
cat('RMSE:', sqrt(mean((hw_prev$mean-power_test)^2)))

# Our second model is based on automatic ARIMA. Even though the result is coherent, RMSE is worse than Holt-Winters because automatic parameters fail to catch the high noise level between the 15 minutes slots.

auto_arima_fit <- auto.arima(power_train, lambda='auto')
auto_arima_prev <- forecast(auto_arima_fit, h=96)
print(auto_arima_fit)
autoplot(power_test, series='data') + autolayer(auto_arima_prev$mean, series='auto_arima')
cat('RMSE:', sqrt(mean((auto_arima_prev$mean-power_test)^2)))

# We try to improve ARIMA by handpicking parameters. In order to select good parameters we first remove the seasonal pattern from the dataset by differentiating with a lag of 96.

diff1 <- diff(power_train, lag=96)
ggtsdisplay(diff1)

# From the ACF plot we observe a clear trend which we'll remove by differentiating again.

diff2 <- diff(diff1)
checkresiduals(diff2)

# Although residuals are now correctly balanced and most of autocorrelation is included in the +/-5% range, we still observe a seasonal peak at lag 96 plus some bumps at lags 1 and 4. We use these values for our ARIMA model which results in a restrained autocorrelation and a RMSE similar to Holt-Winters.

manual_arima_fit <- Arima(power_train, order=c(1,1,4), seasonal=c(0,1,1))
manual_arima_prev <- forecast(manual_arima_fit, h=96)
checkresiduals(manual_arima_fit)
autoplot(power_test, series='data') + autolayer(manual_arima_prev$mean, series='manual_arima')
cat('RMSE:', sqrt(mean((manual_arima_prev$mean-power_test)^2)))

# Our third model leverages the NNAR feed-forward neural network. Automatic model seems acceptable but less precise than the previous one.

auto_nnar_fit <- nnetar(power_train, lambda='auto')
auto_nnar_prev <- forecast(auto_nnar_fit, h=96)
print(auto_nnar_fit)
autoplot(power_test, series='data') + autolayer(auto_nnar_prev, series='auto_nnar')
cat('RMSE:', sqrt(mean((auto_nnar_prev$mean-power_test)^2)))

# We try to improve the NNAR model by changing the p (non-seasonal lags), P (seasonal lags) and size (nodes in the hidden layer) parameters. We increase values until the maximum number of parameters is reached, resulting in a RMSE similar to HW and ARIMA. Lambda is still set automatically.

manual_nnar_fit <- nnetar(power_train, 40, 4, 20, lambda='auto')
manual_nnar_prev <- forecast(manual_nnar_fit, h=96)
autoplot(power_test, series='data') + autolayer(manual_nnar_prev, series='manual_nnar')
cat('RMSE:', sqrt(mean((manual_nnar_prev$mean-power_test)^2)))

# Our three final Holt-Winters, ARIMA and NNAR models have very close RMSE values. However, Holt-Winters is faster and may be more resistant to overfitting so we use it in order to forecast the missing day with the full power dataset.

power_full <- window(power, start=c(1,1), end=c(47,91))
hw_full <- HoltWinters(power_full, alpha=0.00002, beta=1, gamma=0.2)
F02172010_without_temperature <- forecast(hw_full, h=96)
autoplot(tail(power_full, 200), series='data') + autolayer(F02172010_without_temperature, series='hw')
print(F02172010_without_temperature$mean)


# Forecasting power with temperature

temperature <- ts(dataset$`Temp.(C°)`, frequency=96)
temperature_train <- window(temperature, start=c(1,1), end=c(46,91))
temperature_test <- window(temperature, start=c(46,92), end=c(47,91))
print(temperature_test)

# Holt-Winters doesn't take external regressors into account so we can't use it with temperature.
# We use the TSLM linear regression method to build a power consumption model based on temperature. As expected there is a correlation between the 2 components (pointed by the low p-value) but temperature is definitely not enough to predict the power consumption. Even after including trend and season components in TSLM we still observe some trend and seasonal patterns in the residuals, so we can't really use TSLM to forecast.

tslm_temp <- tslm(power_train~temperature_train+trend)
summary(tslm_temp)
checkresiduals(tslm_temp)

# Differentiating the TSLM residuals points at the same parameters used for ARIMA without temperature, so we use the same ARIMA model as before but this time we add the xreg element then check ACF and residuals again to make sure the main components are included.

ggtsdisplay(diff(diff(residuals(tslm_temp), lag=96)))
manual_arima_temp_fit <- Arima(power_train, order=c(1,1,4), seasonal=c(0,1,1), xreg=temperature_train)
checkresiduals(manual_arima_temp_fit)

# It results in a very similar ARIMA model as the one without temperature, so temperature doesn't seem to help.

manual_arima_temp_prev <- forecast(manual_arima_temp_fit, h=96, xreg=temperature_test)
autoplot(power_test, series='data') + autolayer(manual_arima_temp_prev$mean, series='manual_arima_temp')
cat('RMSE:', sqrt(mean((manual_arima_temp_prev$mean-power_test)^2)))

# We do the same for the neural network model: we use the same NNAR parameters as before and add the temperature external regressor. This results in a slightly improved model, although RMSE are still very close.

manual_nnar_temp_fit <- nnetar(power_train, 40, 4, 20, lambda='auto', xreg=temperature_train)
manual_nnar_temp_prev <- forecast(manual_nnar_temp_fit, h=96, xreg=temperature_test)
autoplot(power_test, series='data') + autolayer(manual_nnar_temp_prev, series='manual_nnar_temp')
cat('RMSE:', sqrt(mean((manual_nnar_temp_prev$mean-power_test)^2)))

# Finally, in order to forecast power with temperature, we select NNAR over ARIMA because NNAR is faster and seems to use temperature a little. This may have not been our choice if we had to forecast on a longer time range as our NNAR model may be prone to overfitting due to so many parameters.

temperature_full <- window(temperature, start=c(1,1), end=c(47,91))
temperature_try <- window(temperature, start=c(47,92), end=c(48,91))
manual_nnar_temp_full <- nnetar(power_full, 40, 4, 20, lambda='auto', xreg=temperature_full)
F02172010_with_temperature <- forecast(manual_nnar_temp_full, h=96, xreg=temperature_try)
autoplot(tail(power_full, 200), series='data') + autolayer(F02172010_with_temperature, series='manual_nnar_temp')
print(F02172010_with_temperature$mean)

# Let's write our final results in a file.

write.xlsx(cbind(F02172010_without_temperature$mean,F02172010_with_temperature$mean),'prediction.xlsx')

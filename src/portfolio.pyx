import re
import random
import urllib2
import math
import numpy as np
import scipy.interpolate as ip
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import scipy.optimize as sco
import time
import datetime
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    AdaptiveETA, FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer




class portfolio:
	data = [] # storing stock prices for the companies in the portfolio
	gspc = [] # storing S&P500 index prices for the given period
	returns = []
	start_prices = []
	end_prices = []
	
	min_return = 0.0 # minimal return value 
	#max_return = 0.5 # maximum return value
	step = 50 
	threshold = 0.01 # the companies with weights below this value are not included in portfolio
	
	# Constructor
	def __init__(self, _symbols, _start, _end, _return, _weights = None):
		# Initializing data
		self.data = pd.DataFrame()
		self.gspc = pd.DataFrame()
		self.num_of_days = 0
		self.feasible_returns = []
		self.feasible_volatilities = []
		self.feasible_weights = []
	
		self.effective_volatilities = []
		self.effective_returns = []
		self.effective_weights = []
		
		self.symbols = []
		self.sharpe_symbols = []
		self.min_variance_symbols = []
		
		# 
		self.start = _start
		self.end = _end
		self.max_risk = 0
		self.max_return = _return

		self.fetch(_symbols)
		self.set_returns() 
		self.set_start_end_prices()
		
		if _weights != None:
			self.fetch_sp500()
			self.test(_weights)
			self.show_sp500()
	
	# Get percent-view of an number (-24.5% instead of -0.245)
	def percent(self, i):
		formatted = str(i.round(3) * 100) + '%'
		if i > 0:
			formatted = '+' + formatted
		return formatted
	
	#
	def set_days(self):
		for sym in self.symbols:
			if len(self.data[sym]) == self.num_of_days:
				start = str( re.split(' ', str(self.data[sym].axes[0][0]))[0])
				end = str( re.split(' ', str(self.data[sym].axes[0][-1]))[0])
				if self.start != start or self.end != end:
					self.start = start
					self.end = end
					print 'Investment period adjusted to ' + self.start + ' - ' + self.end
				return
			
		print 'Error setting investment period'
		return            
	
	# Вычисление логарифмически нормализованых доходностей (отношения уровня закрытия текущего дня к предыдущему) за исторический период   
	def set_returns(self):
		self.returns = np.log(self.data / self.data.shift(1))
	
	# 
	def set_start_end_prices(self):
		self.start_prices = []
		self.end_prices = []
		for sym in self.symbols:
			self.start_prices.append(self.data[sym][self.start])
			self.end_prices.append(self.data[sym][self.end])
	
	# Retrieving stock prices
	def fetch(self, symbols):
		for sym in symbols:
			try:
				self.data[sym]  = web.DataReader(sym, data_source='yahoo', start=self.start, end=self.end)['Adj Close']
				self.symbols.append(sym)   
				if self.data[sym].size > self.num_of_days:
					self.num_of_days = self.data[sym].size
			except IOError:
				print sym + ': no data available for requested period'
				continue
				
		self.data.column = self.symbols
		self.set_days()
		print 'Data fetched for ' + str(len(self.symbols)) + ' symbols and '+ str(self.num_of_days)  + ' trading days'
		return
	
	# Retrieving data for S&P500 index:
	def fetch_sp500(self):
		self.gspc = web.DataReader('^GSPC', data_source='yahoo', start=self.start, end=self.end)['Adj Close']
		self.gspc.column = '^GSPC'
		
	#
	def show(self, weights):
		print "Investment period: " + self.start + " - " + self.end
		print "Risk limit: " + str(self.max_risk)
		
		print "Portfolio volatility: " + str( self.statistics(weights)[0].round(3) )
		print "Portfolio return: " + str( self.statistics(weights)[1].round(3) )   
		print "Sharpe ratio: "  + str( self.statistics(weights)[2].round(3) )  
		
		print "Portfolio weights:"    
		for sym, w in zip(self.symbols, weights):
			if w > self.threshold:
				print sym + ': ' + str(w.round(3))
	
	# Show the line with symbol statistics
	# AAPL: 67.802 -> 72.73 (+7.3%)
	def show_symbol(self, sym, weight, start_price, end_price, return_rate):
		if weight == None:
			print sym + ': ' + str(start_price.round(3)) + ' -> ' + str(end_price.round(3)) + ' (' + self.percent(return_rate) + ')'
		else:
			print sym + ' (' + str(weight) + '): ' + str(start_price.round(3)) + ' -> ' + str(end_price.round(3)) + ' (' + self.percent(return_rate) + ')'
	#
	def show_sp500(self):
		start = self.gspc[self.start]
		end = self.gspc[self.end]
		return_rate = (end - start) / start
		self.show_symbol('S&P500', None, start, end, return_rate)

	#
	def test(self, weights):     
		portfolio_start = 0
		portfolio_end = 0
		
		for s, sp, ep, w in zip(self.symbols, self.start_prices, self.end_prices, weights):
			ret = (ep - sp) / sp  
			portfolio_start = portfolio_start + sp * w
			portfolio_end = portfolio_end + ep * w
			self.show_symbol(s, w, sp, ep, ret)
			
		portfolio_return = (portfolio_end - portfolio_start) / portfolio_start
		print '\nPortfolio return: ' + self.percent(portfolio_return)
		
		return
	
	def show_time(self, start_time):
		seconds = (time.time() - start_time)
		print ('Wall time: %s' % str(datetime.timedelta(seconds=seconds)))

	#
	def calc(self):
		start_time = time.time()
		self.get_effective_set()
		print '\nThe highest Sharpe ratio portfolio: '
		self.show( self.get_max_sharpe_weights() )
		
		print '\nThe minimal variance porfolio: '
		self.show( self.get_min_variance_weights() )
		self.show_time(start_time)
		
	
	# 
	def get_risky(self):
		if len(self.sharpe_symbols) > 0:
			return self.sharpe_symbols, self.sharpe_weights
		else:
			print 'The optimal portfolio is not calculated. Run calc() first'
			
	#
	def get_conservative(self):
		if len(self.min_variance_symbols) > 0:
			return self.min_variance_symbols, self.min_variance_weights
		else:
			print 'The optimal portfolio is not calculated. Run calc() first'
		return
	
	# Вычисление эффективного множества (effective set)
	def get_effective_set(self):
		i = 0
		self.effective_volatilities = []
		self.effective_returns = []
		self.effective_weights = []
	
		cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
		bnds = tuple((0, 1) for x in range(len(self.symbols)))

		pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=self.max_return).start()

		for y in np.linspace(self.min_return, self.max_return, self.step):
			cons = ({'type': 'eq', 'fun': lambda x: self.statistics(x)[1] - y}, {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
			result = sco.minimize(self.min_volatility, len(self.symbols) * [1. / len(self.symbols),], method='SLSQP', bounds=bnds, constraints=cons)
			self.effective_volatilities.append(result['fun'])
			self.effective_returns.append(y)
			self.effective_weights.append(result['x'])
			pbar.update(y)
	
		pbar.finish()
		self.effective_volatilities = np.array(self.effective_volatilities)
		self.effective_returns = np.array(self.effective_returns)
		self.effective_weights = np.array(self.effective_weights)

	# Функция, получающая веса бумаг в портфеле в качестве входных параметров, и возвращающая массив 
	# данных о портфеле в формате [волатильность, доходность, коэффициент Шарпа]
	def statistics(self, weights):
		weights = np.array(weights)
		portfolio_return = self.get_portfolio_return(weights)
		portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.returns.cov() * self.num_of_days, weights)))
	
		return np.array([portfolio_volatility, portfolio_return, portfolio_return / portfolio_volatility])
	
	# Функция вычисления минимального отклонения
	def min_volatility(self, weights):
		return self.statistics(weights)[0]
	
	# Функция вычисления доходности портфеля
	def get_portfolio_return(self, weights):
		portfolio_start_price = 0
		portfolio_end_price = 0

		for w, sp, ep in zip(weights, self.start_prices, self.end_prices):
			portfolio_start_price = portfolio_start_price + w * sp
			portfolio_end_price = portfolio_end_price + w * ep
		
		try:
			return ((portfolio_end_price - portfolio_start_price) / portfolio_start_price)
		except ZeroDivisionError:
			print 'Zero division exception'
			print weights
			return 0
		#return np.sum(returns.mean() * weights) * num_of_days
	
	# Функция вычисления портфеля с максимальным коэффицентом Шарпа (отношением доходность/волатильность)
	def get_max_sharpe_weights(self):
		max_sharpe = 0
		weights = []
		for r, v, w in zip(self.effective_returns, self.effective_volatilities, self.effective_weights):
			if r / v > max_sharpe:
				max_sharpe =  r / v
				weights = w
				
		self.sharpe_symbols = []
		self.sharpe_weights = []
		for sym, w in zip(self.symbols, weights):
			if w > self.threshold:
				self.sharpe_symbols.append(sym)
				self.sharpe_weights.append(w.round(3)) # FIXME: shoudn't round value maybe
		
		return weights
	
	#
	def get_min_variance_weights(self):
		min_variance = 10
		weights = []
		for v, w in zip(self.effective_volatilities, self.effective_weights):
			if v < min_variance:
				min_variance =  v
				weights = w
				
		self.min_variance_symbols = []
		self.min_variance_weights = []
		for sym, w in zip(self.symbols, weights):
			if w.round(3) > 0.001:
				self.min_variance_symbols.append(sym)
				self.min_variance_weights.append(w.round(3)) # FIXME: shoudn't round value maybe

		return weights
	
	def plot(self):
		plt.figure(figsize=(12, 6))

		x_sharpe = self.statistics(self.get_max_sharpe_weights())[0]
		y_sharpe = self.statistics(self.get_max_sharpe_weights())[1]

		x_min_vol = self.statistics(self.get_min_variance_weights())[0]
		y_min_vol = self.statistics(self.get_min_variance_weights())[1]

		#plt.scatter(feasible_volatilities, feasible_returns, c = feasible_returns / feasible_volatilities, marker='o')
		plt.scatter(self.effective_volatilities, self.effective_returns, c = self.effective_returns / self.effective_volatilities, marker='x')
		plt.plot(x_sharpe, y_sharpe, 'rs', markersize=8.0) # portfolio with highest Sharpe ratio
		plt.plot(x_min_vol, y_min_vol, 'ys', markersize=8.0) # portfolio with highest Sharpe ratio

		plt.colorbar(label='Sharpe ratio')
		plt.xlabel(r'$\sigma_p$')
		plt.ylabel(r'$\bar{r_p}$')
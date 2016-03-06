import urllib2
import random
from bs4 import BeautifulSoup

class sp_500():
	sector_tickers = {}
	symbols = []
	
	def __init__(self):
		self.symbols = []
		self.sector_tickers = dict()
		
		site = "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
		hdr = {'User-Agent': 'Mozilla/5.0'}
		req = urllib2.Request(site, headers=hdr)
		page = urllib2.urlopen(req)
		soup = BeautifulSoup(page, "lxml")

		table = soup.find('table', {'class': 'wikitable sortable'})
	
		for row in table.findAll('tr'):
			col = row.findAll('td')
			if len(col) > 0:
				sector = str(col[3].string.strip()).lower().replace(' ', '_')
				ticker = str(col[0].string.strip())
				date = str(col[6]).replace('<td>', '').replace('</td>', '')
				#if date < historical_end_date:
				if sector not in self.sector_tickers:
					self.sector_tickers[sector] = list()
				self.sector_tickers[sector].append(ticker)
				self.symbols.append(ticker)
		
	   
	# Show sectors summary
	def sectors(self):
		for s in self.sector_tickers:
			print s + ': ' + str(len(self.sector_tickers[s])) + ' companies'
			
	# Show the sector summary and companies in it
	def show(self, sector = None):
		if sector == None:
			self.sectors()
		else:
			print sector + ' sector of S&P500\n' + str(len(self.sector_tickers[sector])) + ' companies:'
			print self.sector_tickers[sector]
	
	# Get the company symbols
	def get(self, sector = None):
		symbols = []
		if sector == None:
			# Get the companies from all S&P500 sectors
			for s in self.sector_tickers:
				for sym in self.sector_tickers[s]:
					symbols.append(sym)
		else:
			# Get the companies from the given sector
			for sym in self.sector_tickers[sector]:
				symbols.append(sym)
		return symbols
	
	# Get random number of stocks
	def random(self, num):
		return random.sample(self.symbols, num)

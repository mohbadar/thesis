#!/usr/bin/python
import time
from sp500 import sp_500
from portfolio import portfolio

sp500 = sp_500()
max_risk = 0.2

p = portfolio(sp500.get(), '2013-01-01', '2015-01-01', 0.18)
#p = portfolio(sp500.random(100), '2013-01-01', '2015-01-01', max_risk)
p.calc()

print "\nPortfolio with given risk limit (" + str(max_risk) + "):"
r = portfolio(p.get_acceptable()[0], '2015-01-01', '2016-01-01', None, p.get_acceptable()[1])

print "\nRisky portfolio:"
r = portfolio(p.get_risky()[0], '2015-01-01', '2016-01-01', None, p.get_risky()[1])

print "\nConservative portfolio:"
c = portfolio(p.get_conservative()[0], '2015-01-01', '2016-01-01', None, p.get_conservative()[1])

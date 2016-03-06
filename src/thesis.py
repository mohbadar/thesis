#!/usr/bin/python
import time
from sp500 import sp_500
from portfolio import portfolio

sp500 = sp_500()

#p = portfolio(sp500.get('consumer_discretionary'), '2013-01-01', '2015-01-01', 0.25, 37)
p = portfolio(sp500.random(60), '2013-01-01', '2015-01-01', 0.2)
p.calc()

print "\nRisky portfolio:"
r = portfolio(p.get_risky()[0], '2015-01-01', '2016-01-01', None, p.get_risky()[1])

print "\nConservative portfolio:"
c = portfolio(p.get_conservative()[0], '2015-01-01', '2016-01-01', None, p.get_conservative()[1])

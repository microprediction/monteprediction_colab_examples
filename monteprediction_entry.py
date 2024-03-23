import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from monteprediction import SPDR_ETFS
from monteprediction.calendarutil import get_last_wednesday
from monteprediction.submission import send_in_chunks

#################################################
#   Example entry for www.monteprediction.com   #
#################################################

# For explanation see the notebook: https://colab.research.google.com/github/microprediction/monteprediction_colab_examples/blob/main/monteprediction_entry.ipynb
# After modifying this script, run it every weekend on a cron job. 

# Factory defaults (don't modify)
num_samples_per_chunk = int(1048576/8)
num_chunks = 8
num_samples = num_chunks*num_samples_per_chunk

# This example uses Quasi-Monte Carlo on the empirical covariance
# There is absolutely no requirement you follow this pattern

from scipy.stats.qmc import MultivariateNormalQMC
from sklearn.covariance import EmpiricalCovariance

# Get historical weekly returns
last_wednesday = get_last_wednesday()
num_weeks = int(52+4*52*np.random.rand())
start_date = last_wednesday - timedelta(weeks=num_weeks)
data = yf.download(SPDR_ETFS, start=start_date, end=last_wednesday, interval="1wk")
weekly_prices = data['Adj Close']
weekly_returns = weekly_prices.pct_change().dropna()

# Use cov estimation to generate samples
cov_matrix = EmpiricalCovariance().fit(weekly_returns).covariance_
qmc_engine = MultivariateNormalQMC(mean=np.zeros(len(SPDR_ETFS)), cov=cov_matrix)
samples = qmc_engine.random(num_samples)
df = pd.DataFrame(columns=SPDR_ETFS, data = samples)
print(df[:3])

# Verify submission
assert len(df.index)==num_samples,f'Expecting exactly {num_samples} samples'
assert list(df.columns)==SPDR_ETFS,'Columns should match SPDR_ETFS in order'

YOUR_EMAIL = 'totally_not_valid_email_at_nowhere'  # Be sure to change this to real email
YOUR_NAME = 'Forgot to set name'
send_in_chunks(df, num_chunks=num_chunks, email=YOUR_EMAIL, name=YOUR_NAME)

# Don't forget to run this every weekend! 

import os
import requests
import pandas

'''
Simple loader/API wrapper for Tiingo (https://www.tiingo.com/) to load crypto returns data in the form of pandas dataframes/CSVs.
This requires an API key, which one can get free via signing up and pass through as an environment variable (TIINGO_API_KEY).
'''
class CryptoDataLoader:
    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")

    def get_pair_adjusted_returns(self, pair, start_date='2019-01-02', frequency='1day'):
        if not self.api_key:
            raise Exception("Please set a value for TIINGO_API_KEY to use this.")
        headers = {
            'Content-Type': 'application/json'
        }
        base_url = "https://api.tiingo.com/tiingo/crypto/prices"
        combined_url = base_url + "?tickers=" + pair + "&startDate=" + start_date + "&resampleFreq=" + frequency + "&token=" + self.api_key
        requestResponse = requests.get(combined_url, headers=headers)
        if requestResponse.status_code > 300:
            raise Exception("Failed to retrieve data due to: " + requestResponse.text)
        price_data = requestResponse.json()[0]['priceData']
        outdir = './fetched_data'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fullname = os.path.join(outdir, pair + ".csv")
        pandas.DataFrame(price_data).to_csv(fullname)


CryptoDataLoader().get_pair_adjusted_returns("btcusd")


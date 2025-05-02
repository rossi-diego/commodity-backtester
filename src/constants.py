tickers = {'ZS=F', 'ZC=F', 'ZW=F', 'ZM=F', 'ZL=F', 'HO=F', 'CL=F'}

# ==========================================================
# Commodities dictionary
# ==========================================================
commodities_dict = {
    'ZS=F': 'soybean',
    'ZC=F': 'corn',
    'ZW=F': 'wheat',
    'ZM=F': 'soybean_meal',
    'ZL=F': 'soybean_oil',
    'HO=F': 'heating_oil',
    'CL=F': 'crude_oil',
    'RS=F': 'rapeseed_oil',
}

# ==========================================================
# Contract sizes for each commodity
# ==========================================================
contract_sizes = {
    'soybean': 136.0,
    'corn': 127.0,
    'wheat': 136.0,
    'soybean_meal': 0.9072,
    'soybean_oil': 27.22,
    'heating_oil': 133.0,
    'crude_oil': 136.4
}

# ==========================================================
# Tons converters for each commodity
# ==========================================================
tons_conversion = {
    'soybean': 36.74 / 100,
    'corn': 39.37 / 100,
    'wheat': 36.74 / 100,
    'soybean_meal': 1.1023,
    'soybean_oil': 22.0462,
    'heating_oil': 312.5,
    'crude_oil': 7.4
}

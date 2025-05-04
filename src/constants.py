tickers = ['ZS=F', 'ZC=F', 'ZW=F', 'ZM=F', 'ZL=F', 'HO=F', 'CL=F']

# ==========================================================
# Commodities dictionary
# ==========================================================
commodities_dict = {
    'ZS=F': 'Soybean',
    'ZC=F': 'Corn',
    'ZW=F': 'Wheat',
    'ZM=F': 'Soybean meal',
    'ZL=F': 'Soybean oil',
    'HO=F': 'Heating oil',
    'CL=F': 'Crude oil',
}

# ==========================================================
# Contract sizes for each commodity
# ==========================================================
contract_sizes = {
    'Soybean': 136.0,
    'Corn': 127.0,
    'Wheat': 136.0,
    'Soybean meal': 0.9072,
    'Soybean oil': 27.22,
    'Heating oil': 133.0,
    'Crude oil': 136.4
}

# ==========================================================
# Tons converters for each commodity
# ==========================================================
tons_conversion = {
    'Soybean': 36.74 / 100,
    'Corn': 39.37 / 100,
    'Wheat': 36.74 / 100,
    'Soybean meal': 1.1023,
    'Soybean oil': 22.0462,
    'Heating oil': 312.5,
    'Crude oil': 7.4
}

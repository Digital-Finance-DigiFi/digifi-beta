import csv
import datetime
import numpy as np



def get_test_stock_data() -> dict[str, np.ndarray]:
    """
    ## Description
    Import test_stock_data.csv file.
    """
    with open(r"./tests/digifi/test_stock_data.csv", "r") as file:
        # Read CSV file
        reader = csv.reader(file, delimiter=",")
        # Skip header
        headers = next(reader)
        # Data type conversion
        data = [np.array([datetime.datetime.strptime(row[0], '%Y-%m-%d'), float(row[1]), float(row[2]), float(row[3]), float(row[4]),
                          float(row[5]), float(row[6])]) for row in reader]
    # Dictionary with CSV headers as columns
    data = np.array(data).T
    data_dict = dict()
    for i in range(len(headers)):
        if headers[i]=="Date":
            data_dict[headers[i]] = data[i]
        else:
            data_dict[headers[i]] = data[i].astype(float)
    return data_dict



def get_test_portfolio_data() -> dict[str, np.ndarray]:
    """
    ## Description
    Import test_portfolio_data.csv file.
    """
    with open(r"./tests/digifi/test_portfolio_data.csv", "r") as file:
        # Read CSV file
        reader = csv.reader(file, delimiter=",")
        # Skip header
        headers = next(reader)
        # Data type conversion
        data = [np.array([datetime.datetime.strptime(row[0], '%Y-%m-%d'), np.float64(row[1]), np.float64(row[2]), np.float64(row[3]),
                          np.float64(row[4])]) for row in reader]
    # Dictionary with CSV headers as columns
    data = np.array(data).T
    data_dict = dict()
    for i in range(len(headers)):
        if headers[i]=="Date":
            data_dict[headers[i]] = data[i]
        else:
            data_dict[headers[i]] = data[i].astype(float)
    return data_dict



def get_test_capm_data() -> dict[str, np.ndarray]:
    """
    ## Description
    Import test_capm_data.csv file.
    """
    with open(r"./tests/digifi/test_capm_data.csv", "r") as file:
        # Read CSV file
        reader = csv.reader(file, delimiter=",")
        # Skip header
        headers = next(reader)
        # Data type conversion
        data = [np.array([datetime.datetime.strptime(row[0], '%Y-%m-%d'), np.float64(row[1]), np.float64(row[2]), np.float64(row[3]),
                          np.float64(row[4]), np.float64(row[5]), np.float64(row[6]), np.float64(row[7]), np.float64(row[4])]) for row in reader]
    # Dictionary with CSV headers as columns
    data = np.array(data).T
    data_dict = dict()
    for i in range(len(headers)):
        if headers[i]=="Date":
            data_dict[headers[i]] = data[i]
        else:
            data_dict[headers[i]] = data[i].astype(float)
    return data_dict
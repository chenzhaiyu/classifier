import csv
import numpy as np


class DataLoader:

    def load_data(self, csv_path):
        """
        Load data from a csv file
        """
        records = []
        with open(csv_path) as data:
            reader = csv.reader(data)
            for record in reader:
                records.append(record)
        return np.array(records).astype(np.int)


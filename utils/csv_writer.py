import csv
from typing import Any


class CSVWriter:
    def __init__(self, path: str):
        self.path = path
        self.has_written = False

    def writerow(self, row: dict[str, Any]):
        with open(self.path, mode="a" if self.has_written else "w", newline='') as f:
            csv.excel.delimiter = ';'
            writer = csv.writer(f, dialect=csv.excel)

            if not self.has_written:
                writer.writerow(row.keys())
                self.has_written = True

            rounded_values = [round(value, 4) if isinstance(value, float) else value for value in row.values()]
            writer.writerow(rounded_values)

import pandas as pd
import json

class ConfigSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("Creating new instance")
            cls._instance = super(ConfigSingleton, cls).__new__(cls)
            # Initialization only happens here
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        print("Loading config, PATH, GPS, parquet")
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        self.PATH = self.config['path']

        with open(f"{self.PATH}/GPS.json", 'r') as f:
            self.GPS = json.load(f)["GPS"]

        self.PARQUET = pd.read_parquet(f"{self.PATH}/tools/precalculated_combinaison_heavy.parquet", engine='pyarrow')
        print("Loading finished")




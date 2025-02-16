import json


class ModelConfig:
    def __init__(self, company, model):
        self.config_file_path = ""
        with open(self.config_file_path, "r") as f:
            self.config = json.load(f)

        self.api_key = self.config[company]["api_key"]
        self.api_url = self.config[company]["api_url"]
        self.api_endpoint = self.config[company][model]

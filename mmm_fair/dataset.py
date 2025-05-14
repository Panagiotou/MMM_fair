'''
Dataset class copied from https://github.com/mammoth-eu/mammoth-commons
'''

class Dataset:
    #integration = "dsl.Dataset"

    def to_features(self, sensitive):
        raise Exception(
            f"{self.__class__.__name__} has no method to_features(sensitive)"
        )

    def format_description(self):
        if not hasattr(self, "description"):
            return ""
        dataset_desc = "<h1>Dataset</h1>"
        if isinstance(self.description, str):
            dataset_desc += self.description + "<br>"
        elif isinstance(self.description, dict):
            for key, value in self.description.items():
                dataset_desc += f"<h3>{key}</h3>" + value.replace("\n", "<br>") + "<br>"
        else:
            raise Exception("Dataset description must be a string or a dictionary.")
        return dataset_desc

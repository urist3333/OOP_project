import os
import subprocess


class DataLoader:
    """
    Class DataLoader
    This class forms a superclass for subclass DataLoaders such as the MnistDataLoader 
    This class forms the blueprint for loading various data for Machine-Learning purposes.
    Therefore the class has public methods for returning training/testing data 
    In addition to private methods for retrieving the data from a web-source
    Args:
        dset (str): name for the dset that one desires to load
    Attributes:
        dset: name of dataset being loaded 
        _data: The actual loaded data 
        _data_dir: Directory where the data is stored
        _ext: Determined dynamically in subclasses
        _url: The url source for the data
        _file_name: The specific filename of the data


    """
    def __init__(self,dset):
        self.dset = dset
        self._data_dir = "./data"
        self._ext = None
        self._data = None
        self._url =""
        self._file_name = ""


    @property
    def _file_path(self):
        """
        Property to dynamically compute the file_path by joining the _data_dir and _file_name 
        Helper for subclasses
        """
        return os.path.join(self._data_dir, f"{self._file_name}")
    def _get_ext(self):
        """
        Abstract method
        Ensures all subclasses respect the contract and gets the file extension 
        """
        raise NotImplementedError("All subclasses must determine the file extension of the downloaded file")

    def _make_dir(self):
        """
        Private method. 
        If the directory does not exist it will create it
        """
        if not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)
            print(f"Made directory at {self._data_dir}")
        else:
            print(f"Directory already exists at {self._data_dir}")

    def _download_data(self):
        """
        Private method
        Checks if the file_path exists first, if not it downloads the data from the url using wget
        """
        self._make_dir()
       

        if not os.path.exists(self._file_path):
            print(f"Downloading {self.dset} from {self._url}")
            try: 
                subprocess.run(["wget", self._url, "-O", self._file_path], check=True)

            except Exception as error:
                raise ValueError(f"Download failed: {error}")
        else: 
            print(f"{self._file_path} already downloaded")

    def _load_data(self):
        """
        Abstract private method. Ensures all subclasses load the data
        """
        raise NotImplementedError("All subclasses must load_data")
    def _transform_data(self):
        """
        Abstract private method. Ensures all subclasses consider transforming the data
        """
        raise NotImplementedError("All subclasses must consider if transformation is required")

    def get_training_data(self):
        """
        Abstract public method. Ensures all subclasses has a method for retrieving the data for training
        """
        raise NotImplementedError("All subclasses must get training data")
    def get_test_data(self):
        """
        Abstract public method. Ensures all subclasses has a method for retrieving the data for testing
        """
        raise NotImplementedError("All subclasses must get testing data")





        
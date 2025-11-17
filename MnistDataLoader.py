from DataLoader import DataLoader
import numpy as np
import pickle
import tensorflow as tf

class MnistDataLoader(DataLoader):

    """
    MnistDataLoader subclass
    Inherits from DataLoader superclass
    Args:
        dset: Name of dataset being loaded. (mnist_bw,mnist_color)
        version: Version of mnist_color. (m0,m1,m2,m3,m4)
    Attributes:
        dset: Name of dataset being loaded. (mnist_bw,mnist_color)
        version: Version of mnist_color. (m0,m1,m2,m3,m4)
        Attributes intialized in subclass MnistDataloader:
        _url_map_tr: Dictionary of urls to retrieve training data
        _url_map_te: Dictionary of urls to retrieve testing data 
        _url_map_labels: Dictionary of urls to retrive labels for testing data
        
        Attributes Inherited from DataLoader:
        _data: The actual loaded data 
        _data_dir: Directory where the data is stored
        _ext: Determined dynamically in subclasses
        _url: The url source for the data
        _file_name: The specific filename of the data
        _file_path: The path of the file


        
       

    """
    def __init__(self,dset,version =None):
        super().__init__(dset)
        
        self._url_map_tr = {"mnist_bw": "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=1",
                        "mnist_color": "https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=1"
                        }
        self._url_map_te = {
                "mnist_bw_te": "https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=1",
                "mnist_color_te": "https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=1"
            }
        self._url_map_labels = {"mnist_bw_y_te":'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=1',
         
                            "mnist_color_y_te":'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=1'}
        self.version = version
    def _get_ext(self,key):
        """
        Private method
        Determines the correct file extension based on the URL

        """
        if key in self._url_map_labels:
            return ".npy"
        
        url = None
        if key in self._url_map_tr:
            url =self._url_map_tr[key]
        elif key in self._url_map_te:
            url = self._url_map_te[key]

        if ".pkl" in url:
            return ".pkl"
        
        else:
            return ".npy"
    
    def _load_file(self):
        """
        Private method
        Loads the file depending on the file extension
        """
        if self._ext == ".npy":
            return np.load(self._file_path).astype(np.float32)
        if self._ext == ".pkl":
            with open(self._file_path,"rb") as file:
                return pickle.load(file)
            
        raise ValueError(f"Unknown extension {self._ext}")
            
    
            
    def _load_data(self):
        """
        Private method
        Loads the data depending on the extension and the file version if its for mnist_color
        """
        raw_file = self._load_file()
        if self._ext == ".pkl":
            if self.version not in raw_file:
                raise KeyError(f"Version '{self.version}' not found \n, Please specify a version: 'm0', 'm1', 'm2', 'm3', or 'm4'")
            self._data =raw_file[self.version].astype(np.float32)
    
        else:
            
            self._data = raw_file

    def _transform_data(self):
        """
        Private method
        Transforms mnist_bw only 
        """
        if self.dset == "mnist_bw":
            self._data = self._data / 255.0
            self._data = self._data.reshape(len(self._data), -1)
        else:
            print(f"No transformation performed, {self.dset} does not require transformation")
        
    def get_training_data(self,batch_size =256):
            """
            Public method
            Implements the abstract method from DataLoader superclass
            Retrieves training_data in batches in tensorflow applicable format
            Args:
                batch_size: Default value 256 
            """
            key = self.dset
            self._ext = self._get_ext(key)
            self._file_name = f"{self.dset}{self._ext}"
            self._url = self._url_map_tr[self.dset]
        
            self._download_data()
            self._load_data()
            self._transform_data()
            if self._data is not None:
                data_length = len(self._data)
            else:
                raise ValueError("There is no data to get")
            tr_data = tf.data.Dataset.from_tensor_slices(self._data)
            tr_data = tr_data.shuffle(buffer_size=data_length).batch(batch_size)            
            return tr_data
    
    def get_testing_data(self):
            """
            Public method
            Implements the abstract method from DataLoader superclass
            Retrieves the testing data in a tensor.

            """
            
            key = f"{self.dset}_te"
            self._url = self._url_map_te[key]
            self._ext = self._get_ext(key)
            self._file_name = f"{key}{self._ext}"
            
            self._download_data()
            self._load_data()
            self._transform_data()
            test_data = tf.convert_to_tensor(self._data)
            return test_data
    
    
    def get_labels(self): 
            """
            Public method 
            Retrieves the label for mnist_color and mnist_bw, useful for plotting the latent space
            """
            key = f"{self.dset}_y_te"
            self._ext = self._get_ext(key)
            self._file_name = f"{key}{self._ext}"
            self._url = self._url_map_labels[key]
            
            self._download_data()
            self._load_data()
            labels = self._data
            return labels
        

        

        
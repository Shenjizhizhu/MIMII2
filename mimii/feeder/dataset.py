import numpy as np
import grain.python as grain

class Feeder_snr(grain.RandomAccessDataSource):
    def __init__(self,data_path,label_path):
        self._data = np.load(data_path)
        self._label = np.load(label_path)
    def __getitem__(self,idx):
        return {'data':self._data[idx,...,None],'label':self._label[idx,0].astype(np.int32)}
    def __len__(self):
        return len(self._data)
    
class Feeder_device(grain.RandomAccessDataSource):
    def __init__(self,data_path,label_path,snr):
        self._data = np.load(data_path)
        self._data = self._data.reshape(3,-1,self._data.shape[1],self._data.shape[2])
        self._label = np.load(label_path)
       
        for i in range(3):
            if snr == self._label[i][0][0]:
                self._label = self._label[i]
                self._data = self._data[i]
                break
    def __len__(self):
        return len(self._data)
    def __getitem__(self,idx):
        return{'data': self._data[idx, ..., None], 'label': self._label[idx].astype(np.int32)}
    
class Feeder_label(grain.RandomAccessDataSource):
    def __init__(self,data_path,label_path,snr):
        self._data = np.load(data_path)
        self._data = self._data.reshape(3,-1,self._data.shape[1],self._data.shape[2])
        self._label = np.load(label_path).reshape(3,-1)

        for i in range(3):
            if snr == self._label[i][0][0]:
                self._label = self._label[i]
                self._data = self._data[i]
                break
    def __len__(self):
        return len(self._data)
    def __getitem__(self,idx):
        return {'data': self._data[idx, ..., None], 'label': self._label[idx].astype(np.int32)}

def load_data(modal,snr=0,num_workers=8,num_epoch=10,batch_size=256):
    if modal =='snr':
        train_source = Feeder_snr('/home/shenji/shenji111/MIMII/data/data.npy','/home/shenji/shenji111/MIMII/data/label.npy')
        test_source = Feeder_snr('/home/shenji/shenji111/MIMII/data/data.npy','/home/shenji/shenji111/MIMII/data/label.npy')
    elif modal =='device':
        train_source = Feeder_device('/home/shenji/shenji111/MIMII/data/data.npy','/home/shenji/shenji111/MIMII/data/label.npy', snr)
        test_source = Feeder_device('/home/shenji/shenji111/MIMII/data/data.npy','/home/shenji/shenji111/MIMII/data/label.npy', snr)
    elif modal == 'label':
        train_source = Feeder_label('/home/shenji/shenji111/MIMII/data/data.npy','/home/shenji/shenji111/MIMII/data/label.npy', snr)
        test_source = Feeder_label('/home/shenji/shenji111/MIMII/data/data.npy','/home/shenji/shenji111/MIMII/data/label.npy', snr)

    train_loader = grain.load(
        data_source = train_source,
        batch_size = batch_size,
        operations = [grain.Batch(batch_size = batch_size)],
        worker_count = num_workers
    )
    test_loader = grain.load(
        data_source = test_source,
        batch_size = batch_size,
        operation = [grain.Batch(batch_size = batch_size)],
        worker_count = num_workers
    )
    return train_loader,test_loader


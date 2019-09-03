import os
import inkmlParser as parser
import convertInkmlToImg as convertor
from pymongo import MongoClient
import scipy.ndimage as ndimage
import pickle
import bson

def load_data_in_mongo():
    client = MongoClient('localhost', 27017)
    db = client['handwrtingdata']
    collections = db.list_collection_names()
    symbols_dataset = db.get_collection(name = 'Symbols')

    image_size = 50
    rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/train/junk'
    current_dataset_type = 'train'
    #data_range = 18436

    datalist = []

    for filename in os.listdir(rootdir):

        traces, truth, ui, *rest = parser.parse_inkml(rootdir + '/' + filename)
        selected_tr = convertor.get_traces_data(traces)
        im = convertor.convert_to_imgs(selected_tr, image_size)
        im = ndimage.gaussian_filter(im, sigma=(.5, .5), order=0)

        data = {}
        data['_id'] = ui
        data['ui'] = ui
        data['type'] = current_dataset_type
        data['truth'] = truth
        data['traces'] = bson.binary.Binary( pickle.dumps( im, protocol=2) )
        data['stroke_count'] = len(selected_tr)
        data['soft_delete'] = False

        #x = bson.binary.Binary( pickle.dumps( im, protocol=2) )
        #y = pickle.loads(x)

        datalist.append(data)

    try:
        symbols_dataset.insert_many(documents = datalist, ordered=False)
    except Exception as e:
        print(e)
    print(collections)
import os
import inkmlParser as parser
import convertInkmlToImg as convertor
from pymongo import MongoClient
import scipy.ndimage as ndimage
import pickle
import bson
import visualize
import numpy as np

def load_data_in_mongo():
    # Training Data
    load_data_in_mongo_from_root_dir('C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/train/junk')
    load_data_in_mongo_from_root_dir('C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/train/good')
    
    #Test Data
    load_data_in_mongo_from_root_dir('C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/test')

    #Validation Data
    load_data_in_mongo_from_root_dir('C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/validation')



def load_data_in_mongo_from_root_dir(rootdir:str):
    client = MongoClient('localhost', 27017)
    db = client['handwrtingdata']
    symbols_dataset = db.get_collection(name = 'Symbols')

    image_size = 50
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

def display_data_in_mongo():
    client = MongoClient('localhost', 27017)
    db = client['handwrtingdata']
    symbols_dataset = db.get_collection(name = 'Symbols')

    search_term = 'x'

    query = {"truth":search_term , 'soft_delete' : False, "type": "train"}

    results = symbols_dataset.find(query, {"traces": 1 , "_id": 0})

    trace_list = []

    for result in results:
        trace_list.append(pickle.loads(result['traces']))

    print (len(trace_list))

    if (len(trace_list) > 100): 
        visualize.visualize_trace_List(trace_list[:100])

def get_dataset(datasetname:str, databasename = 'handwrtingdata'):
    client = MongoClient('localhost', 27017)
    db = client[databasename]
    return db.get_collection(name = datasetname)

def get_traces_for_id_from_db(id:str):
    symbols_dataset = get_dataset('Symbols')
    query = {'_id': id}
    results =  symbols_dataset.find(query, {"traces": 1 , "truth": 1, "_id": 1})
    return pickle.loads(results[0]['traces'])
    
def get_data_from_db(datatype:str, skipJunk:bool=True):
    symbols_dataset = get_dataset('Symbols')
    query = {'soft_delete' : False, "type": datatype}

    if(skipJunk):
        query['truth'] = {'$ne': 'junk'}
    
    results = symbols_dataset.find(query, {"traces": 1 , "truth": 1, "_id": 1})

    trace_list = []
    truth_list = []
    id_list = []

    for result in results:
        trace_list.append(pickle.loads(result['traces']))
        truth_list.append(result['truth'])
        id_list.append(result['_id'])

    trace_list = np.array(trace_list)
    truth_list = np.array(truth_list)
    id_list = np.array(id_list)

    return trace_list, truth_list, id_list


def get_train_data_from_db(skipJunk:bool=True):
    return get_data_from_db('train', skipJunk)

def get_test_data_from_db(skipJunk:bool=True):
    return get_data_from_db('test', skipJunk)

def get_validation_data_from_db(skipJunk:bool=True):
    return get_data_from_db('validation', skipJunk)

#get_train_data_from_db()
#get_train_data_from_db(skipJunk=False)
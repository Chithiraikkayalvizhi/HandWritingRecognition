import os
from lxml import etree

def parse_inkml(inkml_file_abs_path:str) -> ({}, str):
    if inkml_file_abs_path.endswith('.inkml'):
        tree = etree.parse(inkml_file_abs_path)
        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"
        gt = ''
        ui = ''
        'Stores traces_all with their corresponding id'
	    #MM: multiple all integers and floats by 10K
        traces_all_list = [{'id': trace_tag.get('id'),
                            'coords': [[round(float(axis_coord) * 10000) \
                                            for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                        else [round(float(axis_coord) * 10000) \
                                            for axis_coord in coord.split(' ')] \
                                    for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                                    for trace_tag in root.findall(doc_namespace + 'trace')]
        
        annotations = root.findall(doc_namespace + 'annotation')
        for annotation in annotations:
            if "truth" == annotation.attrib['type']:
                gt = annotation.text
                break
        for annotation in annotations:
            if "UI" == annotation.attrib['type']:
                ui = annotation.text
                break

        'convert in dictionary traces_all  by id to make searching for references faster'
        traces_all = {}
        for t in traces_all_list:
            traces_all[t["id"]] = t["coords"]
        #print("traces_alllalalalal",traces_all)
        #traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
        return traces_all, gt, ui
    else:
        print('File ', inkml_file_abs_path, ' does not exist !')
        return {}, ''

def ValidateFilesInDirectory(rootdir:str, range:int):
    valid = 0
    invalid = 0
    # for filename in os.listdir(rootdir):
    for filename in os.listdir(rootdir)[:range]:
        file = open(rootdir + '/' + filename, mode='r', encoding="utf-8")
        try:
            file_content_string = file.read()
            etree.fromstring(file_content_string)
            print(filename , " : is valid XML")
            valid += 1
        except Exception as e:
            invalid += 1
            print(filename,  " : is invalid XML", e)
    
    print ("Valid: ", valid)
    print ("invalid: ", invalid)
    print ("total", valid + invalid)

def FixInkmlFilesInDirectory(rootdir:str, range:int):

    # Fill the ground truth annotations
    groundTruthList = ReadGroundTruthFromFlatFile(rootdir)

    for filename in os.listdir(rootdir)[:range]:
        isChanged = False
        file = open(rootdir + '/' + filename, mode='r', encoding="utf-8")
        file_content_string = file.read()
        try:
            root = etree.fromstring(file_content_string)
        except Exception as _e:
            file_content_string = FixInkmlFileIDAttributes(file_content_string)
            print(filename, " : Needs xml:id fix")       
            try:
                root = etree.fromstring(file_content_string)
                isChanged = True
            except:
                print(filename, " : Still invalid after trying to fix ID attribute. Skipping")
                continue

        ns = {"inkml": "http://www.w3.org/2003/InkML"}
        unique_id = root.xpath('./inkml:annotation[@type="UI"]', namespaces = ns)[0].text
        root, isChanged = InkmlFillGroundTruth(root, groundTruthList[unique_id], forceUpdate=False)
        # InkmlUnnestTraceGroup(root)
        # InkmlRegenerateIds(root)

        if(isChanged):
            file.close()
            OverwriteFile(rootdir, filename, etree.tostring(root, pretty_print=True))
            print(filename, ": file has been overwritten with required fixes")

def ReadGroundTruthFromFlatFile(rootdir:str) -> (dict):
    groundTruthList = {}
    with open(rootdir + '/' + "_GT.txt", mode='r', encoding="utf-8") as file:
        for line in file:
            values = line.strip().split(",")
            groundTruthList[values[0]] = values[1].strip()
    return groundTruthList

def FixInkmlFileIDAttributes(filecontents:str) -> (str):
    return filecontents.replace("xml:id", "id")

def InkmlFillGroundTruth(root:etree._Element, groundTruth:str, forceUpdate:bool= False) -> (etree._Element, bool):
    isChanged = False
    for _i, children in enumerate(root):
        if (children.tag == '{http://www.w3.org/2003/InkML}annotation' and 'type' in children.attrib and children.attrib['type'] == 'truth'):
            if(children.text is None or forceUpdate):
                children.text = groundTruth
                isChanged = True
    return root, isChanged 

def OverwriteFile(rootdir:str, filename:str, filecontents:str):
    file = open(rootdir + '/' + filename, mode='w', encoding="utf-8")
    file.write(filecontents.decode('utf-8'))
    file.close()

def ProcessInkmlSymbolsDataset(training_set:bool=False, test_set:bool=False, validation_set:bool=False, range:int=0):
    train_good_rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/train/good'
    train_junk_rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/train/junk'

    test_rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/test'

    validation_rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/validation'

    train_good_range = 85803
    train_junk_range = 74285
    test_range = 18436
    validation_range = 12505

    if(range != 0):
        train_good_range = range
        train_junk_range = range
        test_range = range
        validation_range = range

    if(training_set):
        FixInkmlFilesInDirectory(train_good_rootdir, train_good_range)
        FixInkmlFilesInDirectory(train_junk_rootdir, train_junk_range)
    
    if(test_set):
        FixInkmlFilesInDirectory(test_rootdir, test_range)
    
    if(validation_set):
        FixInkmlFilesInDirectory(validation_rootdir, validation_range)




# rootdir = 'C:/github/Kayal_Capstone/HandWritingRecognition/Data/Raw/2016/symbols/validation'
#range = 10
#range = 18436 #test set size
#range = 85803 #Good Traning set size
#range = 74285 #Junk training set size
#range = 12505 #validation set size

# ValidateFilesInDirectory(rootdir, range)
#ProcessInkmlSymbolsDataset(training_set=True, test_set=True, validation_set=True, range=10)
# ValidateFilesInDirectory(rootdir, range)
# LoadDocumentsToMongo(rootdir, range)



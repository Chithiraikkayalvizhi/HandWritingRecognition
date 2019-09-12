import inkmlParser as inkmlParser
import mongo as mongo

inkmlParser.ProcessInkmlSymbolsDataset(training_set=True, test_set=True, validation_set=True, range=10)
mongo.load_data_in_mongo()
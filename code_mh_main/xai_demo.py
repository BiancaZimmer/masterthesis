"""**XAI Demo** - This includes the implemented FLASK app.
"""

from flaskutil import FlaskApp
from flask import Flask,request,g,jsonify
from flask.templating import render_template, render_template_string
import os
import json
import ntpath
import numpy as np

from utils import *
from cnn_model import *
from dataset import *
from feature_extractor import FeatureExtractor
from dataentry import DataEntry
from near_miss_hits_selection import calc_distances_scores, get_nearest_hits, get_nearest_miss



def get_prototypes_by_img_files(data, protos_dict:dict):
    """Based on the image names of the prototypes, which are already stored locally for the online phase, the corresponding DataEntries are loaded and returned in a dictionary.
    This allows an easy access to their attribute, such as image path.

    :param data: Current Selected DataSet
    :type data: DataSet
    :param protos_dict: Dictionary of image names of the prototypes, which should loaded beforehand.
    :type protos_dict: dict
    
    :return: *self* (`dict`) - Dictionary of DataEntries of the pre-calculated prototypes
    """
    assert protos_dict is not None, "No Prototypes selected yet! Please, fit the Selector first!"
    #print('classes: ...', proto_classes)
    prototypes_per_class = {}
    for proto_class in protos_dict.keys():
        #print('proto ..', proto_class)
        prototypes_per_class[proto_class] = [ [entry for entry in data if entry.img_name==proto_img_name][0] for proto_img_name in protos_dict[proto_class]]
    return prototypes_per_class
    
def get_dataentry_by_img_name(data, img_name:str):
    """For a given image name, the corresponding DataEntry object is retrieved from a given list of DataEntry (refering to the dataset).

    :param data: List of DataEntries
    :type data: list
    :param img_name: Image name of the viewed DataEntry (e.g. test input)
    :type img_name: str

    :return: *self* (`DataEntry`) - DataEntry object for the given image name
    """
    data_entry = [entry for entry in data if entry.img_name==img_name][0]
    return data_entry
    
def get_dataentry_by_img_path(data, img_path:str):
    """For a given image path, the corresponding DataEntry object is retrieved from a given list of DataEntry (refering to the dataset).

    :param data: List of DataEntries
    :type data: list
    :param img_path: Image path of the viewed DataEntry (e.g. test input)
    :type img_path: str

    :return: *self* (`DataEntry`) - DataEntry object for the given image path
    """
    data_entry = [entry for entry in data if entry.img_path==img_path][0]
    return data_entry
    
# List if local avaiable datasets (global variable)
dataset_list = get_available_dataset()
# List if local avaiable datasets (global variable)
dict_datasets_and_embedings =  get_dict_datasets_with_all_embeddings()

dict_cnn_models = {}
for dataset_name, embedding_dict in dict_datasets_and_embedings.items():
    #touch the embeddings to load in ram
    for model_name, value in embedding_dict.items():
        [x.feature_embedding for x in value.data]
        if model_name == "SimpleCNN":
            dict_cnn_models[dataset_name] = CNNmodel(selected_dataset = value)
            dict_cnn_models[dataset_name]._preprocess_img_gen()
            dict_cnn_models[dataset_name]._binary_model()
            dict_cnn_models[dataset_name].load_model() 



use_CNN_feature_embeddding = True


class ExamplebasedXAIDemo(FlaskApp):
    """The implemented FLASK app relying on the developed architecture for prototypes, near hits and misses selection.
    """
    
    sel_dataset = None
    test_dataentry = None
    data = None
    data_t = None

    fe = None
    distance_measure = 'cosine'

    cnn_model = None

    prototypes_per_class_img_files = None

    dict_datasets = {}
    for dataset_name, embedding_dict in dict_datasets_and_embedings.items():
        dict_datasets[dataset_name] = embedding_dict['SimpleCNN']


    def doRequest(self):
        """This method takes care of "normal" GET-requests. Should return a full html "file", e.g. a template (using render_template or render_template_string). 

        :return: *self* (`str`) - A full html "file", e.g. a template (using render_template or render_template_string)
        """
        return render_template('xai_demo.html',
                                datasets = dataset_list)
                                    
    def callbacks(self,id:str,type:str,value):
        """This functions use the javascript callbacks return JSON directionary in order to instructs the client to replace the content of the given ID with the new given content.

        :param id: ID of some node in the current HTML
        :type id: str
        :param type: Typ of action, e.g onchange, onclick
        :type type: str
        :param value: The new HTML content of that node
        :type value: str

        :return: *self* (`dict in JSON`) - JSON dictionary of the this ID and HTML content

        """
        if id == 'model-specific-button' and type == 'onchange':
            self.dict_datasets = {}
            for dataset_name, embedding_dict in dict_datasets_and_embedings.items():
                self.dict_datasets[dataset_name] = embedding_dict['SimpleCNN']

            g.datasets = dataset_list

            return jsonify([{'elem':'dataset-dropdown','content':render_template_string('<option selected disabled>Choose here</option>{% for dataset_option in g.datasets %}<option value= {{dataset_option}} >{{ dataset_option }}</option>{% endfor %}')},
            {'elem':'test-dropdown','content':render_template_string('<option selected disabled>Choose here</option>')},
            {'elem':'test-image','content':render_template_string('')},{'elem':'test-prediction','content':render_template_string('')},
            {'elem':'nh-nm-images','content':render_template_string('')}, {'elem':'protos-images','content':render_template_string('')}])
        
        elif id == 'model-agnostic-button' and type == 'onchange':
            self.dict_datasets = {}
            for dataset_name, embedding_dict in dict_datasets_and_embedings.items():
                self.dict_datasets[dataset_name] = embedding_dict['VGG16']

            g.datasets = dataset_list

            return jsonify([{'elem':'dataset-dropdown','content':render_template_string('<option selected disabled>Choose here</option>{% for dataset_option in g.datasets %}<option value= {{dataset_option}} >{{ dataset_option }}</option>{% endfor %}')},
            {'elem':'test-dropdown','content':render_template_string('<option selected disabled>Choose here</option>')},
            {'elem':'test-image','content':render_template_string('')},{'elem':'test-prediction','content':render_template_string('')},
            {'elem':'nh-nm-images','content':render_template_string('')}, {'elem':'protos-images','content':render_template_string('')}])


        elif id == 'distance-dropdown' and type == 'onchange':

            self.distance_measure = value

            g.datasets = dataset_list

            return jsonify([{'elem':'dataset-dropdown','content':render_template_string('<option selected disabled>Choose here</option>{% for dataset_option in g.datasets %}<option value= {{dataset_option}} >{{ dataset_option }}</option>{% endfor %}')},
            {'elem':'test-dropdown','content':render_template_string('<option selected disabled>Choose here</option>')},
            {'elem':'test-image','content':render_template_string('')},{'elem':'test-prediction','content':render_template_string('')},
            {'elem':'nh-nm-images','content':render_template_string('')}, {'elem':'protos-images','content':render_template_string('')}])


        elif id == 'dataset-dropdown' and type == 'onchange':

            self.sel_dataset = value

            self.data =  self.dict_datasets[self.sel_dataset].data
            self.data_t =  self.dict_datasets[self.sel_dataset].data_t
            self.fe = self.dict_datasets[self.sel_dataset].fe
            
            
            g.test_set =  self.dict_datasets[self.sel_dataset].data_t

            return jsonify([{'elem':'test-dropdown','content':render_template_string('<option selected disabled>Choose here</option>{%for test_sample in g.test_set %}<option value= {{test_sample.img_path}} >{{ test_sample.img_name }}</option>{% endfor %}')},
            {'elem':'test-image','content':render_template_string('')},{'elem':'test-prediction','content':render_template_string('')},
            {'elem':'nh-nm-images','content':render_template_string('')}, {'elem':'protos-images','content':render_template_string('')}])
        
        elif id == 'test-dropdown' and type == 'onchange':
            g.test_img_path = value
            self.test_dataentry = get_dataentry_by_img_path(self.data_t, value)
            print('test_dataentry', self.test_dataentry)
            print(self.test_dataentry.img_path)
            return jsonify([{'elem':'test-image','content':render_template_string('<img src="{{ g.test_img_path }}" height="200px">')},
            {'elem':'test-prediction','content':render_template_string('')},
            {'elem':'nh-nm-images','content':render_template_string('')}, {'elem':'protos-images','content':render_template_string('')}])

        elif id == 'classify-button' and type == 'onclick':

            self.cnn_model = dict_cnn_models[self.sel_dataset]
            self.pred_label, self.pred_prob = self.cnn_model.pred_test_img(self.test_dataentry)
            g.pred_label, g.pred_prob = self.pred_label, self.pred_prob

            return jsonify([{'elem':'test-prediction','content':render_template_string('<p>Predicted Label: <b>{{ g.pred_label }}</b></p><p>Predicted Probability: <b>{{ g.pred_prob }}</b></p>')}])

        elif id == 'explain-button' and type == 'onclick':

            top_n = 5
          
            scores_nearest_hit, ranked_nearest_hit_data_entry = get_nearest_hits(self.test_dataentry, self.pred_label, self.data, self.fe, top_n, self.distance_measure)
            scores_nearest_miss, ranked_nearest_miss__data_entry = get_nearest_miss(self.test_dataentry, self.pred_label, self.data, self.fe, top_n, self.distance_measure)

            g.nearest_hits = zip([x.img_path for x in ranked_nearest_hit_data_entry], scores_nearest_hit, ['nh_'+str(i+1) for i in range(top_n)])
            g.nearest_misses = zip([x.img_path for x in ranked_nearest_miss__data_entry], scores_nearest_miss, ['nm_'+str(i+1) for i in range(top_n)])

            # use_CNN_feature_embeddding = True
            num_prototypes= 3
            # if use_CNN_feature_embeddding:
            #     DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR,'static/prototypes', dict_datasets[self.sel_dataset].fe.fe_model.name ,self.sel_dataset)
            # else:
            #     DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR,'static/prototypes', "rawData",self.sel_dataset)

            DIR_PROTOTYPES_DATASET = os.path.join(MAIN_DIR,'static/prototypes', self.dict_datasets[self.sel_dataset].fe.fe_model.name ,self.sel_dataset)

            
            protos_file = os.path.join(DIR_PROTOTYPES_DATASET, str(num_prototypes) + '.json')

            if os.path.exists(protos_file):
                print('LOADING ...')
                with open(protos_file, 'r') as fp:
                    self.prototypes_per_class_img_files = json.load(fp)
            else:
                print(protos_file)

            prototypes_per_class = get_prototypes_by_img_files(data=self.data, protos_dict=self.prototypes_per_class_img_files)

            g.prototypes = prototypes_per_class[self.pred_label]
            print("PROTOTYPES: ", prototypes_per_class)


            return jsonify([
            {'elem':'protos-images',
            'content':render_template_string('<h2>Prototypes:</h2><div class="row"><div class="carousel clearfix"><div class="carousel-view clearfix">\
            {% for proto in g.prototypes %}<div class="box"><figure style="float: left; margin-right: 20px; margin-bottom: 20px;"><img src="{{ proto.img_path }}" height="200px"><figcaption>{{ proto.ground_truth_label }}</figcaption></figure></div>{% endfor %}</div></div></div>')},
            {'elem':'nh-nm-images',
            'content':render_template_string('<h2>Nearest Hits:</h2><div class="row">\
                {% for nearest_hit in g.nearest_hits %}<div id="{{ nearest_hit[2] }}"><figure style="float: left; margin-right: 20px; margin-bottom: 20px;"><img src="{{ nearest_hit[0] }}" height="200px"><figcaption>{{ nearest_hit[1] }}</figcaption></figure></div>{% endfor %}</div>\
                <h2>Nearest Miss:</h2><div class="row">\
                {% for nearest_miss in g.nearest_misses %}<div id="{{ nearest_miss[2] }}"><figure style="float: left; margin-right: 20px; margin-bottom: 20px;"><img src="{{ nearest_miss[0] }}" height="200px"><figcaption>{{ nearest_miss[1] }}</figcaption></figure></div>{% endfor %}</div>')}])
    

        return super(self).callbacks()
        

if __name__ == '__main__':

    app = ExamplebasedXAIDemo(
        # host='localhost'                  # the domain of the server
        # port=5000                         # the port of the server
        # prefix='/demo',                        # the path on the domain where to run the server
        static_folder="./static",        # the folder where static files (e.g. css) are found
        template_folder="./templates"   # where templates are found
        )
    app.run() # runs the server

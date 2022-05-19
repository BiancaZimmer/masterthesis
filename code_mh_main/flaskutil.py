from flask import Flask,request,g,jsonify
from flask.templating import render_template, render_template_string

class FlaskApp:
    prefix : str
    host : str
    port : int
    app : Flask
    elems = None

    def main(self):
        if request.path.startswith("/callbacks"):
            print("Callback " + request.full_path)
            id = request.values.dicts[0].get('id')
            type = request.values.dicts[0].get('type')
            value = request.values.dicts[0].get('value')
            print(id)
            print(type)
            print(value)
            return self.callbacks(id,type,value)
        if not request.path.startswith(self.prefix):
            return None
        if request.path.startswith("/static"):
            return None
        print(request.full_path)
        return self.doRequest()


    def doRequest(self):
        pass

    def callbacks(self,id:str,type:str,value:str):
        pass

    def run(self):
        self.app.run(host=self.host,port=self.port)

    def __init__(self,prefix="/",host="localhost",port=5000,static_folder = "web/static",template_folder = "web/templates"):
        self.prefix = prefix
        self.host = host
        self.port = port
        self.app = Flask(__name__, instance_relative_config=True,static_folder=static_folder,template_folder=template_folder)
        self.app.config.from_mapping(SECRET_KEY='dev') # is this even needed?
        self.app.before_request(self.main)
        pass

# EXAMPLE:
class MyExampleApp(FlaskApp):
    # This method takes care of "normal" GET-requests. Should return a full html "file", e.g. a template (using render_template or render_template_string). 
    # "request.path" can be used to determine which html should be returned. This example only acts on http://localhost:5000/test.
    def doRequest(self):
        g.value = request.path
        if request.path == '/test' or request.path == '/test/':
            g.value = 'TEST'
        #return render_template('test2.html')
        return render_template_string("""{% extends 'base.html' %}{% block body %}
        {{ g.value }}<div id="test" onclick="callback('onclick','test',null)">BLA</div>{% endblock %}""")
    
    # for javascript callbacks, use the javascript function "callback(type,id,value)", as in the return above ^
    # this method can then return a JSON dictionary consisting of 
    # 1. the id of some node in the current HTML and 
    # 2. the new html content of that node.
    # 
    # This example (returned by doRequest) provides a single <div>-node with id "test" with onclick="callback('onclick','test',null)", meaning
    # by clicking on the div, a callback will be executed. The callbacks-method then instructs the client to replace the content of the same
    # <div>-node by the <span> below.
    def callbacks(self,id:str,type:str,value:str):
        if id == 'test' and type == 'onclick':
            return jsonify({'elem':id,'content':render_template_string('<span>Success!</span>')})
        return super(self).callbacks()
        






if __name__ == '__main__':
    app = MyExampleApp(
        # host='localhost'                  # the domain of the server
        # port=5000                         # the port of the server
        # prefix='/'                        # the path on the domain where to run the server
        # static_folder="web/static"        # the folder where static files (e.g. css) are found
        # template_folder="web/templates"   # where templates are found
        )
    app.run() # runs the server
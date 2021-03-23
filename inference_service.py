import json
from Shufflenetv2_predict import shufflenet_inference
from flask import Flask, request
app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def train():
    # default server error
    data = {'status': 'failed', 'message': '500 INTERNAL SERVER ERROR'}
    response_obj = app.response_class(response=json.dumps(data), status=500, mimetype='application/json')
    try:
        if request.method == 'POST':
            classes = str(request.args.get('classes'))
            image = str(request.args.get('image'))
            img_data = shufflenet_inference(classes, image).inference_on_image()
            response_obj = app.response_class(response=json.dumps(img_data), status=200, mimetype='application/json')
            return response_obj
    except Exception as e:
        data = {'status': 'failed', 'message': str(e)}
        response_obj = app.response_class(response=json.dumps(data), status=500, mimetype='application/json')
    return response_obj

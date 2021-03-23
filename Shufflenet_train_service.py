import json
from shufflenetv2_train import Dataloader, SL_shufflenetV2
from flask import Flask, request
app = Flask(__name__)
@app.route('/train', methods=['POST'])
def train1():
    # default server error
    data = {'status': 'failed', 'message': '500 INTERNAL SERVER ERROR'}
    response_obj = app.response_class(response=json.dumps(data), status=500, mimetype='application/json')
    try:
        if request.method == 'POST':
            train_p = str(request.args.get('train_p'))
            val_p = str(request.args.get('val_p'))
            max_epochs_stop = int(request.args.get('max_epochs_stop'))
            n_epochs = int(request.args.get('n_epochs'))
            print_every = int(request.args.get('print_every'))
            SL_model, SL_criterion, SL_optimizer, SL_train, SL_val, savefile = Dataloader(train_p, val_p).data_loader()
            data = SL_shufflenetV2(SL_model, SL_criterion, SL_optimizer, SL_train, SL_val, savefile ,max_epochs_stop,n_epochs,print_every).train()
            response_obj = app.response_class(response=json.dumps(data), status=200, mimetype='application/json')
            return response_obj
    except Exception as e:
        data = {'status': 'failed', 'message': str(e)}
        response_obj = app.response_class(response=json.dumps(data), status=500, mimetype='application/json')
    return response_obj


if __name__ == '__main__':
    app.run( host='0.0.0.0', port=1112)
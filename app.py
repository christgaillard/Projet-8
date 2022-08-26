import os
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import webapp.functions as wp  #import des fonctions
import pickle
import json
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify


app = Flask(__name__,) #initialisation de Flask
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['DATA_FOLDER'] = os.path.join(app.root_path, 'static/database')
modelpath = os.path.join(app.root_path, 'model/FPN-efficientnetb7-tensor')
weightspath = os.path.join(app.root_path, 'model/FPN_pretrained_eficiannet_ecc_jaccard_loss.h5')
# page d'accueil de l'app
@app.route('/')
def index():
    '''
    liste les images présentes dans le dossier database.
    :return:
    '''
    indexes = list(set(["_".join(path.split('_')[:3]) for path in os.listdir(app.config['DATA_FOLDER'])]))
    pathindexes = [os.path.join(app.config['DATA_FOLDER'], path) for path in indexes]

    return render_template('index.html', len=len(indexes), imglist=indexes, pathlist=pathindexes)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

'''

'''
@app.route('/analyse_file', methods=['GET', 'POST'])
def analyse_file():
    '''
    reçoit le nom de l'image, prédit le masque
    créer une image complète des 3 images
    :return: path to image.
    '''
    if request.method == 'POST':
        chem = os.path.join(app.root_path, 'static/database/')
        name =  request.form['img']
        imgname = name + '_leftImg8bit.png'
        maskname = name + '_gtFine_labelIds.png'
        imgpath = chem+imgname
        maskpath = chem+maskname

        val_data = wp.DataGen(imgpath,maskpath,batch_size=1, shuffle=False,augmentation=None)
        valid_dataloader = wp.DataGen(imgpath, maskpath)
        model = wp.load_model(modelpath,weightspath)
        img = image.img_to_array(image.load_img(imgpath, target_size=(160, 224))) / 255.
        gt_mask = image.img_to_array(image.load_img(maskpath, target_size=(160, 224), color_mode="grayscale"))
        _image = valid_dataloader[0][0]
        pr_mask = model.predict(_image, verbose=1)
        pr_mask = np.squeeze(pr_mask)
        pr_mask = cv2.resize(pr_mask, (224,160))
        pr_mask = np.argmax(pr_mask, axis=2)
        newimage = wp.visualize(
                image=img.squeeze(),
                gt_mask=gt_mask.squeeze(),
                pr_mask=wp.denormalize(pr_mask)
            )
        return render_template('register.html',filename='database/'+imgname, maskname='database/'+ maskname, newimage='test/'+newimage)

@app.route('/analyse_api', methods=['GET', 'POST'])
def analyse_api():
    '''
    Reussoit l'image serialiser,
     load le model et predit le mask à partir de l'image.
    :return:  le mask serialiser.
    '''
    response_data = []
    mondict = {}
    request_data = request.get_json()
    data = request_data[0]['body']
    img = pickle.loads(json.loads(data).encode('latin-1'))
    model = wp.load_model(modelpath,weightspath)
    pr_mask = model.predict(img, verbose=1)
    pr_mask = np.squeeze(pr_mask)
    pr_mask = cv2.resize(pr_mask, (224, 160))
    pr_mask = np.argmax(pr_mask, axis=2)
    mask = json.dumps(pickle.dumps(pr_mask).decode('latin-1'))
    mondict['pred']=mask
    response_data.append(mondict)
    return jsonify(response_data)

if __name__ == '__main__':
   app.run()

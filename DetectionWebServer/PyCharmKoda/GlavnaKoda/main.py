import os  # os library

import torch
from pip._vendor.chardet import detect

from app import app  # app object
from flask import Flask, flash, request, redirect, render_template, send_from_directory  # app flask
from werkzeug.utils import secure_filename
import base64
from flask import send_file
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

checkpoint = './BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model']
model = model.to(device)
model.eval()

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])  # dovoljene končnice
PUBLIC_DIR = app.config['UPLOAD_FOLDER']


def allowed_file(filename):  # preveri če je končnica pravilna
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    image_names = [f for f in os.listdir(PUBLIC_DIR) if
                   (f.endswith('png') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('gif'))]

    print(image_names)
    return render_template("upload.html", image_names=image_names)


@app.route('/<filename>')
def send_image(filename):
    return send_from_directory(PUBLIC_DIR, filename)


@app.route('/androidUpload', methods=['GET', 'POST'])
def upload_image():
    encoded_img = request.form['base64']  # 'base64' is the name of the parameter used to post image file
    filename = request.form['ImageName']  # 'ImageName' is name of the parameter used to post image name
    img_data = base64.b64decode(encoded_img)  # decode base64 string back to image
    image_path_name = os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename
    with open(image_path_name, 'wb') as f:  # "w"rite and "b"inary = wb
        f.write(img_data)
    original_image = Image.open(image_path_name, mode='r')
    original_image = original_image.convert('RGB')
    detect(original_image, image_path_name, min_score=0.2, max_overlap=0.5, top_k=200)
    return send_file(image_path_name)


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':  # če ima post zahteva datoteko
        if 'file' not in request.files:
            flash('No file part')  # izpiše če ni
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':  # če ni izbrane datoteke za nalaganje
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):  # če je datoteka izbrana in ustreza
            filename = secure_filename(file.filename)  # pretvori datoteko
            img_path = os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # shranjevanje datoteke
            flash('Datoteka se je uspesno nalozila')  # feedback
            original_image = Image.open(img_path, mode='r')
            original_image = original_image.convert('RGB')
            detect(original_image, img_path, min_score=0.2, max_overlap=0.5, top_k=200)
            return redirect('/response?file=' + filename)
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


@app.route('/response')
def return_file():
    return send_file(app.config['UPLOAD_FOLDER'] + request.args.get('file'))


def detect(original_image, filename, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param filename: name of image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background']
    # in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./arial.ttf", 15)

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    annotated_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # Shranjevanje datoteke
    # return redirect('/response?file=' + annotated_image)
    # return annotated_image


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

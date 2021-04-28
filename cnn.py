
import cv2
import dlib
import sys
from os.path import join



def load_model(model_path, caffemodel, prototxt):
    caffemodel_path = join(model_path, caffemodel)
    prototxt_path = join(model_path, prototxt)
    model = cv2.dnn.readNet(prototxt_path, caffemodel_path)

    return model


def predict(model, img, height, width):
    face_blob = cv2.dnn.blobFromImage(img, 1.0, (height, width), (0.485, 0.456, 0.406))
    model.setInput(face_blob)
    predictions = model.forward()
    class_num = predictions[0].argmax()
    confidence = predictions[0][class_num]

    return class_num, confidence



detector = dlib.get_frontal_face_detector()
font, fontScale, fontColor, lineType = cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2

input_height = 224
input_width = 224

# load gender model
gender_model_path = 'models/gender'
gender_caffemodel = 'gender.caffemodel'
gender_prototxt = 'gender.prototxt'
gender_model = load_model(gender_model_path, gender_caffemodel, gender_prototxt)

# load age model
age_model_path = 'models/age'
age_caffemodel = 'dex_chalearn_iccv2015.caffemodel'
age_prototxt = 'age.prototxt'
age_model = load_model(age_model_path, age_caffemodel, age_prototxt)


for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    if img is not None:
            imgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = detector(imgr, 1)
    for d in dets:
     left = int(0.6 * d.left())     # + 40% margin
     top = int(0.6 * d.top())       # + 40% margin
     right = int(1.4 * d.right())   # + 40% margin
     bottom = int(1.4 * d.bottom()) # + 40% margin
     face_segm = imgr[top:bottom, left:right]
     gender, gender_confidence = predict(gender_model, face_segm, input_height, input_width)
     age, age_confidence = predict(age_model, face_segm, input_height, input_width)
     gender = 'man' if gender == 1 else 'woman'
     text = '{} ({:.2f}%) {} ({:.2f}%)'.format(gender, gender_confidence*100, age-5, age_confidence*100)
     cv2.putText(img, text, (d.left(), d.top() - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
     cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), fontColor, 2)

    cv2.imshow('frame', img )
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break



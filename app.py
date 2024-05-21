from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from paddleocr import PaddleOCR
from ultralytics import YOLO
import os
import cv2
import uuid
from flask_cors import CORS  
import shutil


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize the OCR and YOLO models
ocr = PaddleOCR(use_angle_cls=True, lang='en')
model = YOLO("best.pt")
# model.to('cuda')

app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app
api = Api(app)

parser = reqparse.RequestParser()
# Add the arguments that you expect in the dictionary
parser.add_argument('images', type=list, location='json')
parser.add_argument('marathon_name', type=str, location='json')

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

response_json = []

def detect_bib_number(image):
    results = model.predict(image, imgsz=2016)
    return results

def extract_text(result):
    first_elements = [text[1] for sublist in result for text in sublist]
    return first_elements

def remove_non_alphanumeric(text):
    return ''.join(char for char in text if char.isalnum())

def filter_wanted_text(extracted_text, text_confidence_score):
    # Remove text with low confidence score
    text = [x for x in extracted_text if x[1] > text_confidence_score]
    # Get only the text
    text = [x[0] for x in text]
    # Remove non-alphanumeric characters
    text = [remove_non_alphanumeric(x) for x in text]
    # Remove text that are not digit
    text = [x for x in text if x.isdigit()]
    return text[0] if text else None


@app.route('/upload', methods=['POST'])
def upload_images():
    # Parse the arguments from the incoming HTTP request
    args = parser.parse_args()

    # Access the values of the arguments
    filepaths = args['images']
    marathon_name = args['marathon_name']
    text_confidence_score = 0.3  # Default confidence score

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        new_filepath = os.path.join(UPLOAD_FOLDER, filename)
        shutil.copy(filepath, new_filepath)
        image = cv2.imread(new_filepath)
        print(f'Processing image: {new_filepath}...')
        results = detect_bib_number(new_filepath)
        print(f'Number of detected bounding boxes: {len(results)}')

        bib_numbers = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = image[y1:y2, x1:x2, :]
                bib_number = ocr.ocr(roi, cls=True)

                try:
                    extracted_text = extract_text(bib_number)
                    filtered_text = filter_wanted_text(extracted_text, text_confidence_score)
                    if filtered_text:
                        bib_numbers.append(filtered_text)
                        bib_numbers = list(set(bib_numbers))
                except:
                    pass

        response_json.append({'marathon_name': marathon_name, 'image_path': filename, 'detected_numbers': bib_numbers})
        print(response_json)

    return jsonify({'message': 'Images uploaded and processed successfully'})

@app.route('/search', methods=['GET'])
def search_bib_number():
    desired_bib_number = request.args.get('desired_bib_number')
    marathon_name = request.args.get('marathon_name')
    results = []

    for record in response_json:
        if desired_bib_number in record['detected_numbers'] and record['marathon_name'] == marathon_name:
            results.append(record['image_path'])

    return jsonify({'desired_bib_number': desired_bib_number, 'marathon_name': marathon_name, 'image_paths': results})

if __name__ == '__main__':
    app.run(port=5006)

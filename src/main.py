# pip freeze | xargs pip uninstall -y
# pip freeze > requirements.txt
# * -- Imports
import os
import sys

import cv2
import numpy as np
from skimage.filters import threshold_local
from skimage import measure
import imutils
import tensorflow.compat.v1 as tf
from PIL import Image


# * -- Variables
title = "LPR"
PATH = os.path.realpath(__file__)  # ? Directory path
ASSETS_PATH = PATH + "\\..\\assets\\samples\\images"  # ? Assets path
OUTPUT_FOLDER_PATH = PATH + "\\..\\output"  # ? Output folder path
MODEL_FILE_PATH = PATH + "\\..\\model\\binary_128_0.50_ver3.pb"  # ? Model folder path
LABEL_FILE_PATH = PATH + \
    "\\..\\model\\binary_128_0.50_labels_ver2.txt"  # ? Label folder path
tf.disable_v2_behavior()  # ? Disable tensorflow v2

# * -- Functions


def clearConsole() -> None:  # ? Clear console
    os.system("cls" if os.name == "nt" else "clear")


def checkPaths() -> None:
    print(f"[{title}#checkPaths] PATH: ", PATH)
    print(f"[{title}#checkPaths] ASSETS_PATH: ", ASSETS_PATH)
    print(f"[{title}#checkPaths] OUTPUT_FOLDER_PATH: ", OUTPUT_FOLDER_PATH)
    print(f"[{title}#checkPaths] MODEL_FILE_PATH: ", MODEL_FILE_PATH)
    print(f"[{title}#checkPaths] LABEL_FILE_PATH: ", LABEL_FILE_PATH)


def sort_cont(character_contours):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                      key=lambda b: b[1][i], reverse=False))
    return character_contours


def segment_chars(plate_img, fixed_width):
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2]

    T = threshold_local(V, 29, offset=15, method='gaussian')

    thresh = (V > T).astype('uint8') * 255

    thresh = cv2.bitwise_not(thresh)

    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    labels = measure.label(thresh, neighbors=8, background=0)

    charCandidates = np.zeros(thresh.shape, dtype='uint8')

    characters = []
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255

        cnts = cv2.findContours(
            labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

            aspectRatio = boxW / float(boxH)
            solidity = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio = boxH / float(plate_img.shape[0])

            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14:
                hull = cv2.convexHull(c)

                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    _, contours, hier = cv2.findContours(
        charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contours = sort_cont(contours)
        addPixel = 4
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if y > addPixel:
                y = y - addPixel
            else:
                y = 0
            if x > addPixel:
                x = x - addPixel
            else:
                x = 0
            temp = bgr_thresh[y:y + h +
                              (addPixel * 2), x:x + w + (addPixel * 2)]

            characters.append(temp)
        return characters
    else:
        return None


def lpr() -> None:
    print(f"[{title}#lpr] init")

    try:
        findPlate = PlateFinder()
        model = NeuralNetwork()

        files = os.listdir(ASSETS_PATH)
        print(f"[{title}#lpr] files: ", files)

        print("= = = = = = = = = = = = = = = = = = = = =")
        for file in files:
            try:
                print("- - - - - - - - - - - - - - - - - - - - -")
                print(f"[{title}#lpr] file: ", file)

                filepath = os.path.join(ASSETS_PATH, file)
                print(f"[{title}#lpr] filepath: ", filepath)

                possible_plates = findPlate.find_possible_plates(filepath)
                print(f"[{title}#lpr] possible_plates", possible_plates)
                if possible_plates is not None:
                    print(f"[{title}#lpr] bbb")
                    for i, _ in enumerate(possible_plates):
                        print(f"[{title}#lpr] ccc")
                        chars_on_plate = findPlate.char_on_plate[i]
                        print(f"[{title}#lpr] ddd")
                        recognized_plate, _ = model.label_image_list(
                            chars_on_plate, imageSizeOuput=128)
                        print(f"[{title}#lpr] eee")

                        output = recognized_plate
                        print(f"[{title}#lpr] ({file}) output: ", output)

                        output_destination = OUTPUT_FOLDER_PATH + \
                            f"/{file}.txt"
                        print(f"[{title}#lpr] output_destination: ",
                              output_destination)

                        outputFolderExists = os.path.exists(OUTPUT_FOLDER_PATH)
                        print(f"[{title}#lpr] outputFolderExists: ",
                              outputFolderExists)

                        if not outputFolderExists:
                            print(f"[{title}#lpr] create output folder")
                            os.mkdir(OUTPUT_FOLDER_PATH)

                        with open(output_destination, "w", encoding="utf-8") as result:
                            result.write(output)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(exc_type, fname, exc_tb.tb_lineno)
                print(
                    f"[{title}#lpr] ({file}) error (line: {exc_tb.tb_lineno}): ", e)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print(exc_type, fname, exc_tb.tb_lineno)
        print(
            f"[{title}#lpr] error (line: {exc_tb.tb_lineno}): ", e)
    print("= = = = = = = = = = = = = = = = = = = = =")


# * -- Classes

class PlateFinder:
    def __init__(self):
        self.min_area = 4500
        self.max_area = 30000

        self.element_structure = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(22, 3))

    def preprocess(self, input_img):
        images = np.asarray(Image.open(input_img))
        imgBlurred = cv2.GaussianBlur(
            images, (7, 7), 0)
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        ret2, threshold_img = cv2.threshold(
            sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        element = self.element_structure
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE,
                         kernel=element, dst=morph_n_thresholded_img)
        return morph_n_thresholded_img

    def extract_contours(self, after_preprocess):
        contours, _ = cv2.findContours(after_preprocess, mode=cv2.RETR_EXTERNAL,
                                          method=cv2.CHAIN_APPROX_NONE)
        return contours

    def clean_plate(self, plate):
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            areas = [cv2.contourArea(c) for c in contours]

            max_index = np.argmax(areas)

            max_cnt = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            rect = cv2.minAreaRect(max_cnt)
            rotatedPlate = plate
            if not self.ratioCheck(max_cntArea, rotatedPlate.shape[1], rotatedPlate.shape[0]):
                return plate, False, None
            return rotatedPlate, True, [x, y, w, h]
        else:
            return plate, False, None

    def check_plate(self, input_img, contour):
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            x, y, w, h = cv2.boundingRect(contour)
            after_validation_img = input_img[y:y + h, x:x + w]
            after_clean_plate_img, plateFound, coordinates = self.clean_plate(
                after_validation_img)
            if plateFound:
                characters_on_plate = self.find_characters_on_plate(
                    after_clean_plate_img)
                if (characters_on_plate is not None and len(characters_on_plate) == 8):
                    x1, y1, w1, h1 = coordinates
                    coordinates = x1 + x, y1 + y
                    after_check_plate_img = after_clean_plate_img
                    return after_check_plate_img, characters_on_plate, coordinates
        return None, None, None

    def find_possible_plates(self, input_img):
        plates = []
        self.char_on_plate = []
        self.corresponding_area = []

        self.after_preprocess = self.preprocess(input_img)
        possible_plate_contours = self.extract_contours(self.after_preprocess)

        for cnts in possible_plate_contours:
            plate, characters_on_plate, coordinates = self.check_plate(
                input_img, cnts)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(characters_on_plate)
                self.corresponding_area.append(coordinates)
                print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        if (len(plates) > 0):
            return plates
        else:
            return None

    def find_characters_on_plate(self, plate):

        charactersFound = segment_chars(plate, 400)
        if charactersFound:
            return charactersFound

    def ratioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 3
        ratioMax = 6

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def preRatioCheck(self, area, width, height):
        min = self.min_area
        max = self.max_area

        ratioMin = 2.5
        ratioMax = 7

        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio

        if (area < min or area > max) or (ratio < ratioMin or ratio > ratioMax):
            return False
        return True

    def validateRatio(self, rect):
        (x, y), (width, height), rect_angle = rect

        if (width > height):
            angle = -rect_angle
        else:
            angle = 90 + rect_angle

        if angle > 15:
            return False
        if (height == 0 or width == 0):
            return False

        area = width * height
        if not self.preRatioCheck(area, width, height):
            return False
        else:
            return True


class NeuralNetwork:
    def __init__(self):
        global MODEL_FILE_PATH, LABEL_FILE_PATH
        self.model_file = MODEL_FILE_PATH
        self.label_file = LABEL_FILE_PATH
        self.label = self.load_label(self.label_file)
        self.graph = self.load_graph(self.model_file)
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, modelFile):
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        with open(modelFile, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    def load_label(self, labelFile):
        label = []
        proto_as_ascii_lines = tf.compat.v2.io.gfile.GFile(
            labelFile).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def convert_tensor(self, image, imageSizeOuput):
        image = cv2.resize(image, dsize=(
            imageSizeOuput, imageSizeOuput), interpolation=cv2.INTER_CUBIC)
        np_image_data = np.asarray(image)
        np_image_data = cv2.normalize(np_image_data.astype(
            'float'), None, -0.5, .5, cv2.NORM_MINMAX)
        np_final = np.expand_dims(np_image_data, axis=0)
        return np_final

    def label_image(self, tensor):

        input_name = "import/input"
        output_name = "import/final_result"

        input_operation = self.graph.get_operation_by_name(input_name)
        output_operation = self.graph.get_operation_by_name(output_name)

        results = self.sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: tensor})
        results = np.squeeze(results)
        labels = self.label
        top = results.argsort()[-1:][::-1]
        return labels[top[0]]

    def label_image_list(self, listImages, imageSizeOuput):
        plate = ""
        for img in listImages:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            plate = plate + \
                self.label_image(self.convert_tensor(img, imageSizeOuput))
        return plate, len(plate)


#! Main
def main() -> None:
    # clearConsole()

    print(f"[{title}#main] init")

    checkPaths()
    lpr()

    print(f"[{title}#main] exit")


if __name__ == "__main__":
    main()
    # try:
    #     main()
    # except Exception as e:
    #     exc_type, exc_obj, exc_tb = sys.exc_info()
    #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    #     print(exc_type, fname, exc_tb.tb_lineno)

import sys
import subprocess
import cv2
import os
import pytesseract
import numpy as np
import pyttsx3
from playsound import playsound
import shutil
from os import listdir

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()


def preprocess(img):
    # contrast
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)

    norm_img = diff_img.copy()  # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255,
                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # contrast
    return thr_img


# modify the extracted coordinates to apply Tesseract
def extractCord():
    with open("result/img_result.txt") as f:
        text = f.read()

    cord = []
    # path = 'F:\Abdallah\CompuerScience\DeepLearning\CRAFT-pytorch-master\captures'
    for count, data in enumerate(text.splitlines()):
        data = data.split(',')
        if(count & 1):
            continue
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = int(data[0]), int(data[1]), int(data[2]), int(
                data[3]), int(data[4]), int(data[5]), int(data[6]), int(data[7])
            startX = min([x1, x2, x3, x4])
            startY = min([y1, y2, y3, y4])
            endX = max([x1, x2, x3, x4])
            endY = max([y1, y2, y3, y4])
            cord.append([startX, startY, endX, endY])
            # ROI = img[startY:endY, startX:endX]
            # cv2.imwrite(os.path.join(path, 'ROI_{}.png'.format(ROI_number)), ROI)
            # ROI_number += 1

    # sort coordinates according to the startY
    cord = sorted(cord, key=lambda x: int(x[1]))
    return cord


# check if the detected boxes on the same row
def isOnSameLine(boxOne, boxTwo):
    boxOneStartY = boxOne[1]
    boxOneEndY = boxOne[3]
    boxTwoStartY = boxTwo[1]
    boxTwoEndY = boxTwo[3]
    if((boxTwoStartY <= boxOneEndY and boxTwoStartY >= boxOneStartY)
       or (boxTwoEndY <= boxOneEndY and boxTwoEndY >= boxOneStartY)
       or (boxTwoEndY >= boxOneEndY and boxTwoStartY <= boxOneStartY)):
        return True
    else:
        return False


# Sort boxes as groups
def sortGroups(cord):
    # list of indexes
    temp = []
    i = 0
    sorted_box_group = cord
    # sort the detected boxes
    while i < len(cord):
        for j in range(i + 1, len(cord)):
            if(isOnSameLine(cord[i], cord[j])):
                if i not in temp:
                    temp.append(i)
                if j not in temp:
                    temp.append(j)
            # append temp with i if the current box (i) is not on the same line with any other box
            if len(temp) == 0:
                temp.append(i)
        # put boxes on same line into lined_box_group array
        lined_box_group = [cord[i] for i in temp]
        # sort boxes by startX value
        lined_box_group = sorted(lined_box_group, key=lambda x: int(x[0]))
        # copy sorted boxes on same line into sorted_box_group
        if len(lined_box_group) > 0:
            sorted_box_group[i:temp[-1]+1] = lined_box_group
        # print(sorted_box_group)
        # skip to the index of the box that is not on the same line
        if len(temp) > 0:
            i = temp[-1] + 1
        else:
            break
        # clear list of indexes
        temp = []

    return sorted_box_group


# Extract ROI
def extractROI(img, sorted_box_group):
    ROI_number = 0
    path = 'CRAFT-Tesseract/captures'
    for box in sorted_box_group:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        ROI = img[y1:y2, x1:x2]
        cv2.imwrite(os.path.join(path, 'ROI_{}.png'.format(ROI_number)), ROI)
        ROI_number += 1


# Assemble Text
def assembleText():
    final_txt = []  # Result
    cnf = "--psm 8 --oem 3"  # modyifing the default config of tesseract
    path = "CRAFT-Tesseract/captures"
    dirFiles = os.listdir(path)

    # Assemble the images into a string
    for cnt in range(len(dirFiles)):
        input_path = os.path.join(path, "ROI_{}.png".format(cnt))
        cur_img = cv2.imread(input_path)
        word = pytesseract.image_to_string(
            cur_img, lang='eng', config=cnf)

        cleanup_text(word)
        final_txt.append(word)

    final_txt = ' '.join(final_txt).replace('\n', '').split()
    final_txt = " ".join(list(map(str, final_txt)))
    return final_txt


def text_to_voice(txt):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    engine.save_to_file(txt, 'Results/text_to_voice.wav')
    engine.runAndWait()


def saveText(ocr_result):
    with open("Results/text.txt", "w") as text_file:
        text_file.write(ocr_result)


def main():
    img = cv2.imread("CRAFT-Tesseract/figures/sample1.jpg")
    img = preprocess(img)
    cv2.imwrite("CRAFT-Tesseract/figures/sample1.jpg", img)

    subprocess.call([sys.executable, 'CRAFT-Tesseract/test.py',
                    '--trained_model=CRAFT-Tesseract/craft_mlt_25k.pth', '--test_folder=CRAFT-Tesseract/figures'])

    cord = extractCord()
    sortedBoxes = sortGroups(cord)
    extractROI(img, sortedBoxes)
    final_txt = assembleText()
    saveText(final_txt)
    text_to_voice(final_txt)
    # playsound('Results/text_to_voice.wav')
    shutil.rmtree(r'CRAFT-Tesseract/captures')
    os.mkdir(r"CRAFT-Tesseract/captures")


if __name__ == "__main__":
    main()

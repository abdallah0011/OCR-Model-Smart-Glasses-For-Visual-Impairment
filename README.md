# OCR Model (Smart Glasses For Visual Impairment)

This project aims to help individuals who suffer from visual impairments to go about their daily acitivites, a part of this project was OCR model which can provide
the needed assistance to the user.<br>

OCR is an abbreviation of optical character recognition method , it refers to extracting text from every possible image, be it a standard printed page or from a book,
a document, or a random image.<br>

I did integrate the follwoing two models that can help me to complete this model and acheive acceptable accuracy:

- CRAFT (Text Detection).
- Tesseract (Text Recognition).

The user should expect a voice output coming from the glasses, but for testing purposes the result is saved in <b>"Results"</b> folder.

---

## Demo-Preview

![Imgur](https://i.imgur.com/tkJgrq6.jpg)
![Imgur](https://i.imgur.com/0LdoAvZ.jpg)

## Table of Contents

- [Project Title](#OCR-Model)
- [Table of contents](#table-of-contents)
- [Demo-Preview](#demo-preview)
- [Install dependencies](#install-dependecies)
- [Installation](#installation)
- [Development](#development)
- [Resources](#resources)

## Install dependencies

[(Back to top)](#install-dependecies)

- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- check requiremtns.txt
- pytesseract
- opencv
- pyttsx3
- playsound

## Installation

[(Back to top)](#table-of-contents)

Firstly, Locate figures folder in "CRAFT-Tesseract" folder, there where you would put the testing image, and it should be named "sample1".I did also include some test images that can be found in test folder.<br>

## Development

## Resources

https://paperswithcode.com/paper/character-region-awareness-for-text-detection

https://github.com/clovaai/CRAFT-pytorch

https://pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/

https://ai-facets.org/tesseract-ocr-best-practices/

https://francescopochetti.com/easyocr-vs-tesseract-vs-amazon-textract-an-ocr-engine-comparison/

https://towardsdatascience.com/pytorch-scene-text-detection-and-recognition-by-craft-and-a-four-stage-network-ec814d39db05

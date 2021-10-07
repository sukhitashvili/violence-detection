import argparse

import cv2

from model import Model


def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection")
    parser.add_argument('--image-path', type=str,
                        default='./data/7.jpg',
                        help='path to your image')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = argument_parser()
    model = Model()
    image = cv2.imread(args.image_path)
    prediction = model.predict(image=image)
    label = prediction['label']
    conf_score = prediction['confidence']
    print('predicted label: ', label, ' with confidence: ', conf_score)
    cv2.imshow(label.title(), image)
    cv2.waitKey(0)

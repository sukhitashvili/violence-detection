if __name__ == '__main__':
    import cv2
    from processor import Model
    model = Model()
    image_path = './data/7.jpg'
    image = cv2.imread(image_path)
    label = model.predict(image=image)
    print('pred label: ', label)
    cv2.imshow(label, image)
    cv2.waitKey(0)
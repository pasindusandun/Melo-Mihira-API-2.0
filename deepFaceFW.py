from deepface import DeepFace



def predict_age(path):
    # path = "./Images/wsx.jpg"
    objs = DeepFace.analyze(img_path = path, actions = ['age'] )

    print(objs)
    return objs

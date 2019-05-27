import face_model
import argparse
import cv2


parser = argparse.ArgumentParser(description='Face recognition using ArcFace encoding')
parser.add_argument('--mode', default='recognize', help='Image size')
parser.add_argument('--face_filter_size', default='0,0', help='Image size')
parser.add_argument('--image-size', default='112,112', help='Image size')
parser.add_argument('--ga-model', default='', help='path to ga model')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--model', default='./model-r100-ii/model,0', help='path to arcface model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--threshold', default=0.9, type=float, help='distance threshold to compare images')

args = parser.parse_args()

model = face_model.FaceModel(args)

frame = cv2.imread('frames/100.jpg')
faces, bboxes = model.get_input(frame)

print('faces len :', len(faces))

for face,bbox in zip(faces, bboxes):
    encoding = model.get_feature(face)
    print('Encoding :', encoding)

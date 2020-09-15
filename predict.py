import argparse
from mymodel import MyModel

#SAMPLE: ython predict.py flowers/test/1/image_06743.jpg checkpoint.pth --gpu

parser = argparse.ArgumentParser(description='Flower Prediction')

parser.add_argument('image_path', action="store", type=str, help='flower image filepath')
parser.add_argument('checkpoint', action="store", type=str, help='model checkpoint file, eg: checkpoint.pth')
parser.add_argument('--top_k', action="store", type=int, help='topk return', default=5)
parser.add_argument('--category_names', action="store", type=str, help='category name json file', default="cat_to_name.json")
parser.add_argument('--gpu', action='store_true', help='use gpu, if not specified, will use cpu', default=False)

args = parser.parse_args()

mm = MyModel(category_name_filepath=args.category_names, use_gpu=args.gpu)
model, optimizer = mm.load_checkpoint(args.checkpoint)

answer = mm.predict(args.image_path, model, topk=args.top_k)

print("Predictions: ")
print(answer)

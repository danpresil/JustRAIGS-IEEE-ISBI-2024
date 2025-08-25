from PIL import Image

from helper import inference_tasks
from model import predict


def run():
    for jpg_image_file_name, save_prediction in inference_tasks():
        image = Image.open(jpg_image_file_name)
        is_referable_glaucoma, likelihood, features = predict(image)
        save_prediction(is_referable_glaucoma, likelihood, features)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

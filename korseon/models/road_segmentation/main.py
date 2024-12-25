import os, argparse
from io import BytesIO
from model import get_fast_scnn
from visualise import get_color_pallete

from PIL import Image
import torch
from torchvision import transforms

import sys, os, base64, time
sys.path.append(os.path.abspath("./modules"))
from lightweight_communication_bridge import LCB

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./img.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=True)

args = parser.parse_args()

# Run object detection and return response
def onMessage(payload: dict[str, object]) -> None:
    print('received message')
    imgBytes = base64.b64decode(payload['image'])
    image = Image.open(BytesIO(imgBytes)).convert('RGB')
    print('running detection')
    result = runDetection(image)
    print('sending result')
    lcb.send('asp2-korseon.korseon-main', {'source':'asp2-korseon.road-segmentation', 'code': 'result', 'image': base64.b64encode(result).decode('utf-8')})


# Set up LCB and register listener
lcb = LCB(onMessage, host='host.docker.internal', port='8078')

def runDetection(image: Image.Image):
    device = torch.device("cpu")

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).to(device)
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset).convert('RGB')
    with BytesIO() as buffer:
        mask.save(buffer, format="JPEG")
        return buffer.getvalue()

def main():
    while True:
        time.sleep(1)

main()
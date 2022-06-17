import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *
import time
from pathlib import Path

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    # print("*********correct=", correct.size())
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 6]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

def test(cfg,
         data,
         weights=None,
         batch_size=32,
         img_size=540,
         iou_thres=0.6,
         conf_thres=0.001,
         nms_thres=0.4,
         save_json=False,
         model=None,
         single_cls=False,  # treat as single-class dataset
         save_hybrid=False): # save label+prediction hybrid results to *.txt

    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device)
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min([os.cpu_count(), batch_size, 16]),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        imgs = imgs.float()  # uint8 to fp16/32
        # imgs /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = imgs.shape

        # Plot images with bounding boxes
        if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # time1= time.time()
        # Run model
        inf_out, train_out = model(imgs) # inference and training outputs
        # print("$$$$$@@@@@Time",(time.time()-time1))
        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
        # print("*******output=",output)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1
            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            clip_coords(predn, (height, width))
            # scale_coords(imgs[si].shape[1:],predn[:, :4],shapes[si])
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                # scale_coords(imgs[si].shape[1:], tbox, shapes[si])
                # clip_coords(tbox, (height, width))
                # tbox[:, [0, 2]] *= width
                # tbox[:, [1, 3]] *= height
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))  # (correct, conf, pcls, tcls)
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, f1, ap, ap_class = ap_per_class_re(*stats)
        # ap50_95 = ap.mean(1)
        print('*****ap[]=', ap)
        print("@@@@@@@@@ap@.5:0.95=", (ap.mean(1)).mean())
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    # if verbose and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # # Save JSON
    # if save_json and map and len(jdict):
    #     try:
    #         imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
    #         with open('results.json', 'w') as file:
    #             json.dump(jdict, file)

    #         from pycocotools.coco import COCO
    #         from pycocotools.cocoeval import COCOeval

    #         # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #         cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
    #         cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

    #         cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    #         cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
    #         cocoEval.evaluate()
    #         cocoEval.accumulate()
    #         cocoEval.summarize()
    #         map = cocoEval.stats[1]  # update mAP to pycocotools mAP
    #     except:
    #         print('WARNING: missing dependency pycocotools from requirements.txt. Can not compute official COCO mAP.')

    # # Return results
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    # return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps
    return mp, mr, map, mf1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.iou_thres,
             opt.conf_thres,
             opt.nms_thres,
             opt.save_json)

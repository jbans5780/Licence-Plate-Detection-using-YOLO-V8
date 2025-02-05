# Ultralytics YOLO ð€, GPL-3.0 license

import hydra
import torch
import time

from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import RPi.GPIO as GPIO  # Imports the standard Raspberry Pi GPIO library
from time import sleep   # Imports sleep (aka wait or pause) into the program


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string, []
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        labels = []
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                labels.append(label)
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)
        print("HELLO")
        return log_string, labels

def cleanup():
    out1 = 17
    out2 = 18
    out3 = 27
    out4 = 22
    GPIO.output( out1, GPIO.LOW )
    GPIO.output( out2, GPIO.LOW )
    GPIO.output( out3, GPIO.LOW )
    GPIO.output( out4, GPIO.LOW )
    GPIO.cleanup()
    
def moveMotor():
    out1 = 17
    out2 = 18
    out3 = 27
    out4 = 22

    step_sleep = 0.01

    step_count = 200

    GPIO.setmode( GPIO.BCM )
    GPIO.setup( out1, GPIO.OUT )
    GPIO.setup( out2, GPIO.OUT )
    GPIO.setup( out3, GPIO.OUT )
    GPIO.setup( out4, GPIO.OUT )

    GPIO.output( out1, GPIO.LOW )
    GPIO.output( out2, GPIO.LOW )
    GPIO.output( out3, GPIO.LOW )
    GPIO.output( out4, GPIO.LOW )
    
    try:
        i=0
    
        for i in range(step_count):
            if i%4==0:
                GPIO.output( out4, GPIO.HIGH )
                GPIO.output( out3, GPIO.LOW )
                
                GPIO.output( out2, GPIO.LOW )
                GPIO.output( out1, GPIO.LOW )
            elif i%4 == 1:
                
                GPIO.output( out4, GPIO.LOW )
                GPIO.output( out3, GPIO.LOW )
                GPIO.output( out2, GPIO.HIGH )
                GPIO.output( out1, GPIO.LOW )
            elif i%4 == 2:
                
                GPIO.output( out4, GPIO.LOW )
                GPIO.output( out3, GPIO.HIGH )
                GPIO.output( out2, GPIO.LOW )
                GPIO.output( out1, GPIO.LOW )
            elif i%4 == 3:
                print( "hello" )
                GPIO.output( out4, GPIO.LOW )
                GPIO.output( out3, GPIO.LOW )
                GPIO.output( out2, GPIO.LOW )
                GPIO.output( out1, GPIO.HIGH )
            
            time.sleep( step_sleep )

    except Exception as e:
        print(f"Error: {e}")
        cleanup()
        exit( 1 )

    cleanup()
    exit( 2 )


def prediction(labels):
    most_recent_entry = labels[-1].split()[0]
    #if lines:  # Ensure the file is not empty
        #most_recent_entry = lines[-1].strip()  # Get the last line and remove any trailing newlines
        #print("Most recent entry:", most_recent_entry)
    #else:
        #print("The file is empty.")
    print()
    print(most_recent_entry)
    if most_recent_entry == "metal" or most_recent_entry == "uht carton" or most_recent_entry == "plastic":
        print("metal")
        #servo.angle = 180
        #sleep(2)
        #servo.angle = 0
        ##move motor 180 degrees
        ## WAIT 2 SECONDS
        #function to return motor starting position to 0 degrees
        moveMotor()         # Resets the GPIO pins back to defaults
    else:
        print("not metal")
        #servo.angle = -180
        #sleep(2)
        #servo.angle = 0
        ##move motor -180 degrees
        # WAIT 2 SECOND
        ##function to return motor starting position to 0 degrees
        moveMotor()
        


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    print("1", flush=True)
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    c1, labels = predictor()
    print(labels, flush=True)
    prediction(labels) ### need to define this or need to figure out how to implement it


if __name__ == "__main__":
    print("HELLO WO", flush=True)
    predict()

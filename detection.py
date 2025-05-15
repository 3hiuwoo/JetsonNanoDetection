# demon focal length: 1170.7599, distance from camera: 89.5cm
import sys
import math
import argparse
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaDrawLine, cudaFont, Log

parser = argparse.ArgumentParser(description="Locate toy balls in camera/video/image.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument('input', type=str, default='csi://0', nargs='?', help='URI of the input stream')
parser.add_argument('output', type=str, default='display://0', nargs='?', help='URI of the output stream')
parser.add_argument('--threshold', type=float, default=0.25, help='detection threshold')
parser.add_argument('--overlay', type=str, default='lines', help='overlay method')
parser.add_argument('--nbox', type=int, default=1, help='number of boxes to show')
parser.add_argument('--calib1', type=float, nargs='+', default=None,
                    help='the distance between target and center and between center and camera')
parser.add_argument('--calib2', type=float, nargs='+', default=None,
                    help='focal length and the distance between center and camera')

def main():
    args = parser.parse_known_args()[0]

    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)
    # net = detectNet('ssd-mobilenet-v2', sys.argv, threshold=args.threshold)
    # the first two arguments should be modified if needed
    net = detectNet(model='../training/detection/ssd/models/balls_model/ssd-mobilenet.onnx',
                    labels='../training/detection/ssd/models/balls_model/labels.txt',
                    input_blob='input_0',
                    output_cvg='scores',
                    output_bbox='boxes',
                    threshold=args.threshold)

    width = input.GetWidth()
    height = input.GetHeight()
    lineWidth = net.GetLineWidth()

    font = cudaFont()
    fontHeight = font.GetSize()

    # calibration mode to find focal length using given
    # [distance from the camera to the origin, measured r value of the target]
    if args.calib1 and len(args.calib1) == 2:
        print('-'*10, 'Calibrating the camera for 10 iteration', '-'*10)
        rr, d = args.calib1[0], args.calib1[1]
        fs = []
        isCalib = True
    else:
        isCalib = False

    # using given [focal length, distance from the camera to the origin] to calculate the real r value of the target
    if args.calib2 and len(args.calib2) == 2:
        print('-'*10, 'Using focal length', '-'*10)
        ff, d = args.calib2[0], args.calib2[1]
        isFocal = True
        unit = 'cm'
    else:
        isFocal = False
        unit = 'pix'

    # streaming loop
    while True:
        img = input.Capture()

        if img is None:
            continue
        
        detections = net.Detect(img, overlay=args.overlay)

        # draw x and y axis
        cudaDrawLine(img, (width/2, 0), (width/2, height), font.Green, lineWidth)
        cudaDrawLine(img, (0, height/2), (width, height/2), font.Green, lineWidth)
        
        if len(detections) != 0:
            nbox = min(args.nbox, len(detections))
            for i in range(nbox):
                detection = detections[i] 
                # cudaDrawLine(img, (width/2, height/2), detection.Center, font.Green, lineWidth)

                # calculate polar coordinate in degree
                r, theta = toPolar((width/2, height/2), detection.Center)
                if isFocal:
                    r = calibrate(r, ff, d) # pix -> cm

                if isCalib:
                    f = r / (rr / d)
                    fs.append(f)

                else:    
                    # put the coordinate text on left top on the box
                    x = detection.Left
                    y = detection.Top - fontHeight
                    font.OverlayText(img, width, height, f'({r:.2f}{unit}, {theta:.2f}deg)', int(x), int(y), font.Green)

        # ouput processed image
        output.Render(img)

        output.SetStatus(f'FPS: {net.GetNetworkFPS():.0f}')

        if not input.IsStreaming() or not output.IsStreaming():
            break

        # average all focal length calculated by each frames to produce the result if is calibration mode
        if isCalib and len(fs) >= 10:
            print('-'*10, f'Calibration done with focal length: {sum(fs)/len(fs):.4f}', '-'*10)
            break


def toPolar(center, target):
    ''' cartesian coordinate -> polar coordinate
    Args:
        center: tuple of x, y value of the origin
        target: tuple of x, y value of target
    
    Returns:
        r: polar coordinate
        theta: polar coordinate
    '''
    nx = target[0] - center[0]
    ny = center[1] - target[1]
    r = math.sqrt(nx**2 + ny**2)
    theta = math.atan2(ny, nx)
    theta = theta / math.pi * 180
    return r, theta


def calibrate(r, f, d):
    ''' use the given focal length and the distance from the camera to the origin
        to calculate the real r of the detected object

    Args:
        r: r in pixels
        f: focal length
        d: distance from the camera to the origin

    Returns:
        rr: real r in cm(or other units same as d) 
    '''
    rr = r / f * d
    return rr

if __name__ == '__main__':
    main()

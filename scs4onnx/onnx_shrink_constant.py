#! /usr/bin/env python

import os
import sys
import shutil
from argparse import ArgumentParser
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Variable

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

def main():
    parser = ArgumentParser()
    parser.add_argument(
        'onnx_file_path',
        type=str,
        help='Input onnx file path.'
    )
    parser.add_argument(
        'mode',
        type=str,
        choices=[
            'shrink',
            'npy',
        ],
        default='shrink',
        help="\
            Constant Value Compression Mode. \
            shrink: Share constant values inside the model as much as possible. \
            npy: Outputs constant values used repeatedly in the model to an external file .npy. \
            Default: shrink"
    )
    args = parser.parse_args()


    work_file_path = shutil.copy(args.onnx_file_path, f'{os.path.splitext(args.onnx_file_path)[0]}_shrunken.onnx')

    graph = gs.import_onnx(onnx.load(work_file_path))






    graph.cleanup().toposort()

    new_model = None
    try:
        new_model = onnx.shape_inference.infer_shapes(gs.export_onnx(graph))
    except:
        new_model = gs.export_onnx(graph)
        print(
            f'{Color.YELLOW}WARNING:{Color.RESET} '+
            'The input shape of the next OP does not match the output shape. '+
            'Be sure to open the .onnx file to verify the certainty of the geometry.'
        )
    onnx.save(new_model, f'{work_file_path}')

if __name__ == '__main__':
    main()
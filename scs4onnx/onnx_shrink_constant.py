#! /usr/bin/env python

import os
import sys
import shutil
from pprint import pprint
from argparse import ArgumentParser
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Constant

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
            The model size is slightly larger because some shared constant values remain \
            inside the model, but performance is maximized. \
            npy: Outputs constant values used repeatedly in the model to an external file .npy. \
            Instead of the smallest model body size, the file loading overhead is greater. \
            Default: shrink"
    )
    args = parser.parse_args()

    # file existence check
    if not os.path.exists(args.onnx_file_path) or \
        not os.path.isfile(args.onnx_file_path) or \
        not os.path.splitext(args.onnx_file_path)[-1] == '.onnx':

        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The specified file (.onnx) does not exist. or not an onnx file. File: {args.onnx_file_path}'
        )
        sys.exit(1)

    # Working file generation
    work_file_path = shutil.copy(
        args.onnx_file_path,
        f'{os.path.splitext(args.onnx_file_path)[0]}_shrunken.onnx'
    )

    # Loading Graphs
    graph = gs.import_onnx(onnx.load(work_file_path))

    # Constant Value Extraction
    constants = {}
    for graph_node in graph.nodes:
        for graph_node_input in graph_node.inputs:
            if not isinstance(graph_node_input, Constant):
                continue
            if len(graph_node_input.shape) == 0:
                continue
            # if np.asarray(graph_node_input.values).size <= 1:
            #     continue
            if np.isscalar(graph_node_input.values):
                continue
            constants[graph_node_input.name] = graph_node_input
    print(
        f'{Color.GREEN}INFO:{Color.RESET} '+
        f'Number of constant values to be studied: {len(constants)}'
    )

    constants_list = list(constants.items())

    # Processing is performed only when the number of constant tensors to be processed is 2 or more
    if len(constants_list) >= 2:
        # Aggregate list generation
        aggregate_constants_name_list = {}
        aggregate_constants_value_list = {}
        aggregate_constants_matched_list = [False] * len(constants_list)

        for comparator_idx in range(0, len(constants_list)-1):
            if not aggregate_constants_matched_list[comparator_idx]:
                for comparison_dest_idx in range(comparator_idx+1, len(constants_list)):
                    if not aggregate_constants_matched_list[comparison_dest_idx]:
                        # constants_list[0] = Key, constants_list[1] = Value
                        if (constants_list[comparator_idx][1].values.shape == constants_list[comparison_dest_idx][1].values.shape) and \
                            (constants_list[comparator_idx][1].values.dtype == constants_list[comparison_dest_idx][1].values.dtype) and \
                            (constants_list[comparator_idx][1].values == constants_list[comparison_dest_idx][1].values).all():

                            aggregate_constants_matched_list[comparator_idx] = True
                            aggregate_constants_matched_list[comparison_dest_idx] = True
                            aggregate_constants_name_list.setdefault(constants_list[comparator_idx][0], [])
                            aggregate_constants_name_list[constants_list[comparator_idx][0]].append(constants_list[comparison_dest_idx][0])
                            aggregate_constants_value_list.setdefault(constants_list[comparator_idx][0], constants_list[comparator_idx][1])

    """
    aggregate_constants
    {'425': ['434', '5506'], '1646': ['1910', '2608', '2872', '3570', '3834'], '5550': ['4107']}
    """
    for layer_name_quoted_src, layer_name_quoted_dists in aggregate_constants_name_list.items():
        i = None
        if args.mode == 'npy':
            # Export constant values to a numpy file
            external_file_name = f"{os.path.splitext(args.onnx_file_path)[0]}_shrunken_exported_{layer_name_quoted_src.replace(':','_').replace(';','_').replace('/','_').replace(',','_')}.npy"
            np.save(
                external_file_name,
                aggregate_constants_value_list[layer_name_quoted_src].values
            )
            # Generate Inputs
            i = gs.Variable(
                name=external_file_name,
                dtype=aggregate_constants_value_list[layer_name_quoted_src].values.dtype,
                shape=aggregate_constants_value_list[layer_name_quoted_src].values.shape,
            )
            aggregate_constants_value_list[layer_name_quoted_src] = i
            graph.inputs.append(i)

        for graph_node_idx, graph_node in enumerate(graph.nodes):
            for input_idx, graph_node_input in enumerate(graph_node.inputs):
                for layer_name_quoted_dist in layer_name_quoted_dists:
                    if not graph_node_input.name == layer_name_quoted_src and graph_node_input.name == layer_name_quoted_dist:
                        graph.nodes[graph_node_idx].inputs[input_idx] = aggregate_constants_value_list[layer_name_quoted_src]
                    if args.mode == 'npy' and graph_node_input.name == layer_name_quoted_src:
                        graph.nodes[graph_node_idx].inputs[input_idx] = i

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
    print(f'{Color.GREEN}INFO:{Color.RESET} Results:')
    pprint(aggregate_constants_name_list)
    print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

if __name__ == '__main__':
    main()

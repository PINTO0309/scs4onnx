#! /usr/bin/env python

import os
import sys
import traceback
from pprint import pprint
from argparse import ArgumentParser
import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx_graphsurgeon.ir.tensor import Constant
from typing import Tuple, Optional, List

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

OP_TO_BE_EXCLUDED_FROM_DOWNCAST_PROCESSING = [
    "ConstantOfShape",
    "Concat",
]

def shrinking(
    input_onnx_file_path: Optional[str] = '',
    output_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    mode: Optional[str] = 'shrink',
    forced_extraction_op_names: List[str] = [],
    forced_extraction_constant_names: List[str] = [],
    disable_auto_downcast: Optional[bool] = False,
    non_verbose: Optional[bool] = False,
) -> Tuple[onnx.ModelProto, List[str]]:

    """
    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.\n\
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    mode: Optional[str]
        Constant Value Compression Mode.\n\
        'shrink': Share constant values inside the model as much as possible.\n\
            The model size is slightly larger because some shared constant values remain\n\
            inside the model, but performance is maximized.\n\
        'npy': Outputs constant values used repeatedly in the model to an external file .npy.\n\
            Instead of the smallest model body size, the file loading overhead is greater.\n\
        Default: shrink

    forced_extraction_op_names: List[str]
        Extracts the constant value of the specified OP name to .npy regardless of the mode specified.\n\
        Cannot be used with forced_extraction_constant_names at the same time.\n\
        e.g. ['aaa','bbb','ccc']

    forced_extraction_constant_names: List[str]
        Extracts the constant value of the specified Constant name to .npy regardless of the mode specified.\n\
        Cannot be used with forced_extraction_op_names at the same time.\n\
        e.g. ['aaa','bbb','ccc']

    disable_auto_downcast: Optional[bool]
        Disables automatic downcast processing from Float64 to Float32 and INT64 to INT32.\n\
        Try enabling it and re-running it if you encounter type-related errors.\n\
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n\
        Default: False

    Returns
    -------
    shrunken_graph: onnx.ModelProto
        Shrunken onnx ModelProto

    npy_file_paths: List[str]
        List of paths to externally output .npy files.
        An empty list is always returned when in 'shrink' mode.
    """

    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        graph = gs.import_onnx(onnx.load(input_onnx_file_path))
    else:
        graph = gs.import_onnx(onnx_graph)

    # Constant Value Extraction
    constants = {}
    for graph_node in graph.nodes:
        for graph_node_input in graph_node.inputs:
            if not isinstance(graph_node_input, Constant):
                continue
            if len(graph_node_input.shape) == 0:
                continue
            if np.isscalar(graph_node_input.values):
                continue

            # Try downcast
            if not disable_auto_downcast \
                and len(graph_node_input.outputs) > 0 \
                and graph_node_input.outputs[0].op not in OP_TO_BE_EXCLUDED_FROM_DOWNCAST_PROCESSING:

                ### INT64 -> INT32
                if graph_node_input.values.dtype == np.int64:
                    orig = graph_node_input.values
                    dist = graph_node_input.values.astype(np.int32)
                    if (orig == dist).all():
                        graph_node_input.values = dist

                ### Float64 -> Float32
                if graph_node_input.values.dtype == np.float64:
                    orig = graph_node_input.values
                    dist = graph_node_input.values.astype(np.float32)
                    if (orig == dist).all():
                        graph_node_input.values = dist

            constants[graph_node_input.name] = graph_node_input

    if not non_verbose:
        print(
            f'{Color.GREEN}INFO:{Color.RESET} '+
            f'Number of constant values to be studied: {len(constants)}'
        )

    constants_list = list(constants.items())

    # Aggregate list generation
    aggregate_constants_name_list = {}
    aggregate_constants_value_list = {}

    # Processing is performed only when the number of constant tensors to be processed is 2 or more
    if len(constants_list) >= 2:
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
    # Automatic aggregation of constants
    npy_file_paths = []
    for layer_name_quoted_src, layer_name_quoted_dists in aggregate_constants_name_list.items():
        i = None
        if mode == 'npy':
            # Export constant values to a numpy file
            external_file_name = ''
            if output_onnx_file_path:
                external_file_name = \
                    f"{os.path.splitext(os.path.basename(output_onnx_file_path))[0]}" + \
                    f"_exported_{layer_name_quoted_src.replace(':','_').replace(';','_').replace('/','_').replace(',','_')}.npy"
            else:
                external_file_name = \
                    f"exported_{layer_name_quoted_src.replace(':','_').replace(';','_').replace('/','_').replace(',','_')}.npy"
            np.save(
                external_file_name,
                aggregate_constants_value_list[layer_name_quoted_src].values
            )
            npy_file_paths.append(external_file_name)
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
                    if mode == 'npy' and graph_node_input.name == layer_name_quoted_src:
                        graph.nodes[graph_node_idx].inputs[input_idx] = i

    graph.cleanup().toposort()


    # Searches the entire model by the OP name specified in forced_extraction_op_names/forced_extraction_constant_names,
    # and if an OP with a matching name is found, the constant is forced to be extracted to the .npy file.

    # Constant Value Extraction
    # 1. OP Name
    constants = {}
    if forced_extraction_op_names:
        graph_nodes = [node for node in graph.nodes if node.name in forced_extraction_op_names]
        for graph_node in graph_nodes:
            for graph_node_input in graph_node.inputs:
                if not isinstance(graph_node_input, Constant):
                    continue
                if len(graph_node_input.shape) == 0:
                    continue
                if np.isscalar(graph_node_input.values):
                    continue
                constants[graph_node_input.name] = graph_node_input

    # 2. Constant Name
    if forced_extraction_constant_names:
        for graph_node in graph.nodes:
            for graph_node_input in graph_node.inputs:
                if graph_node_input.name in forced_extraction_constant_names:
                    if not isinstance(graph_node_input, Constant):
                        continue
                    if len(graph_node_input.shape) == 0:
                        continue
                    if np.isscalar(graph_node_input.values):
                        continue
                    constants[graph_node_input.name] = graph_node_input

    if not non_verbose:
        print(
            f'{Color.GREEN}INFO:{Color.RESET} '+
            f'Forced-extraction number of constant values to be studied: {len(constants)}'
        )

    constants_list = list(constants.items())

    for constant_key, constant_value in constants_list:
        i = None
        # Export constant values to a numpy file
        external_file_name = ''
        if output_onnx_file_path:
            external_file_name = \
                f"{os.path.splitext(os.path.basename(output_onnx_file_path))[0]}" + \
                f"_exported_{constant_key.replace(':','_').replace(';','_').replace('/','_').replace(',','_')}.npy"
        else:
            external_file_name = \
                f"exported_{constant_key.replace(':','_').replace(';','_').replace('/','_').replace(',','_')}.npy"
        np.save(
            external_file_name,
            constant_value.values
        )
        npy_file_paths.append(external_file_name)
        # Generate Inputs
        i = gs.Variable(
            name=external_file_name,
            dtype=constant_value.values.dtype,
            shape=constant_value.values.shape,
        )
        for graph_node_idx, graph_node in enumerate(graph.nodes):
            for input_idx, graph_node_input in enumerate(graph_node.inputs):
                if graph_node_input.name == constant_key:
                    graph.nodes[graph_node_idx].inputs[input_idx] = i
        graph.inputs.append(i)

    graph.cleanup().toposort()

    if not non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Results:')
        pprint(aggregate_constants_name_list)
        if len(npy_file_paths) > 0:
            print(f'{Color.GREEN}INFO:{Color.RESET} .npy files:')
            pprint(npy_file_paths)

    shrunken_graph = gs.export_onnx(graph)

    new_model = None
    try:
        new_model = onnx.shape_inference.infer_shapes(shrunken_graph)
    except Exception as e:
        new_model = shrunken_graph
        if not non_verbose:
            print(
                f'{Color.YELLOW}WARNING:{Color.RESET} '+
                'The input shape of the next OP does not match the output shape. '+
                'Be sure to open the .onnx file to verify the certainty of the geometry.'
            )
            tracetxt = traceback.format_exc().splitlines()[-1]
            print(f'{Color.YELLOW}WARNING:{Color.RESET} {tracetxt}')

    # Save
    if output_onnx_file_path:
        onnx.save(new_model, f'{output_onnx_file_path}')

    return new_model, npy_file_paths


def main():
    parser = ArgumentParser()
    parser.add_argument(
        'input_onnx_file_path',
        type=str,
        help='Input onnx file path.'
    )
    parser.add_argument(
        'output_onnx_file_path',
        type=str,
        help='Output onnx file path.'
    )
    parser.add_argument(
        '--mode',
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
    parser.add_argument(
        '--forced_extraction_op_names',
        type=str,
        nargs='+',
        help="\
            Extracts the constant value of the specified OP name to .npy \
            regardless of the mode specified. \
            Cannot be used with --forced_extraction_constant_names at the same time. \
            e.g. --forced_extraction_op_names aaa bbb ccc"
    )
    parser.add_argument(
        '--forced_extraction_constant_names',
        type=str,
        nargs='+',
        help="\
            Extracts the constant value of the specified Constant name to .npy \
            regardless of the mode specified. \
            Cannot be used with --forced_extraction_op_names at the same time. \
            e.g. --forced_extraction_constant_names aaa bbb ccc"
    )
    parser.add_argument(
        '--disable_auto_downcast',
        action='store_true',
        help="\
            Disables automatic downcast processing from Float64 to Float32 and INT64 to INT32. \
            Try enabling it and re-running it if you encounter type-related errors."
    )
    parser.add_argument(
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    # file existence check
    if not os.path.exists(args.input_onnx_file_path) or \
        not os.path.isfile(args.input_onnx_file_path) or \
        not os.path.splitext(args.input_onnx_file_path)[-1] == '.onnx':

        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The specified file (.onnx) does not exist. or not an onnx file. File: {args.input_onnx_file_path}'
        )
        sys.exit(1)

    forced_extraction_op_names = args.forced_extraction_op_names
    forced_extraction_constant_names = args.forced_extraction_constant_names

    if forced_extraction_op_names and forced_extraction_constant_names:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Only one of forced_extraction_op_names and forced_extraction_constant_names can be specified. '+
            f'--forced_extraction_op_names: {forced_extraction_op_names}, '+
            f'--forced_extraction_constant_names: {forced_extraction_constant_names}'
        )
        sys.exit(1)

    # Model shrink
    shrunken_graph, npy_file_paths = shrinking(
        input_onnx_file_path=args.input_onnx_file_path,
        output_onnx_file_path=args.output_onnx_file_path,
        mode=args.mode,
        forced_extraction_op_names=forced_extraction_op_names,
        forced_extraction_constant_names=forced_extraction_constant_names,
        disable_auto_downcast=args.disable_auto_downcast,
        non_verbose=args.non_verbose
    )

    if not args.non_verbose:
        print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')

if __name__ == '__main__':
    main()

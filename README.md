# scs4onnx
A very simple tool that compresses the overall size of the ONNX model by aggregating duplicate constant values as much as possible. **S**imple **C**onstant value **S**hrink for **ONNX**.

[![Downloads](https://static.pepy.tech/personalized-badge/scs4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/scs4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/scs4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/scs4onnx?color=2BAF2B)](https://pypi.org/project/scs4onnx/) [![CodeQL](https://github.com/PINTO0309/scs4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/scs4onnx/actions?query=workflow%3ACodeQL)

# Key concept

- [x] If the same constant tensor is found by scanning the entire graph for Constant values, it is aggregated into a single constant tensor.
- [x] Ignore scalar values.
- [x] Ignore variables.
- [ ] ~Finally, create a Fork of **[onnx-simplifier](https://github.com/daquexian/onnx-simplifier)** and merge this process just before the onnx file output process~ -> Temporarily abandoned because it turned out that the onnx-simplifier specification needed to be changed in a major way.
- [x] Implementation of a specification for separating the weight of a specified OP name to an external file.
- [ ] Implementation of a specification for separating the weight of a specified Constant name to an external file.
- [x] Added option to downcast from Float64 to Float32 and INT64 to INT32 to attempt size compression.
- [x] Post an issue of onnx-simplifier. [Excessive bloating of ONNX files due to over-efficient conversion of "Tile" to constants (Protocol Buffers .onnx > 2GB) #178](https://github.com/daquexian/onnx-simplifier/issues/178)
- [ ] Final work-around idea for breaking the 2GB limit, since the internal logic of onnx has a Protocol Buffers limit of 2GB checked. Recombine after optimization. Splitting and merging seems like it would be easy. For each partitioned onnx component, optimization is performed in the order of onnx-simplifier → scs4onnx to optimize the structure while keeping the buffer size to a minimum, and then the optimized components are recombined to reconstruct the whole graph. Finally, run scs4onnx again on the reconstructed, optimized overall graph to further reduce the model-wide constant.


## 1. Setup
### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U scs4onnx
```
### 1-2. Docker
```bash
### docker pull
$ docker pull pinto0309/scs4onnx:latest

### docker build
$ docker build -t pinto0309/scs4onnx:latest .

### docker run
$ docker run --rm -it -v `pwd`:/workdir pinto0309/scs4onnx:latest
$ cd /workdir
```

## 2. CLI Usage
```bash
$ scs4onnx -h

usage:
  scs4onnx [-h]
  [--mode {shrink,npy}]
  [--forced_extraction_op_names FORCED_EXTRACTION_OP_NAMES]
  [--disable_auto_downcast]
  [--non_verbose]
  input_onnx_file_path output_onnx_file_path


positional arguments:
  input_onnx_file_path
                        Input onnx file path.
  output_onnx_file_path
                        Output onnx file path.

optional arguments:
  -h, --help
                        show this help message and exit
  --mode {shrink,npy}
                        Constant Value Compression Mode.
                        shrink: Share constant values inside the model as much as possible.
                                The model size is slightly larger because
                                some shared constant values remain inside the model,
                                but performance is maximized.
                        npy:    Outputs constant values used repeatedly in the model to an
                                external file .npy. Instead of the smallest model body size,
                                the file loading overhead is greater.
                        Default: shrink
  --forced_extraction_op_names FORCED_EXTRACTION_OP_NAMES
                        Extracts the constant value of the specified OP name to .npy
                        regardless of the mode specified.
                        Specify the name of the OP, separated by commas.
                        e.g. --forced_extraction_op_names aaa,bbb,ccc
  --disable_auto_downcast
                        Disables automatic downcast processing from Float64 to Float32 and INT64
                        to INT32. Try enabling it and re-running it if you encounter type-related
                        errors.
  --non_verbose
                        Do not show all information logs. Only error logs are displayed.
```

## 3. In-script Usage
```bash
$ python
>>> from scs4onnx import shrinking
>>> help(shrinking)

Help on function shrinking in module scs4onnx.onnx_shrink_constant:

shrinking(
  input_onnx_file_path: Union[str, NoneType] = '',
  output_onnx_file_path: Union[str, NoneType] = '',
  onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None,
  mode: Union[str, NoneType] = 'shrink',
  forced_extraction_op_names: List[str] = [],
  disable_auto_downcast: Union[bool, NoneType] = False
  non_verbose: Union[bool, NoneType] = False
) -> Tuple[onnx.onnx_ml_pb2.ModelProto, str]

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.

    output_onnx_file_path: Optional[str]
        Output onnx file path.
        If output_onnx_file_path is not specified, no .onnx file is output.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    mode: Optional[str]
        Constant Value Compression Mode.
        'shrink': Share constant values inside the model as much as possible.
            The model size is slightly larger because some shared constant values remain
            inside the model, but performance is maximized.
        'npy': Outputs constant values used repeatedly in the model to an external file .npy.
            Instead of the smallest model body size, the file loading overhead is greater.
        Default: shrink

    forced_extraction_op_names: List[str]
        Extracts the constant value of the specified OP name to .npy
        regardless of the mode specified. e.g. ['aaa','bbb','ccc']

    disable_auto_downcast: Optional[bool]
        Disables automatic downcast processing from Float64 to Float32 and INT64 to INT32.
        Try enabling it and re-running it if you encounter type-related errors.
        Default: False

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.
        Default: False

    Returns
    -------
    shrunken_graph: onnx.ModelProto
        Shrunken onnx ModelProto

    npy_file_paths: List[str]
        List of paths to externally output .npy files.
        An empty list is always returned when in 'shrink' mode.
```

## 3. CLI Execution
```bash
$ scs4onnx input.onnx output.onnx --mode shrink
```
![image](https://user-images.githubusercontent.com/33194443/161479166-6929a72d-4231-4e6d-9a1f-83f9c1d70886.png)

## 4. In-script Execution
### 4-1. When an onnx file is used as input
If `output_onnx_file_path` is not specified, no .onnx file is output.
```python
from scs4onnx import shrinking

shrunk_graph, npy_file_paths = shrinking(
  input_onnx_file_path='input.onnx',
  output_onnx_file_path='output.onnx',
  mode='npy',
  non_verbose=False
)
```
![image](https://user-images.githubusercontent.com/33194443/161514266-dc24c3a4-5968-4d2b-9eba-9c7f1538f974.png)

### 4-2. When entering the onnx.ModelProto
`onnx_graph` If specified, ignore `input_onnx_file_path` and process `onnx_graph`.
```python
from scs4onnx import shrinking

shrunk_graph, npy_file_paths = shrinking(
  onnx_graph=graph,
  mode='npy',
  non_verbose=True
)
```

## 5. Sample
### 5-1. **`shrink`** mode sample
- 297.8MB -> 67.4MB (.onnx)

  ```bash
  $ scs4onnx gmflow_sintel_480x640.onnx gmflow_sintel_480x640_opt.onnx
  ```

  ![image](https://user-images.githubusercontent.com/33194443/161478190-301428b2-6ae7-4e59-bd56-d17e6a7bbe50.png)
  ![image](https://user-images.githubusercontent.com/33194443/161479347-af571cef-2162-4581-bc61-aca74bd2f387.png)

- 1.8GB -> 886.8MB (.onnx)

  ```bash
  $ scs4onnx hitnet_sf_finalpass_720x960.onnx hitnet_sf_finalpass_720x960_opt.onnx
  ```

  ![image](https://user-images.githubusercontent.com/33194443/161931579-8b656482-9608-4fe4-91c3-93465280634c.png)

- 1.8GB -> 2.1MB (.onnx) + 884.7MB (.npy)

  ```bash
  $ scs4onnx \
  hitnet_sf_finalpass_720x960.onnx \
  hitnet_sf_finalpass_720x960_opt.onnx \
  --forced_extraction_op_names GatherElements_660
  ```

  ![image](https://user-images.githubusercontent.com/33194443/161932394-da8917b6-31c0-4e79-917d-9c3109325392.png)
  ![image](https://user-images.githubusercontent.com/33194443/161933802-4a3a055c-a1cb-4b46-a89f-de8611a0671a.png)
  ![image](https://user-images.githubusercontent.com/33194443/161935942-78740546-11d5-473b-be7c-7ce4879546e7.png)

### 5-2. **`npy`** mode sample
- 297.8MB -> 21.3MB (.onnx)

  ![image](https://user-images.githubusercontent.com/33194443/161477818-9cce1821-a471-4dd5-90d2-d46f7c4576b9.png)
  ![image](https://user-images.githubusercontent.com/33194443/161479281-58df1cd6-cfcc-44d0-a4e9-234adc7e3f7a.png)

### 5-3. **`.npy`** file view
```python
$ python
>>> import numpy as np
>>> param = np.load('gmflow_sintel_480x640_shrunken_exported_1646.npy')
>>> param.shape
(8, 1200, 1200)
>>> param
array([[[   0.,    0.,    0., ...,    0.,    0.,    0.],
        [   0.,    0.,    0., ...,    0.,    0.,    0.],
        [   0.,    0.,    0., ...,    0.,    0.,    0.],
        ...,
        [-100., -100., -100., ...,    0.,    0.,    0.],
        [-100., -100., -100., ...,    0.,    0.,    0.],
        [-100., -100., -100., ...,    0.,    0.,    0.]]], dtype=float32)
```

## 6. Reference
1. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
2. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
3. https://github.com/PINTO0309/sne4onnx
4. https://github.com/PINTO0309/snd4onnx

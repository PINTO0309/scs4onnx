# scs4onnx
A very simple tool that compresses the overall size of the ONNX model by aggregating duplicate constant values as much as possible. **S**imple **C**onstant value **S**hrink for **ONNX**.

[![Downloads](https://static.pepy.tech/personalized-badge/scs4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/scs4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/scs4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/scs4onnx?color=2BAF2B)](https://pypi.org/project/scs4onnx/)

# Key concept
1. If the same constant tensor is found by scanning the entire graph for Constant values, it is aggregated into a single constant tensor.
2. Ignore scalar values.
3. Ignore variables.

## 1. Setup
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx \
&& python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
&& pip install -U scs4onnx
```

## 2. Usage
```bash
$ scs4onnx -h

usage: scs4onnx [-h] [--mode {shrink,npy}] [--non_verbose] input_onnx_file_path output_onnx_file_path

positional arguments:
  input_onnx_file_path
                        Input onnx file path.
  output_onnx_file_path
                        Output onnx file path.

optional arguments:
  -h, --help            show this help message and exit
  --mode {shrink,npy}   Constant Value Compression Mode.
                        shrink: Share constant values inside the model as much as possible.
                                The model size is slightly larger because
                                some shared constant values remain inside the model,
                                but performance is maximized.
                        npy:    Outputs constant values used repeatedly in the model to an
                                external file .npy. Instead of the smallest model body size,
                                the file loading overhead is greater.
                        Default: shrink
  --non_verbose         Do not show all information logs. Only error logs are displayed.
```

## 3. CLI Execution
```bash
$ scs4onnx input.onnx output.onnx --mode shrink
```
![image](https://user-images.githubusercontent.com/33194443/161479166-6929a72d-4231-4e6d-9a1f-83f9c1d70886.png)

## 4. In-script Execution
```python
from scs4onnx import shrinking

shrunk_graph, npy_file_paths = shrinking('input.onnx', 'output.onnx', mode='npy')
```
![image](https://user-images.githubusercontent.com/33194443/161514266-dc24c3a4-5968-4d2b-9eba-9c7f1538f974.png)

## 5. Sample
### 5-1. **`shrink`** mode sample
- 297.8MB -> 67.4MB

  ![image](https://user-images.githubusercontent.com/33194443/161478190-301428b2-6ae7-4e59-bd56-d17e6a7bbe50.png)
  ![image](https://user-images.githubusercontent.com/33194443/161479347-af571cef-2162-4581-bc61-aca74bd2f387.png)

### 5-2. **`npy`** mode sample
- 297.8MB -> 21.3MB

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

# scs4onnx
A very simple tool that compresses the overall size of the ONNX model by aggregating duplicate constant values as much as possible. **S**imple **C**onstant value **S**hrink for **ONNX**.

[![Downloads](https://static.pepy.tech/personalized-badge/scs4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/scs4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/scs4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/scs4onnx?color=2BAF2B)](https://pypi.org/project/scs4onnx/)

# key concept
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

usage: scs4onnx [-h] onnx_file_path {shrink,npy}

positional arguments:
  onnx_file_path  Input onnx file path.
  {shrink,npy}    Constant Value Compression Mode.
  
                  shrink: Share constant values inside the model as much as possible.
                          The model size is slightly larger because some shared constant values
                          remain inside the model, but performance is maximized.
                  npy:    Outputs constant values used repeatedly in the model to an external
                          file .npy. Instead of the smallest model body size, the file loading
                          overhead is greater.
                  Default: shrink

optional arguments:
  -h, --help      show this help message and exit
```

## 3. Execution
```bash
$ scs4onnx input.onnx shrink
```
## 4. Sample
### 4-1. **`shrink`** mode sample
- 297.8MB -> 67.4MB

  ![image](https://user-images.githubusercontent.com/33194443/161478190-301428b2-6ae7-4e59-bd56-d17e6a7bbe50.png)

### 4-2. **`npy`** mode sample
- 297.8MB -> 21.3MB

  ![image](https://user-images.githubusercontent.com/33194443/161477818-9cce1821-a471-4dd5-90d2-d46f7c4576b9.png)

# scs4onnx
A very simple tool that compresses the overall size of the ONNX model by aggregating duplicate constant values as much as possible. **S**imple **C**onstant value **S**hrink for **ONNX**.

# 04.04.2022 WIP

# key concept
1. If the same constant tensor is found by scanning the entire graph for Constant values, it is aggregated into a single constant tensor.
2. Ignore scalar values.
3. Ignore variables.

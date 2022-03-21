# Standard
import re

# PIP
import cupy
import torch

kernel_Softsplat_updateOutput = """
    extern "C" __global__ void kernel_Softsplat_updateOutput(
        const int n,
        const float* input,
        const float* flow,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX = ( intIndex                                                    ) % SIZE_3(output);

        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (intSoutheastY) - fltOutputY   );
        float fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY   );
        float fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * (fltOutputY    - (float) (intNortheastY));
        float fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * (fltOutputY    - (float) (intNorthwestY));

        if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) * fltNorthwest);
        }

        if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) * fltNortheast);
        }

        if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) * fltSouthwest);
        }

        if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
            atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) * fltSoutheast);
        }
    } }
"""

kernel_Softsplat_updateGradInput = """
    extern "C" __global__ void kernel_Softsplat_updateGradInput(
        const int n,
        const float* input,
        const float* flow,
        const float* grad_output,
        float* grad_input,
        float* grad_flow
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        const int intN = ( intIndex / SIZE_3(grad_input) / SIZE_2(grad_input) / SIZE_1(grad_input) ) % SIZE_0(grad_input);
        const int intC = ( intIndex / SIZE_3(grad_input) / SIZE_2(grad_input)                     ) % SIZE_1(grad_input);
        const int intY = ( intIndex / SIZE_3(grad_input)                                         ) % SIZE_2(grad_input);
        const int intX = ( intIndex                                                             ) % SIZE_3(grad_input);

        float fltGradInput = 0.0;

        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (intSoutheastY) - fltOutputY   );
        float fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - fltOutputY   );
        float fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * (fltOutputY    - (float) (intNortheastY));
        float fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * (fltOutputY    - (float) (intNorthwestY));

        if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(grad_output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(grad_output))) {
            fltGradInput += VALUE_4(grad_output, intN, intC, intNorthwestY, intNorthwestX) * fltNorthwest;
        }

        if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(grad_output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(grad_output))) {
            fltGradInput += VALUE_4(grad_output, intN, intC, intNortheastY, intNortheastX) * fltNortheast;
        }

        if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(grad_output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(grad_output))) {
            fltGradInput += VALUE_4(grad_output, intN, intC, intSouthwestY, intSouthwestX) * fltSouthwest;
        }

        if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(grad_output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(grad_output))) {
            fltGradInput += VALUE_4(grad_output, intN, intC, intSoutheastY, intSoutheastX) * fltSoutheast;
        }

        grad_input[intIndex] = fltGradInput;
    } }
"""

kernel_Softsplat_updateGradFlow = """
    extern "C" __global__ void kernel_Softsplat_updateGradFlow(
        const int n,
        const float* input,
        const float* flow,
        const float* grad_output,
        float* grad_input,
        float* grad_flow
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float fltGradFlow = 0.0;

        const int intN = ( intIndex / SIZE_3(grad_flow) / SIZE_2(grad_flow) / SIZE_1(grad_flow) ) % SIZE_0(grad_flow);
        const int intC = ( intIndex / SIZE_3(grad_flow) / SIZE_2(grad_flow)                    ) % SIZE_1(grad_flow);
        const int intY = ( intIndex / SIZE_3(grad_flow)                                       ) % SIZE_2(grad_flow);
        const int intX = ( intIndex                                                          ) % SIZE_3(grad_flow);

        float fltOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
        float fltOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

        int intNorthwestX = (int) (floor(fltOutputX));
        int intNorthwestY = (int) (floor(fltOutputY));
        int intNortheastX = intNorthwestX + 1;
        int intNortheastY = intNorthwestY;
        int intSouthwestX = intNorthwestX;
        int intSouthwestY = intNorthwestY + 1;
        int intSoutheastX = intNorthwestX + 1;
        int intSoutheastY = intNorthwestY + 1;

        float fltNorthwest = 0.0;
        float fltNortheast = 0.0;
        float fltSouthwest = 0.0;
        float fltSoutheast = 0.0;

        if (intC == 0) {
            fltNorthwest = ((float) (-1.0)) * ((float) (intSoutheastY) - fltOutputY   );
            fltNortheast = ((float) (+1.0)) * ((float) (intSouthwestY) - fltOutputY   );
            fltSouthwest = ((float) (-1.0)) * (fltOutputY    - (float) (intNortheastY));
            fltSoutheast = ((float) (+1.0)) * (fltOutputY    - (float) (intNorthwestY));

        } else if (intC == 1) {
            fltNorthwest = ((float) (intSoutheastX) - fltOutputX   ) * ((float) (-1.0));
            fltNortheast = (fltOutputX    - (float) (intSouthwestX)) * ((float) (-1.0));
            fltSouthwest = ((float) (intNortheastX) - fltOutputX   ) * ((float) (+1.0));
            fltSoutheast = (fltOutputX    - (float) (intNorthwestX)) * ((float) (+1.0));

        }

        for (int intChannel = 0; intChannel < SIZE_1(grad_output); intChannel += 1) {
            float fltInput = VALUE_4(input, intN, intChannel, intY, intX);

            if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(grad_output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(grad_output))) {
                fltGradFlow += fltInput * VALUE_4(grad_output, intN, intChannel, intNorthwestY, intNorthwestX) * fltNorthwest;
            }

            if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(grad_output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(grad_output))) {
                fltGradFlow += fltInput * VALUE_4(grad_output, intN, intChannel, intNortheastY, intNortheastX) * fltNortheast;
            }

            if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(grad_output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(grad_output))) {
                fltGradFlow += fltInput * VALUE_4(grad_output, intN, intChannel, intSouthwestY, intSouthwestX) * fltSouthwest;
            }

            if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(grad_output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(grad_output))) {
                fltGradFlow += fltInput * VALUE_4(grad_output, intN, intChannel, intSoutheastY, intSoutheastX) * fltSoutheast;
            }
        }

        grad_flow[intIndex] = fltGradFlow;
    } }
"""


def cupy_kernel(func_name, var_object):
    kernel_name = globals()[func_name]

    while True:
        match_object = re.search("(SIZE_)([0-4])(\()([^\)]*)(\))", kernel_name)

        if match_object is None:
            break

        tensor_name = match_object.group(4)
        size_list = var_object[tensor_name].size()

        index = int(match_object.group(2))
        kernel_name = kernel_name.replace(match_object.group(), str(size_list[index]))

    while True:
        match_object = re.search("(OFFSET_)([0-4])(\()([^\)]+)(\))", kernel_name)

        if match_object is None:
            break

        num_args = int(match_object.group(2))
        args_list = match_object.group(4).split(",")

        tensor_name = args_list[0]
        stride_list = var_object[tensor_name].stride()

        index_list = []
        for index in range(num_args):
            tmp = args_list[index + 1].replace("{", "(").replace("}", ")").strip()
            tmp = f"(({tmp})*{stride_list[index]})"
            index_list.append(tmp)

        kernel_name = kernel_name.replace(match_object.group(0), f'({"+".join(index_list)})')

    while True:
        match_object = re.search("(VALUE_)([0-4])(\()([^\)]+)(\))", kernel_name)

        if match_object is None:
            break

        num_args = int(match_object.group(2))
        args_list = match_object.group(4).split(",")

        tensor_name = args_list[0]
        stride_list = var_object[tensor_name].stride()

        index_list = []
        for index in range(num_args):
            tmp = args_list[index + 1].replace("{", "(").replace("}", ")").strip()
            tmp = f"(({tmp})*{stride_list[index]})"
            index_list.append(tmp)

        kernel_name = kernel_name.replace(match_object.group(0), tensor_name + f'[{"+".join(index_list)}]')

    return kernel_name


@cupy.memoize(for_each_device=True)
def cupy_launch(func_name, kernel_name):
    return cupy.cuda.compile_with_cache(kernel_name).get_function(func_name)


class SoftSplatFunc(torch.autograd.Function):
    @staticmethod
    def forward(self, input, flow):
        self.save_for_backward(input, flow)

        [num_samples, input_depth, input_height, input_width] = input.shape
        flow_depth, flow_height, flow_width = (
            flow.shape[1],
            flow.shape[2],
            flow.shape[3],
        )

        assert flow_depth == 2
        assert input_height == flow_height
        assert input_width == flow_width
        assert input.is_contiguous()
        assert flow.is_contiguous()

        output = input.new_zeros([num_samples, input_depth, input_height, input_width])

        n = output.nelement()
        cupy_launch("kernel_Softsplat_updateOutput", cupy_kernel("kernel_Softsplat_updateOutput", {"input": input, "flow": flow, "output": output},),)(
            grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
            block=tuple([512, 1, 1]),
            args=[n, input.data_ptr(), flow.data_ptr(), output.data_ptr()],
        )

        return output

    @staticmethod
    def backward(self, grad_output):
        input, flow = self.saved_tensors

        [num_samples, input_depth, input_height, input_width] = input.shape
        flow_depth, flow_height, flow_width = (
            flow.shape[1],
            flow.shape[2],
            flow.shape[3],
        )

        assert flow_depth == 2
        assert input_height == flow_height
        assert input_width == flow_width
        assert grad_output.is_contiguous()

        if self.needs_input_grad[0]:
            grad_input = input.new_zeros([num_samples, input_depth, input_height, input_width])
        else:
            grad_input = None

        if self.needs_input_grad[1]:
            grad_flow = input.new_zeros([num_samples, flow_depth, flow_height, flow_width])
        else:
            grad_flow = None

        if grad_input is not None:
            n = grad_input.nelement()
            cupy_launch(
                "kernel_Softsplat_updateGradInput",
                cupy_kernel(
                    "kernel_Softsplat_updateGradInput",
                    {
                        "input": input,
                        "flow": flow,
                        "grad_output": grad_output,
                        "grad_input": grad_input,
                        "grad_flow": grad_flow,
                    },
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n,
                    input.data_ptr(),
                    flow.data_ptr(),
                    grad_output.data_ptr(),
                    grad_input.data_ptr(),
                    None,
                ],
            )

        if grad_flow is not None:
            n = grad_flow.nelement()
            cupy_launch(
                "kernel_Softsplat_updateGradFlow",
                cupy_kernel(
                    "kernel_Softsplat_updateGradFlow",
                    {
                        "input": input,
                        "flow": flow,
                        "grad_output": grad_output,
                        "grad_input": grad_input,
                        "grad_flow": grad_flow,
                    },
                ),
            )(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    n,
                    input.data_ptr(),
                    flow.data_ptr(),
                    grad_output.data_ptr(),
                    None,
                    grad_flow.data_ptr(),
                ],
            )

        return grad_input, grad_flow


def softmax_splatting(input, flow, metric):
    assert metric is None or metric.shape[1] == 1

    input = torch.cat([input * metric.exp(), metric.exp()], dim=1)
    out = SoftSplatFunc.apply(input, flow)
    normalize = out[:, -1:, :, :]
    normalize[normalize == 0.0] = 1.0
    out = out[:, :-1, :, :] / normalize

    return out

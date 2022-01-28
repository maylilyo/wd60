# Standard
import re

# PIP
import cupy
import torch

kernel_Correlation_rearrange = '''
    extern "C" __global__ void kernel_Correlation_rearrange(
        const int n,
        const float* input,
        float* output
    ) {
      int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

      if (intIndex >= n) {
        return;
      }

      int num_samples = blockIdx.z;
      int intChannel = blockIdx.y;

      float fltValue = input[(((num_samples * SIZE_1(input)) + intChannel) * SIZE_2(input) * SIZE_3(input)) + intIndex];

      __syncthreads();

      int intPaddedY = (intIndex / SIZE_3(input)) + 4;
      int intPaddedX = (intIndex % SIZE_3(input)) + 4;
      int intRearrange = ((SIZE_3(input) + 8) * intPaddedY) + intPaddedX;

      output[(((num_samples * SIZE_1(output) * SIZE_2(output)) + intRearrange) * SIZE_1(input)) + intChannel] = fltValue;
    }
'''

kernel_Correlation_updateOutput = '''
    extern "C" __global__ void kernel_Correlation_updateOutput(
      const int n,
      const float* rbot0,
      const float* rbot1,
      float* top
    ) {
      extern __shared__ char patch_data_char[];

      float *patch_data = (float *)patch_data_char;

      // First (upper left) position of kernel upper-left corner in current center position of neighborhood in image 1
      int x1 = blockIdx.x + 4;
      int y1 = blockIdx.y + 4;
      int item = blockIdx.z;
      int ch_off = threadIdx.x;

      // Load 3D patch into shared shared memory
      for (int j = 0; j < 1; j++) { // HEIGHT
        for (int i = 0; i < 1; i++) { // WIDTH
          int ji_off = (j + i) * SIZE_3(rbot0);
          for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
            int idx1 = ((item * SIZE_1(rbot0) + y1+j) * SIZE_2(rbot0) + x1+i) * SIZE_3(rbot0) + ch;
            int idxPatchData = ji_off + ch;
            patch_data[idxPatchData] = rbot0[idx1];
          }
        }
      }

      __syncthreads();

      __shared__ float sum[32];

      // Compute correlation
      for (int top_channel = 0; top_channel < SIZE_1(top); top_channel++) {
        sum[ch_off] = 0;

        int s2o = top_channel % 9 - 4;
        int s2p = top_channel / 9 - 4;

        for (int j = 0; j < 1; j++) { // HEIGHT
          for (int i = 0; i < 1; i++) { // WIDTH
            int ji_off = (j + i) * SIZE_3(rbot0);
            for (int ch = ch_off; ch < SIZE_3(rbot0); ch += 32) { // CHANNELS
              int x2 = x1 + s2o;
              int y2 = y1 + s2p;

              int idxPatchData = ji_off + ch;
              int idx2 = ((item * SIZE_1(rbot0) + y2+j) * SIZE_2(rbot0) + x2+i) * SIZE_3(rbot0) + ch;

              sum[ch_off] += patch_data[idxPatchData] * rbot1[idx2];
            }
          }
        }

        __syncthreads();

        if (ch_off == 0) {
          float total_sum = 0;
          for (int idx = 0; idx < 32; idx++) {
            total_sum += sum[idx];
          }
          const int sumelems = SIZE_3(rbot0);
          const int index = ((top_channel*SIZE_2(top) + blockIdx.y)*SIZE_3(top))+blockIdx.x;
          top[index + item*SIZE_1(top)*SIZE_2(top)*SIZE_3(top)] = total_sum / (float)sumelems;
        }
      }
    }
'''

kernel_Correlation_updateGradFirst = '''
    # define ROUND_OFF 50000

    extern "C" __global__ void kernel_Correlation_updateGradFirst(
      const int n,
      const int num_samples,
      const float* rbot0,
      const float* rbot1,
      const float* grad_output,
      float* grad_first,
      float* grad_second
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
      int n = intIndex % SIZE_1(grad_first); // channels
      int l = (intIndex / SIZE_1(grad_first)) % SIZE_3(grad_first) + 4; // w-pos
      int m = (intIndex / SIZE_1(grad_first) / SIZE_3(grad_first)) % SIZE_2(grad_first) + 4; // h-pos

      // round_off is a trick to enable integer division with ceil, even for negative numbers
      // We use a large offset, for the inner part not to become negative.
      const int round_off = ROUND_OFF;
      const int round_off_s1 = round_off;

      // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
      int xmin = (l - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)
      int ymin = (m - 4 + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4)

      // Same here:
      int xmax = (l - 4 + round_off_s1) - round_off; // floor (l - 4)
      int ymax = (m - 4 + round_off_s1) - round_off; // floor (m - 4)

      float sum = 0;
      if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(grad_output)-1) && (ymin<=SIZE_2(grad_output)-1)) {
        xmin = max(0,xmin);
        xmax = min(SIZE_3(grad_output)-1,xmax);

        ymin = max(0,ymin);
        ymax = min(SIZE_2(grad_output)-1,ymax);

        for (int p = -4; p <= 4; p++) {
          for (int o = -4; o <= 4; o++) {
            // Get rbot1 data:
            int s2o = o;
            int s2p = p;
            int idxbot1 = ((num_samples * SIZE_1(rbot0) + (m+s2p)) * SIZE_2(rbot0) + (l+s2o)) * SIZE_3(rbot0) + n;
            float bot1tmp = rbot1[idxbot1]; // rbot1[l+s2o,m+s2p,n]

            // Index offset for grad_output in following loops:
            int op = (p+4) * 9 + (o+4); // index[o,p]
            int idxopoffset = (num_samples * SIZE_1(grad_output) + op);

            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxgrad_output = (idxopoffset * SIZE_2(grad_output) + y) * SIZE_3(grad_output) + x; // grad_output[x,y,o,p]
                sum += grad_output[idxgrad_output] * bot1tmp;
              }
            }
          }
        }
      }
      const int sumelems = SIZE_1(grad_first);
      const int bot0index = ((n * SIZE_2(grad_first)) + (m-4)) * SIZE_3(grad_first) + (l-4);
      grad_first[bot0index + num_samples*SIZE_1(grad_first)*SIZE_2(grad_first)*SIZE_3(grad_first)] = sum / (float)sumelems;
    } }
'''

kernel_Correlation_updateGradSecond = '''
    # define ROUND_OFF 50000

    extern "C" __global__ void kernel_Correlation_updateGradSecond(
      const int n,
      const int num_samples,
      const float* rbot0,
      const float* rbot1,
      const float* grad_output,
      float* grad_first,
      float* grad_second
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
      int n = intIndex % SIZE_1(grad_second); // channels
      int l = (intIndex / SIZE_1(grad_second)) % SIZE_3(grad_second) + 4; // w-pos
      int m = (intIndex / SIZE_1(grad_second) / SIZE_3(grad_second)) % SIZE_2(grad_second) + 4; // h-pos

      // round_off is a trick to enable integer division with ceil, even for negative numbers
      // We use a large offset, for the inner part not to become negative.
      const int round_off = ROUND_OFF;
      const int round_off_s1 = round_off;

      float sum = 0;
      for (int p = -4; p <= 4; p++) {
        for (int o = -4; o <= 4; o++) {
          int s2o = o;
          int s2p = p;

          //Get X,Y ranges and clamp
          // We add round_off before_s1 the int division and subtract round_off after it, to ensure the formula matches ceil behavior:
          int xmin = (l - 4 - s2o + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)
          int ymin = (m - 4 - s2p + round_off_s1 - 1) + 1 - round_off; // ceil (l - 4 - s2o)

          // Same here:
          int xmax = (l - 4 - s2o + round_off_s1) - round_off; // floor (l - 4 - s2o)
          int ymax = (m - 4 - s2p + round_off_s1) - round_off; // floor (m - 4 - s2p)

          if (xmax>=0 && ymax>=0 && (xmin<=SIZE_3(grad_output)-1) && (ymin<=SIZE_2(grad_output)-1)) {
            xmin = max(0,xmin);
            xmax = min(SIZE_3(grad_output)-1,xmax);

            ymin = max(0,ymin);
            ymax = min(SIZE_2(grad_output)-1,ymax);

            // Get rbot0 data:
            int idxbot0 = ((num_samples * SIZE_1(rbot0) + (m-s2p)) * SIZE_2(rbot0) + (l-s2o)) * SIZE_3(rbot0) + n;
            float bot0tmp = rbot0[idxbot0]; // rbot1[l+s2o,m+s2p,n]

            // Index offset for grad_output in following loops:
            int op = (p+4) * 9 + (o+4); // index[o,p]
            int idxopoffset = (num_samples * SIZE_1(grad_output) + op);

            for (int y = ymin; y <= ymax; y++) {
              for (int x = xmin; x <= xmax; x++) {
                int idxgrad_output = (idxopoffset * SIZE_2(grad_output) + y) * SIZE_3(grad_output) + x; // grad_output[x,y,o,p]
                sum += grad_output[idxgrad_output] * bot0tmp;
              }
            }
          }
        }
      }
      const int sumelems = SIZE_1(grad_second);
      const int bot1index = ((n * SIZE_2(grad_second)) + (m-4)) * SIZE_3(grad_second) + (l-4);
      grad_second[bot1index + num_samples*SIZE_1(grad_second)*SIZE_2(grad_second)*SIZE_3(grad_second)] = sum / (float)sumelems;
    } }
'''


def cupy_kernel(func_name, var_object):
    kernel_name = globals()[func_name]

    while True:
        match_object = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', kernel_name)

        if match_object is None:
            break

        tensor_name = match_object.group(4)
        size_list = var_object[tensor_name].size()

        index = int(match_object.group(2))
        kernel_name = kernel_name.replace(match_object.group(), str(size_list[index]))

    while True:
        match_object = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', kernel_name)

        if match_object is None:
            break

        num_args = int(match_object.group(2))
        args_list = match_object.group(4).split(',')

        tensor_name = args_list[0]
        stride_list = var_object[tensor_name].stride()

        index_list = []
        for index in range(num_args):
            tmp = args_list[index + 1].replace('{', '(').replace('}', ')').strip()
            tmp = f'(({tmp})*{stride_list[index]})'
            index_list.append(tmp)

        kernel_name = kernel_name.replace(match_object.group(0), f'({"+".join(index_list)})')

    return kernel_name


@cupy.memoize(for_each_device=True)
def cupy_launch(func_name, kernel_name):
    return cupy.cuda.compile_with_cache(kernel_name).get_function(func_name)


class CorrelationFunc(torch.autograd.Function):
    @staticmethod
    def forward(self, first, second):
        rbot0 = first.new_zeros([first.shape[0], first.shape[2] + 8, first.shape[3] + 8, first.shape[1]])
        rbot1 = first.new_zeros([first.shape[0], first.shape[2] + 8, first.shape[3] + 8, first.shape[1]])

        self.save_for_backward(first, second, rbot0, rbot1)

        assert(first.is_contiguous())
        assert(second.is_contiguous())

        output = first.new_zeros([first.shape[0], 81, first.shape[2], first.shape[3]])

        n = first.shape[2] * first.shape[3]
        cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
            'input': first,
            'output': rbot0
        }))(
            grid=tuple([int((n + 16 - 1) / 16), first.shape[1], first.shape[0]]),
            block=tuple([16, 1, 1]),
            args=[n, first.data_ptr(), rbot0.data_ptr()]
        )

        n = second.shape[2] * second.shape[3]
        cupy_launch('kernel_Correlation_rearrange', cupy_kernel('kernel_Correlation_rearrange', {
            'input': second,
            'output': rbot1
        }))(
            grid=tuple([int((n + 16 - 1) / 16), second.shape[1], second.shape[0]]),
            block=tuple([16, 1, 1]),
            args=[n, second.data_ptr(), rbot1.data_ptr()]
        )

        n = output.shape[1] * output.shape[2] * output.shape[3]
        cupy_launch('kernel_Correlation_updateOutput', cupy_kernel('kernel_Correlation_updateOutput', {
            'rbot0': rbot0,
            'rbot1': rbot1,
            'top': output
        }))(
            grid=tuple([output.shape[3], output.shape[2], output.shape[0]]),
            block=tuple([32, 1, 1]),
            shared_mem=first.shape[1] * 4,
            args=[n, rbot0.data_ptr(), rbot1.data_ptr(), output.data_ptr()]
        )

        return output

    @staticmethod
    def backward(self, grad_output):
        first, second, rbot0, rbot1 = self.saved_tensors

        assert(grad_output.is_contiguous())

        [num_samples, first_depth, first_height, first_width] = first.shape
        if self.needs_input_grad[0]:
            grad_first = first.new_zeros([num_samples, first_depth, first_height, first_width])
        else:
            grad_first = None

        if self.needs_input_grad[1]:
            grad_second = first.new_zeros([num_samples, first_depth, first_height, first_width])
        else:
            grad_second = None

        if grad_first is not None:
            for num_samples in range(first.shape[0]):
                n = first.shape[1] * first.shape[2] * first.shape[3]
                cupy_launch('kernel_Correlation_updateGradFirst', cupy_kernel('kernel_Correlation_updateGradFirst', {
                    'rbot0': rbot0,
                    'rbot1': rbot1,
                    'grad_output': grad_output,
                    'grad_first': grad_first,
                    'grad_second': None
                }))(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[n, num_samples, rbot0.data_ptr(), rbot1.data_ptr(), grad_output.data_ptr(), grad_first.data_ptr(), None]
                )

        if grad_second is not None:
            for num_samples in range(first.shape[0]):
                n = first.shape[1] * first.shape[2] * first.shape[3]
                cupy_launch('kernel_Correlation_updateGradSecond', cupy_kernel('kernel_Correlation_updateGradSecond', {
                    'rbot0': rbot0,
                    'rbot1': rbot1,
                    'grad_output': grad_output,
                    'grad_first': None,
                    'grad_second': grad_second
                }))(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[n, num_samples, rbot0.data_ptr(), rbot1.data_ptr(), grad_output.data_ptr(), None, grad_second.data_ptr()]
                )

        return grad_first, grad_second


def correlation(first, second):
    return CorrelationFunc.apply(first, second)

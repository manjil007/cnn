import numpy as np


class SingleConvolution:
    def __init__(self, in_shape, kernel_shape, padding=0, stride=1):
        self.in_shape = in_shape
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride
        self._weight_init()

    def _weight_init(self):
        H = int(self.in_shape - self.kernel_shape + 2 * self.padding // self.stride) + 1
        self.kernel = np.random.randint(0, 2, (self.kernel_shape, self.kernel_shape))
        self.bias = np.zeros(1)
        self.out = np.zeros((H, H))

    def _helper_forward(self, X, K, out_shape):
        """It takes the image and does the convolution on it."""
        Y = np.zeros(out_shape)
        i = 0
        k_h, k_w = K.shape
        while i < out_shape[0]:
            j = 0
            while j < out_shape[1]:
                curr_x = X[i : i + k_h, j : j + k_w]
                Y[i, j] = (curr_x * K).sum()
                j += 1
            i += 1
        return Y

    def forward(self, X):
        self.input = X
        out = self._helper_forward(X, self.kernel, self.out.shape)
        return out

    def backward(self, dl_dz):
        self.weight_gradient = self._helper_forward(
            self.input, dl_dz, self.kernel.shape
        )
        self.bias_gradient = np.sum(dl_dz)
        padding_h = self.kernel_shape // 2

        padded_dl_dz = np.pad(
            dl_dz,
            pad_width=(
                (padding_h, padding_h),
                (padding_h, padding_h),
            ),
            mode="constant",
            constant_values=0,
        )
        kernal_180 = np.rot90(self.kernel, 2)

        self.dl_dx = self._helper_forward(padded_dl_dz, kernal_180, (self.kernel.shape))

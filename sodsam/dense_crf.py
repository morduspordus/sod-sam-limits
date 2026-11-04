"""Dense CRF (Kraehenbuehl & Koltun) post-processing of a soft SOD map.

Defaults are the settings commonly used for SOD post-processing (DSS, Hou et al.):
unary -log p with p = map/255 clipped to [eps, 1-eps], a Gaussian kernel (sxy=3, compat=3)
and a bilateral kernel (sxy=60, srgb=5, compat=5), 5 mean-field iterations.
"""
import numpy as np


def dense_crf(img, sod_map, sxy_g=3, compat_g=3, sxy_b=60, srgb=5, compat_b=5, n_iter=5, eps=1e-5):
    import pydensecrf.densecrf as dcrf
    h, w = sod_map.shape
    p = np.clip(sod_map.astype(np.float64) / 255.0, eps, 1 - eps)
    unary = -np.log(np.stack([1 - p, p]).reshape(2, -1)).astype(np.float32)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(np.ascontiguousarray(unary))
    d.addPairwiseGaussian(sxy=sxy_g, compat=compat_g)
    d.addPairwiseBilateral(sxy=sxy_b, srgb=srgb, rgbim=np.ascontiguousarray(img), compat=compat_b)
    q = np.array(d.inference(n_iter)).reshape(2, h, w)
    return (q[1] * 255).round().astype(np.uint8)

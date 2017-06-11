#include <algorithm>
#include <vector>

#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/grn_layer.hpp"

namespace caffe {

template <typename Dtype>
void GRNLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();

  caffe_gpu_powx(
      count,
      bottom_data,  // x_i
      Dtype(2),
      bottom_sq_.mutable_gpu_data());  // x_i^2
  caffe_gpu_gemv(
      CblasNoTrans,
      num,
      channels,
      Dtype(1.0),
      bottom_sq_.gpu_data(),  // x_i^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      norm_sq_.mutable_gpu_data());  // \Sum x_i^2
  caffe_gpu_powx(num, norm_sq_.gpu_data(), Dtype(0.5), norm_.mutable_gpu_data()); // sqrt(\Sum x_i^2)
  caffe_gpu_gemm(
      CblasNoTrans,
      CblasTrans,
      num,
      channels,
      1,
      Dtype(1.0),
      norm_.gpu_data(),  // sqrt(\Sum x_i^2)
      summer_vec_.gpu_data(),
      Dtype(0.0),
      norm_scale_.mutable_gpu_data());
      
  caffe_gpu_div(count, bottom_data, norm_scale_.gpu_data(), top_data);
}

template <typename Dtype>
void GRNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    
    caffe_gpu_powx(
      count,
      top_data,  // y_i
      Dtype(2),
      top_sq_.mutable_gpu_data());  // y_i^2
    caffe_gpu_sub(
      count,
      ones_.gpu_data(),  // 1
      top_sq_.gpu_data(),  // y_i^2
      temp_.mutable_gpu_data()); // 1 - y_i^2

    caffe_gpu_div(count, top_diff, norm_scale_.gpu_data(), bottom_diff);

    caffe_gpu_mul(count, temp_.gpu_data(), bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GRNLayer);

}  // namespace caffe

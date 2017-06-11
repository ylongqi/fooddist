#include <algorithm>
#include <vector>

#include "caffe/layers/grn_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GRNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  top_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  bottom_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  norm_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  norm_scale_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  ones_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  for (int i = 0; i < bottom[0]->count(); ++i)
    ones_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void GRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();

  caffe_copy(count, bottom_data, top_data);

  for (int i = 0; i < num; ++i) {
    // norm = sqrt( \Sum x_i^2 )
    norm_.mutable_cpu_data()[i] = sqrt(caffe_cpu_dot(channels,
        bottom_data + (i*channels), bottom_data + (i*channels)));
    caffe_scal(channels, Dtype(1.0 / norm_.cpu_data()[i]), top_data + (i*channels));
    // for (int j = 0; j < channels; ++j) {
    //   top_data[i*channels + j] = bottom_data[i*channels + j] / Dtype(sqrt(norm_sq_.cpu_data()[i]));
    // }
  }
}

template <typename Dtype>
void GRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();

    caffe_copy(count, top_diff, bottom_diff);
    caffe_powx(
      count,
      top_data,  // y_i
      Dtype(2),
      top_sq_.mutable_cpu_data());  // y_i^2
    caffe_sub(
      count,
      ones_.cpu_data(),  // 1
      top_sq_.cpu_data(),  // y_i^2
      temp_.mutable_cpu_data()); // 1 - y_i^2

    for (int i = 0; i < num; ++i) {
      caffe_scal(channels, Dtype(1.0 / norm_.cpu_data()[i]), bottom_diff + (i*channels));
      // for (int j = 0; j < channels; ++j) {
      //   bottom_diff[i*channels + j] = bottom_diff[i*channels + j] / Dtype(sqrt(norm_sq_.cpu_data()[i]));
      // }
    }
    // xi_diff = yi_diff / norm[i] * (1 - y_i^2)
    caffe_mul(count, temp_.cpu_data(), bottom_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GRNLayer);
#endif

INSTANTIATE_CLASS(GRNLayer);
REGISTER_LAYER_CLASS(GRN);

}  // namespace caffe
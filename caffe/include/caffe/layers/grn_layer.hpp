#ifndef CAFFE_GRN_LAYER_HPP_
#define CAFFE_GRN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes @f$ y = x / ||x||_2 @f$
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the inputs @f$ x @f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the computed outputs @f$ y = x / ||x||_2 @f$
 */
template <typename Dtype>
class GRNLayer : public NeuronLayer<Dtype> {
 public:
  explicit GRNLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GRN"; }

 protected:
  /// @copydoc GRNLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the GRN inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x}
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> top_sq_;
  Blob<Dtype> bottom_sq_;
  Blob<Dtype> temp_;
  Blob<Dtype> norm_sq_;
  Blob<Dtype> norm_;
  Blob<Dtype> norm_scale_;
  Blob<Dtype> summer_vec_;
  Blob<Dtype> ones_;
};

}  // namespace caffe

#endif  // CAFFE_GRN_LAYER_HPP_

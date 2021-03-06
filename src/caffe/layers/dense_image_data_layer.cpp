#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/dense_image_data_layer.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
DenseImageDataLayer<Dtype>::~DenseImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void DenseImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.dense_image_data_param().new_height();
  const int new_width  = this->layer_param_.dense_image_data_param().new_width();
  const int crop_height = this->layer_param_.dense_image_data_param().crop_height();
  const int crop_width  = this->layer_param_.dense_image_data_param().crop_width();
  const bool is_color  = this->layer_param_.dense_image_data_param().is_color();
  string root_folder = this->layer_param_.dense_image_data_param().root_folder();
  const float scale  = this->layer_param_.dense_image_data_param().scale();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  CHECK((crop_height == 0 && crop_width == 0) ||
      (crop_height > 0 && crop_width > 0)) << "Current implementation requires "
      "crop_height and crop_width to be set at the same time.";
  CHECK((scale != 0)) << "Scale must not be 0";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.dense_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  string label_filename;
  while (infile >> filename >> label_filename) {
    lines_.push_back(std::make_pair(filename, label_filename));
  }
  const string& sourcesynth = this->layer_param_.dense_image_data_param().synth_source();
  if (sourcesynth!="")
  {
    LOG(INFO) << "Opening file " << sourcesynth;
    std::ifstream synthinfile(sourcesynth.c_str());
    string synthfilename;
    string synthlabel_filename;
    while (synthinfile >> synthfilename >> synthlabel_filename) {
      synthlines_.push_back(std::make_pair(synthfilename, synthlabel_filename));
    }
  }
  //printf("data size: %i \n", lines_.size());
  //printf("synth data size: %i \n", synthlines_.size());
  if (this->layer_param_.dense_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " real examples.";
  LOG(INFO) << "A total of " << synthlines_.size() << " synthetic examples.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.dense_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.dense_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blobs.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  // sanity check label image
  cv::Mat cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                    (float)new_height/scale, (float)new_width/scale, false, true);
  CHECK(cv_lab.channels() == 1) << "Can only handle grayscale label images";
  if (scale==1.0) 
  {
  CHECK(cv_lab.rows == height && cv_lab.cols == width) << "Input and label "
      << "image heights and widths must match";
   }

  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.dense_image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    // this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(batch_size, channels, crop_size, crop_size);
    }
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
    // similarly reshape label data blobs
    top[1]->Reshape(batch_size, 1, crop_size, crop_size);
    // this->prefetch_label_.Reshape(batch_size, 1, crop_size, crop_size);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(batch_size, 1, crop_size, crop_size);
    }
    this->transformed_label_.Reshape(1, 1, crop_size, crop_size);
  } else if (crop_height > 0 && crop_width > 0) {
    top[0]->Reshape(batch_size, channels, crop_height, crop_width);
    // this->prefetch_data_.Reshape(batch_size, channels, crop_height, crop_width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(batch_size, channels, crop_height, crop_width);
    }
    this->transformed_data_.Reshape(1, channels, crop_height, crop_width);
    // similarly reshape label data blobs
    top[1]->Reshape(batch_size, 1, crop_height, crop_width);
    // this->prefetch_label_.Reshape(batch_size, 1, crop_height, crop_width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(batch_size, 1, crop_height, crop_width);
    }
    this->transformed_label_.Reshape(1, 1, crop_height, crop_width);
  } else {
      
    top[0]->Reshape(batch_size, channels, height, width);
    // this->prefetch_data_.Reshape(batch_size, channels, height, width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(batch_size, channels, height, width);
    }
    this->transformed_data_.Reshape(1, channels, height, width);
    // similarly reshape label data blobs
    top[1]->Reshape(batch_size, 1, height/scale, width/scale);
    // this->prefetch_label_.Reshape(batch_size, 1, height, width);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->label_.Reshape(batch_size, 1, height/scale, width/scale);
    }
    this->transformed_label_.Reshape(1, 1, height/scale, width/scale);
    
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

template <typename Dtype>
void DenseImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
  if (synthlines_.size()!=0)
  {
  prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(synthlines_.begin(), synthlines_.end(), prefetch_rng);
  }
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DenseImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  // CHECK(this->prefetch_data_.count());
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  DenseImageDataParameter dense_image_data_param = this->layer_param_.dense_image_data_param();
  const int batch_size = dense_image_data_param.batch_size();
  const int new_height = dense_image_data_param.new_height();
  const int new_width = dense_image_data_param.new_width();
  const int crop_height = dense_image_data_param.crop_height();
  const int crop_width  = dense_image_data_param.crop_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = dense_image_data_param.is_color();
  const float scale = dense_image_data_param.scale();
  
  string root_folder = dense_image_data_param.root_folder();

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0 && crop_height == 0 && crop_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    //this->prefetch_data_.Reshape(1, cv_img.channels(),
    //    cv_img.rows, cv_img.cols);
    //for (int i = 0; i < this->prefetch_.size(); ++i) {
    //    this->prefetch_[i]->data_.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);
    //}
    batch->data_.Reshape(1, cv_img.channels(), cv_img.rows, cv_img.cols);

    this->transformed_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
    // this->prefetch_label_.Reshape(1, 1, cv_img.rows, cv_img.cols);
    //for (int i = 0; i < this->prefetch_.size(); ++i) {
    //    this->prefetch_[i]->label_.Reshape(1, 1, cv_img.rows, cv_img.cols);
    //}
    batch->label_.Reshape(1, 1, cv_img.rows, cv_img.cols);

    this->transformed_label_.Reshape(1, 1, cv_img.rows, cv_img.cols);
  }
  // Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  // Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  const int lines_synth_size= synthlines_.size();
  int batch_limit = batch_size/2;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    if (lines_synth_size!=0)
    {
      CHECK_GT(lines_synth_size, lines_id_);
    }
    cv::Mat cv_img;
    cv::Mat cv_lab;

    if (item_id<batch_limit && synthlines_.size()!=0)
    {
       cv_img = ReadImageToCVMat(root_folder + synthlines_[lines_id_].first,
           new_height, new_width, is_color);
       CHECK(cv_img.data) << "Could not load " << synthlines_[lines_id_].first;
       cv_lab = ReadImageToCVMat(root_folder + synthlines_[lines_id_].second,
           (float)new_height/scale, (float)new_width/scale, false, true);
       CHECK(cv_lab.data) << "Could not load " << synthlines_[lines_id_].second;
       //printf("%s %s\n", synthlines_[lines_id_].first.c_str(), synthlines_[lines_id_].second.c_str() );
    }
    else{
      cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
         new_height, new_width, is_color);
     CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
     cv_lab = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
         (float)new_height/scale, (float)new_width/scale, false, true);
     CHECK(cv_lab.data) << "Could not load " << lines_[lines_id_].second;
     //printf("%s %s\n", lines_[lines_id_].first.c_str(), lines_[lines_id_].second.c_str() );
    }
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply random horizontal mirror of images
    if (this->layer_param_.dense_image_data_param().mirror()) {
      const bool do_mirror = caffe_rng_rand() % 2;
      if (do_mirror) {
        cv::flip(cv_img,cv_img,1);
        cv::flip(cv_lab,cv_lab,1);
      }
    }
    // Apply crop
    int height = cv_img.rows;
    int width = cv_img.cols;

    int h_off = 0;
    int w_off = 0;
    if (crop_height>0 && crop_width>0) {
      h_off = caffe_rng_rand() % (height - crop_height + 1);
      w_off = caffe_rng_rand() % (width - crop_width + 1);
      cv::Rect myROI(w_off, h_off, crop_width, crop_height);
      cv_img = cv_img(myROI);
      cv_lab = cv_lab(myROI);
    }

    // Apply transformations (mirror, crop...) to the image
    // int offset = this->prefetch_data_.offset(item_id);
    //for (int i = 0; i < this->prefetch_.size(); ++i) {
    //    Dtype* prefetch_data = this->prefetch_[i]->data_.mutable_cpu_data();
    //    int offset = this->prefetch_[i]->data_.offset(item_id);
    //    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    //}
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);

    this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    // transform label the same way

    // int label_offset = this->prefetch_label_.offset(item_id);
    //for (int i = 0; i < this->prefetch_.size(); ++i) {
    //    Dtype* prefetch_label = this->prefetch_[i]->label_.mutable_cpu_data();
    //    int label_offset = this->prefetch_[i]->label_.offset(item_id);
    //    this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
    //}
    int label_offset = batch->label_.offset(item_id);
    this->transformed_label_.set_cpu_data(prefetch_label + label_offset);

    this->data_transformer_->Transform(cv_lab, &this->transformed_label_, true);
    CHECK(!this->layer_param_.transform_param().mirror() &&
        this->layer_param_.transform_param().crop_size() == 0)
        << "FIXME: Any stochastic transformation will break layer due to "
        << "the need to transform input and label images in the same way";
    trans_time += timer.MicroSeconds();

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.dense_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DenseImageDataLayer);
REGISTER_LAYER_CLASS(DenseImageData);

}  // namespace caffe
#endif //Use OpenCV

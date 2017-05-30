/*
 * This file is part of https://github.com/martinruenz/co-fusion
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#pragma once

#include "Slic.h"
#include "../FrameData.h"
#include <Eigen/Core>
#include <thread>
#include <opencv2/imgproc/imgproc.hpp>

class Model;
typedef std::list<std::shared_ptr<Model>> ModelList;
typedef ModelList::iterator ModelListIterator;

struct SegmentationResult {

    cv::Mat fullSegmentation;

  bool hasNewLabel = false;
  float depthRange;

  // Optional
  cv::Mat lowCRF;
  cv::Mat lowRGB;
  cv::Mat lowDepth;

  struct ModelData {
    // Warning order makes a difference here!
    unsigned id;
    ModelListIterator modelListIterator;

    cv::Mat lowICP;
    cv::Mat lowConf;

    unsigned superPixelCount = 0;
    float avgConfidence = 0;

    float depthMean = 0;
    float depthStd = 0;

    // The following values are only approximations:
    unsigned short top = std::numeric_limits<unsigned short>::max();
    unsigned short right = std::numeric_limits<unsigned short>::min();
    unsigned short bottom = std::numeric_limits<unsigned short>::min();
    unsigned short left = std::numeric_limits<unsigned short>::max();

    // Required for partially supported C++14 (in g++ 4.9.4)
    ModelData(unsigned t_id);

    ModelData(unsigned t_id, ModelListIterator const& t_modelListIterator, cv::Mat const& t_lowICP, cv::Mat const& t_lowConf,
              unsigned t_superPixelCount = 0, float t_avgConfidence = 0);
  };
  std::vector<ModelData> modelData;
};

class Segmentation {
 public:
  enum class METHOD { CONNECTED_COMPONENTS, TEMPORAL };

 public:
  void init(int width, int height, METHOD method);

  SegmentationResult performSegmentation(std::list<std::shared_ptr<Model>>& models, const FrameData& frame, unsigned char nextModelID,
                                         bool allowNew);

  SegmentationResult performSegmentationCRF(std::list<std::shared_ptr<Model>>& models, const FrameData& frame, unsigned char nextModelID,
                                             bool allowNew);

  /**
     * @brief denseCRF Compute a segmentation of labels based on a fully connected CRF, using icp+projection unary terms and rgb+position+depth pairwise terms
     * @param rgb input RGB image
     * @param depth input depth image
     */ /* @param icp
* @param mask
* @param scaleRGB
* @param scalePos
* @param scaleDepth
* @param bilateralWeight
* @param scaleLog
* @param offsetLog
* @param prevResult
* @param result
* @param allowNewLabel
* @return
*/
  // Static, because used by external testing application
  static cv::Mat denseCRF(cv::Mat rgb, cv::Mat depth, cv::Mat icp,
                          cv::Mat mask,  // for ground-truth comparisson
                          // cv::Mat prevResult,
                          // int numExistingLabels,
                          // const std::vector<ModelDescription>& modelDescriptions,
                          float scaleRGB, float scalePos, float scaleDepth, float bilateralWeight, float scaleLog, float offsetLog,
                          const Eigen::MatrixXf& prevResult, Eigen::MatrixXf& result, bool allowNewLabel);

  inline void setPairwiseSigmaRGB(float v) { scaleFeaturesRGB = 1.0f / v; }
  inline void setPairwiseSigmaDepth(float v) { scaleFeaturesDepth = 1.0f / v; }
  inline void setPairwiseSigmaPosition(float v) { scaleFeaturesPos = 1.0f / v; }
  inline void setPairwiseWeightAppearance(float v) { weightAppearance = v; }
  inline void setPairwiseWeightSmoothness(float v) { weightSmoothness = v; }
  inline void setUnaryThresholdNew(float v) { unaryThresholdNew = v; }
  inline void setUnaryWeightError(float v) { unaryWeightError = v; }
  inline void setUnaryKError(float v) { unaryKError = v; }
  inline void setIterationsCRF(unsigned i) { crfIterations = i; }
  inline void setNewModelMinRelativeSize(float v) { minRelSizeNew = v; }
  inline void setNewModelMaxRelativeSize(float v) { maxRelSizeNew = v; }

  // Parameters
 private:
  const float MAX_DEPTH = 100;  // 100m
  unsigned crfIterations = 10;

  // pairwise
  float scaleFeaturesRGB = 1.0f / 30;
  float scaleFeaturesDepth = 1.0f / 0.4;  // TODO: AUTO
  float scaleFeaturesPos = 1.0f / 8;
  float weightAppearance = 40;
  float weightSmoothness = 40;

  // unary
  float unaryThresholdNew = 5;
  float unaryKError = 0.01;
  float unaryWeightError = 40;
  float unaryWeightErrorBackground = 10;
  float unaryWeightConfBackground = 0.1;

  // post-processing
  float maxRelSizeNew = 0.4;
  float minRelSizeNew = 0.07;

 private:
  Slic slic;
  Eigen::MatrixXf lastRawCRF;
  METHOD method = METHOD::CONNECTED_COMPONENTS;
};
